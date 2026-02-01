from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum, auto
import random

from problem import (
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    build_mem_image,
    reference_kernel2,
)


class Resource(Enum):
    ALU = auto()
    VALU = auto()
    LOAD = auto()
    STORE = auto()
    FLOW = auto()


LIMITS = {
    Resource.ALU: 12,
    Resource.VALU: 6,
    Resource.LOAD: 2,
    Resource.STORE: 2,
    Resource.FLOW: 1,
}


@dataclass
class Op:
    id: int
    resource: Resource
    slots: int
    latency: int
    instr_type: str
    instr_data: List[Tuple]
    batch: int = -1
    round: int = -1
    stage: str = ""
    preds: List['Op'] = field(default_factory=list)
    succs: List['Op'] = field(default_factory=list)
    asap: int = 0
    alap: int = 0
    slack: int = 0
    color: int = -1

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id


class DAG:
    def __init__(self):
        self.ops: Dict[int, Op] = {}
        self.op_counter = 0
        self._topo_order = None
        self._analysis_done = False

    def add_op(self, resource: Resource, slots: int, latency: int,
               instr_type: str, instr_data: List[Tuple],
               batch: int = -1, round: int = -1, stage: str = "") -> Op:
        op = Op(
            id=self.op_counter, resource=resource, slots=slots, latency=latency,
            instr_type=instr_type, instr_data=instr_data,
            batch=batch, round=round, stage=stage,
        )
        self.ops[self.op_counter] = op
        self.op_counter += 1
        self._topo_order = None
        self._analysis_done = False
        return op

    def add_edge(self, from_op: Op, to_op: Op):
        if to_op not in from_op.succs:
            from_op.succs.append(to_op)
        if from_op not in to_op.preds:
            to_op.preds.append(from_op)
        self._topo_order = None
        self._analysis_done = False

    def topological_order(self) -> List[Op]:
        if self._topo_order is not None:
            return self._topo_order
        in_degree = {op.id: len(op.preds) for op in self.ops.values()}
        queue = [op for op in self.ops.values() if in_degree[op.id] == 0]
        result = []
        while queue:
            queue.sort(key=lambda op: op.id)
            op = queue.pop(0)
            result.append(op)
            for succ in op.succs:
                in_degree[succ.id] -= 1
                if in_degree[succ.id] == 0:
                    queue.append(succ)
        self._topo_order = result
        return result

    def analyze(self):
        if self._analysis_done:
            return
        topo = self.topological_order()
        for op in topo:
            if not op.preds:
                op.asap = 0
            else:
                op.asap = max(p.asap + p.latency for p in op.preds)
        critical_path = max(op.asap + op.latency for op in topo) if topo else 0
        for op in reversed(topo):
            if not op.succs:
                op.alap = critical_path - op.latency
            else:
                op.alap = min(s.alap - op.latency for s in op.succs)
            op.slack = op.alap - op.asap
        self._analysis_done = True


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE
        return addr

    def alloc_vec(self, name=None):
        return self.alloc_scratch(name, VLEN)

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        n_batches = batch_size // VLEN
        n_levels = forest_height + 1

        # Calculate tree structure
        # Level L has nodes from index (2^L - 1) to (2^(L+1) - 2)
        # Levels 0, 1, 2 can use cached nodes (7 total: 0, 1-2, 3-6)
        # Levels 3+ require scattered loads
        max_cached_level = min(2, forest_height)
        n_cached_nodes = (1 << (max_cached_level + 1)) - 1  # 2^(L+1) - 1
        # At level 2, indices are 3,4,5,6. We use idx < 5 to select 3,4 vs 5,6
        level_2_threshold = 5

        # Allocate contexts for each batch
        contexts = []
        for g in range(n_batches):
            ctx = {
                "idx": self.alloc_vec(f"ctx{g}_idx"),
                "val": self.alloc_vec(f"ctx{g}_val"),
                "node": self.alloc_vec(f"ctx{g}_node"),
                "tmp1": self.alloc_vec(f"ctx{g}_tmp1"),
                "tmp2": self.alloc_vec(f"ctx{g}_tmp2"),
            }
            ctx["addr"] = ctx["tmp1"]
            contexts.append(ctx)

        # Vector constants
        v_zero = self.alloc_vec("v_zero")
        v_one = self.alloc_vec("v_one")
        v_two = self.alloc_vec("v_two")
        v_threshold = self.alloc_vec("v_threshold")  # For level 2 comparison
        v_n_nodes = self.alloc_vec("v_n_nodes")
        v_hash_const1 = [self.alloc_vec(f"v_hash{i}_const1") for i in range(6)]
        # Only stages 1, 3, 5 need shift constants (even stages use multiply_add)
        v_hash_const2 = {i: self.alloc_vec(f"v_hash{i}_const2") for i in [1, 3, 5]}
        v_mult0 = self.alloc_vec("v_mult0")
        v_mult2 = self.alloc_vec("v_mult2")
        v_mult4 = self.alloc_vec("v_mult4")
        v_forest_nodes = [self.alloc_vec(f"v_forest_node_{i}") for i in range(n_cached_nodes)]

        # Scalar temps and constants
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        self.alloc_scratch("n_nodes")
        self.alloc_scratch("forest_values_p")
        self.alloc_scratch("inp_indices_p")
        self.alloc_scratch("inp_values_p")
        c_0 = self.alloc_scratch("c_0")
        c_1 = self.alloc_scratch("c_1")
        c_2 = self.alloc_scratch("c_2")
        c_threshold = self.alloc_scratch("c_threshold")
        hash_c1_addrs = [self.alloc_scratch(f"hash{i}_c1") for i in range(6)]
        # Only stages 1, 3, 5 need shift constants
        hash_c2_addrs = {i: self.alloc_scratch(f"hash{i}_c2") for i in [1, 3, 5]}
        mult_addrs = [self.alloc_scratch("mult_0"), self.alloc_scratch("mult_1")]
        forest_cache = self.alloc_scratch("forest_cache", max(8, n_cached_nodes))

        # === BUILD INIT INSTRUCTIONS ===
        # Load pointers from memory header
        self.instrs.append({"load": [("const", tmp1, 1), ("const", tmp2, 4)]})
        self.instrs.append({"load": [("load", self.scratch["n_nodes"], tmp1),
                                      ("load", self.scratch["forest_values_p"], tmp2)]})
        self.instrs.append({"load": [("const", tmp1, 5), ("const", tmp2, 6)]})
        self.instrs.append({"load": [("load", self.scratch["inp_indices_p"], tmp1),
                                      ("load", self.scratch["inp_values_p"], tmp2)]})

        # Load scalar constants
        self.instrs.append({"load": [("const", c_0, 0), ("const", c_1, 1)]})
        # Start first batch's address calculation early (FLOW is unused here)
        # inp_indices_p was loaded 2 cycles ago, so it's ready
        ctx0 = contexts[0]
        self.instrs.append({
            "load": [("const", c_2, 2), ("const", c_threshold, level_2_threshold)],
            "flow": [("add_imm", ctx0["tmp1"], self.scratch["inp_indices_p"], 0)]
        })

        # Broadcast basic vector constants + batch 0 vloads
        # Note: add_imm from previous cycle has latency 1, so addresses ready now
        self.instrs.append({
            "valu": [
                ("vbroadcast", v_two, c_2),
                ("vbroadcast", v_one, c_1),
                ("vbroadcast", v_zero, c_0),
                ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
            ],
            "load": [("vload", ctx0["idx"], ctx0["tmp1"])],  # Batch 0 idx vload
            "flow": [("add_imm", ctx0["tmp2"], self.scratch["inp_values_p"], 0)]
        })
        # Continue broadcasts + batch 0 val vload
        self.instrs.append({
            "valu": [("vbroadcast", v_threshold, c_threshold)],
            "load": [("vload", ctx0["val"], ctx0["tmp2"])]  # Batch 0 val vload
        })

        # Hash constants
        hash_const1_vals = [HASH_STAGES[i][1] for i in range(6)]
        hash_const2_vals = {i: HASH_STAGES[i][4] for i in [1, 3, 5]}  # Only odd stages
        mult_vals = [1 + (1 << 12), 1 + (1 << 5), 1 + (1 << 3)]

        # Load hash_const1 scalars (all 6 needed)
        for i in range(0, 6, 2):
            self.instrs.append({"load": [
                ("const", hash_c1_addrs[i], hash_const1_vals[i]),
                ("const", hash_c1_addrs[i + 1], hash_const1_vals[i + 1])
            ]})
        # Load hash_const2 scalars (only 1, 3, 5 needed) and broadcast const1
        self.instrs.append({
            "load": [
                ("const", hash_c2_addrs[1], hash_const2_vals[1]),
                ("const", hash_c2_addrs[3], hash_const2_vals[3])
            ],
            "valu": [
                ("vbroadcast", v_hash_const1[0], hash_c1_addrs[0]),
                ("vbroadcast", v_hash_const1[1], hash_c1_addrs[1])
            ]
        })
        self.instrs.append({
            "load": [
                ("const", hash_c2_addrs[5], hash_const2_vals[5]),
                ("const", mult_addrs[0], mult_vals[0])
            ],
            "valu": [
                ("vbroadcast", v_hash_const1[2], hash_c1_addrs[2]),
                ("vbroadcast", v_hash_const1[3], hash_c1_addrs[3])
            ]
        })
        self.instrs.append({
            "load": [
                ("const", mult_addrs[1], mult_vals[1]),
                ("const", tmp1, mult_vals[2])
            ],
            "valu": [
                ("vbroadcast", v_hash_const1[4], hash_c1_addrs[4]),
                ("vbroadcast", v_hash_const1[5], hash_c1_addrs[5]),
                ("vbroadcast", v_hash_const2[1], hash_c2_addrs[1]),
                ("vbroadcast", v_hash_const2[3], hash_c2_addrs[3]),
            ]
        })
        self.instrs.append({
            "load": [("vload", forest_cache, self.scratch["forest_values_p"])],
            "valu": [
                ("vbroadcast", v_hash_const2[5], hash_c2_addrs[5]),
            ]
        })

        # Reuse hash0_c1 for c_8 (VLEN constant for incremental addressing)
        c_8 = hash_c1_addrs[0]

        # Broadcast mult constants and forest nodes
        self.instrs.append({"valu": [
            ("vbroadcast", v_mult0, mult_addrs[0]),
            ("vbroadcast", v_mult2, mult_addrs[1]),
            ("vbroadcast", v_mult4, tmp1),
            ("vbroadcast", v_forest_nodes[n_cached_nodes - 1], forest_cache + n_cached_nodes - 1),
        ]})
        self.instrs.append({
            "valu": [("vbroadcast", v_forest_nodes[i], forest_cache + i) for i in range(min(6, n_cached_nodes - 1))],
            "load": [("const", c_8, VLEN)]
        })

        # === BUILD DAG ===
        dag = DAG()
        prev_init_addr_ops = None
        batch_final_ops = [None] * n_batches

        def build_hash(batch: int, rnd: int, prev: Op) -> tuple:
            ctx = contexts[batch]

            h0 = dag.add_op(Resource.VALU, 1, 1, "valu",
                [("multiply_add", ctx["val"], ctx["val"], v_mult0, v_hash_const1[0])],
                batch, rnd, "H0")
            dag.add_edge(prev, h0)

            h1a = dag.add_op(Resource.VALU, 1, 1, "valu",
                [(HASH_STAGES[1][0], ctx["tmp1"], ctx["val"], v_hash_const1[1])],
                batch, rnd, "H1A")
            dag.add_edge(h0, h1a)

            h1b = dag.add_op(Resource.VALU, 1, 1, "valu",
                [(HASH_STAGES[1][3], ctx["tmp2"], ctx["val"], v_hash_const2[1])],
                batch, rnd, "H1B")
            dag.add_edge(h0, h1b)

            h1 = dag.add_op(Resource.VALU, 1, 1, "valu",
                [(HASH_STAGES[1][2], ctx["val"], ctx["tmp1"], ctx["tmp2"])],
                batch, rnd, "H1")
            dag.add_edge(h1a, h1)
            dag.add_edge(h1b, h1)

            h2v = dag.add_op(Resource.VALU, 1, 1, "valu",
                [("multiply_add", ctx["val"], ctx["val"], v_mult2, v_hash_const1[2])],
                batch, rnd, "H2V")
            dag.add_edge(h1, h2v)

            h2i = dag.add_op(Resource.VALU, 1, 1, "valu",
                [("multiply_add", ctx["idx"], ctx["idx"], v_two, v_one)],
                batch, rnd, "H2I")
            dag.add_edge(prev, h2i)

            h3a = dag.add_op(Resource.VALU, 1, 1, "valu",
                [(HASH_STAGES[3][0], ctx["tmp1"], ctx["val"], v_hash_const1[3])],
                batch, rnd, "H3A")
            dag.add_edge(h2v, h3a)

            h3b = dag.add_op(Resource.VALU, 1, 1, "valu",
                [(HASH_STAGES[3][3], ctx["tmp2"], ctx["val"], v_hash_const2[3])],
                batch, rnd, "H3B")
            dag.add_edge(h2v, h3b)

            h3 = dag.add_op(Resource.VALU, 1, 1, "valu",
                [(HASH_STAGES[3][2], ctx["val"], ctx["tmp1"], ctx["tmp2"])],
                batch, rnd, "H3")
            dag.add_edge(h3a, h3)
            dag.add_edge(h3b, h3)

            h4 = dag.add_op(Resource.VALU, 1, 1, "valu",
                [("multiply_add", ctx["val"], ctx["val"], v_mult4, v_hash_const1[4])],
                batch, rnd, "H4")
            dag.add_edge(h3, h4)

            h5a = dag.add_op(Resource.VALU, 1, 1, "valu",
                [(HASH_STAGES[5][0], ctx["tmp1"], ctx["val"], v_hash_const1[5])],
                batch, rnd, "H5A")
            dag.add_edge(h4, h5a)

            h5b = dag.add_op(Resource.VALU, 1, 1, "valu",
                [(HASH_STAGES[5][3], ctx["tmp2"], ctx["val"], v_hash_const2[5])],
                batch, rnd, "H5B")
            dag.add_edge(h4, h5b)

            h5 = dag.add_op(Resource.VALU, 1, 1, "valu",
                [(HASH_STAGES[5][2], ctx["val"], ctx["tmp1"], ctx["tmp2"])],
                batch, rnd, "H5")
            dag.add_edge(h5a, h5)
            dag.add_edge(h5b, h5)

            return h5, h2i

        def build_scattered(batch: int, rnd: int, level: int, val_ready: Op, idx_ready: Op) -> tuple:
            ctx = contexts[batch]
            addr = dag.add_op(Resource.ALU, 8, 1, "alu",
                [("+", ctx["addr"] + i, self.scratch["forest_values_p"], ctx["idx"] + i) for i in range(8)],
                batch, rnd, "ADDR")
            dag.add_edge(idx_ready, addr)

            load_ops = []
            for ld in range(4):
                i = ld * 2
                load_op = dag.add_op(Resource.LOAD, 2, 2, "load",
                    [("load", ctx["node"] + i, ctx["addr"] + i),
                     ("load", ctx["node"] + i + 1, ctx["addr"] + i + 1)],
                    batch, rnd, f"LOAD{ld}")
                dag.add_edge(addr, load_op)
                load_ops.append(load_op)

            xor = dag.add_op(Resource.ALU, 8, 1, "alu",
                [("^", ctx["val"] + i, ctx["val"] + i, ctx["node"] + i) for i in range(8)],
                batch, rnd, "XOR")
            for load_op in load_ops:
                dag.add_edge(load_op, xor)
            dag.add_edge(val_ready, xor)

            hash_end, h2i = build_hash(batch, rnd, xor)
            dag.add_edge(idx_ready, h2i)

            idx_and = dag.add_op(Resource.VALU, 1, 1, "valu",
                [("&", ctx["tmp1"], ctx["val"], v_one)],
                batch, rnd, "IDX_AND")
            dag.add_edge(hash_end, idx_and)

            idx_add = dag.add_op(Resource.VALU, 1, 1, "valu",
                [("+", ctx["idx"], ctx["idx"], ctx["tmp1"])],
                batch, rnd, "IDX_ADD")
            dag.add_edge(idx_and, idx_add)
            dag.add_edge(h2i, idx_add)

            if level == forest_height:
                ovf_lt = dag.add_op(Resource.VALU, 1, 1, "valu",
                    [("<", ctx["tmp1"], ctx["idx"], v_n_nodes)],
                    batch, rnd, "OVF_LT")
                dag.add_edge(idx_add, ovf_lt)
                ovf_sel = dag.add_op(Resource.FLOW, 1, 1, "flow",
                    [("vselect", ctx["idx"], ctx["tmp1"], ctx["idx"], v_zero)],
                    batch, rnd, "OVF_SEL")
                dag.add_edge(ovf_lt, ovf_sel)
                return ovf_sel, hash_end, ovf_sel
            return idx_add, hash_end, idx_add

        def build_non_scattered(batch: int, rnd: int, level: int, val_ready: Op, idx_ready: Op) -> tuple:
            ctx = contexts[batch]
            fn = v_forest_nodes

            if level == 0:
                xor = dag.add_op(Resource.ALU, 8, 1, "alu",
                    [("^", ctx["val"] + i, ctx["val"] + i, fn[0] + i) for i in range(8)],
                    batch, rnd, "XOR")
                dag.add_edge(val_ready, xor)
                xor_end = xor
            elif level == 1:
                and1 = dag.add_op(Resource.VALU, 1, 1, "valu",
                    [("&", ctx["tmp1"], ctx["idx"], v_one)],
                    batch, rnd, "AND1")
                dag.add_edge(idx_ready, and1)
                vsel1 = dag.add_op(Resource.FLOW, 1, 1, "flow",
                    [("vselect", ctx["tmp2"], ctx["tmp1"], fn[1], fn[2])],
                    batch, rnd, "VSEL1")
                dag.add_edge(and1, vsel1)
                xor = dag.add_op(Resource.ALU, 8, 1, "alu",
                    [("^", ctx["val"] + i, ctx["val"] + i, ctx["tmp2"] + i) for i in range(8)],
                    batch, rnd, "XOR")
                dag.add_edge(vsel1, xor)
                dag.add_edge(val_ready, xor)
                xor_end = xor
            else:  # level == 2
                and1 = dag.add_op(Resource.VALU, 1, 1, "valu",
                    [("&", ctx["tmp1"], ctx["idx"], v_one)],
                    batch, rnd, "AND1")
                dag.add_edge(idx_ready, and1)
                vsel1 = dag.add_op(Resource.FLOW, 1, 1, "flow",
                    [("vselect", ctx["tmp2"], ctx["tmp1"], fn[3], fn[4])],
                    batch, rnd, "VSEL1")
                dag.add_edge(and1, vsel1)
                vsel2 = dag.add_op(Resource.FLOW, 1, 1, "flow",
                    [("vselect", ctx["node"], ctx["tmp1"], fn[5], fn[6])],
                    batch, rnd, "VSEL2")
                dag.add_edge(vsel1, vsel2)
                lt = dag.add_op(Resource.VALU, 1, 1, "valu",
                    [("<", ctx["tmp1"], ctx["idx"], v_threshold)],
                    batch, rnd, "LT")
                dag.add_edge(vsel2, lt)
                dag.add_edge(idx_ready, lt)
                vsel3 = dag.add_op(Resource.FLOW, 1, 1, "flow",
                    [("vselect", ctx["tmp2"], ctx["tmp1"], ctx["tmp2"], ctx["node"])],
                    batch, rnd, "VSEL3")
                dag.add_edge(lt, vsel3)
                xor = dag.add_op(Resource.ALU, 8, 1, "alu",
                    [("^", ctx["val"] + i, ctx["val"] + i, ctx["tmp2"] + i) for i in range(8)],
                    batch, rnd, "XOR")
                dag.add_edge(vsel3, xor)
                dag.add_edge(val_ready, xor)
                xor_end = xor

            hash_end, h2i = build_hash(batch, rnd, xor_end)
            dag.add_edge(idx_ready, h2i)

            idx_and = dag.add_op(Resource.VALU, 1, 1, "valu",
                [("&", ctx["tmp1"], ctx["val"], v_one)],
                batch, rnd, "IDX_AND")
            dag.add_edge(hash_end, idx_and)
            idx_add = dag.add_op(Resource.VALU, 1, 1, "valu",
                [("+", ctx["idx"], ctx["idx"], ctx["tmp1"])],
                batch, rnd, "IDX_ADD")
            dag.add_edge(idx_and, idx_add)
            dag.add_edge(h2i, idx_add)
            return idx_add, hash_end, idx_add

        # Build init loads for each batch
        # Batch 0's vloads are done in init phase, so we skip DAG ops for it
        for batch in range(n_batches):
            ctx = contexts[batch]
            if batch == 0:
                # Batch 0 addresses and vloads done in init - create dummy ops for chaining
                # These ops produce no instructions but allow batch 1 to chain properly
                addr_idx = dag.add_op(Resource.FLOW, 0, 0, "flow", [], batch, -1, "INIT_ADDR_IDX_SKIP")
                addr_val = dag.add_op(Resource.FLOW, 0, 0, "flow", [], batch, -1, "INIT_ADDR_VAL_SKIP")
                # The vloads for batch 0 are done in init, data is already in ctx0 idx/val
                # We don't need load_idx/load_val ops, just set prev_idx_update appropriately
                prev_init_addr_ops = (addr_idx, addr_val)
                prev_h5 = None
                # Create a dummy op to represent that batch 0's data is ready
                # This needs to have the right timing - the init vloads finish around cycle 8
                load_val = dag.add_op(Resource.LOAD, 0, 0, "load", [], batch, -1, "INIT_LOAD_VAL_SKIP")
                prev_idx_update = load_val
            else:
                prev_ctx = contexts[batch - 1]
                prev_addr_idx, prev_addr_val = prev_init_addr_ops
                addr_idx = dag.add_op(Resource.ALU, 1, 1, "alu",
                    [("+", ctx["tmp1"], prev_ctx["tmp1"], c_8)],
                    batch, -1, "INIT_ADDR_IDX")
                dag.add_edge(prev_addr_idx, addr_idx)
                addr_val = dag.add_op(Resource.ALU, 1, 1, "alu",
                    [("+", ctx["tmp2"], prev_ctx["tmp2"], c_8)],
                    batch, -1, "INIT_ADDR_VAL")
                dag.add_edge(prev_addr_val, addr_val)

                load_idx = dag.add_op(Resource.LOAD, 1, 2, "load",
                    [("vload", ctx["idx"], ctx["tmp1"])],
                    batch, -1, "INIT_LOAD_IDX")
                dag.add_edge(addr_idx, load_idx)
                load_val = dag.add_op(Resource.LOAD, 1, 2, "load",
                    [("vload", ctx["val"], ctx["tmp2"])],
                    batch, -1, "INIT_LOAD_VAL")
                dag.add_edge(addr_val, load_val)

                prev_init_addr_ops = (addr_idx, addr_val)
                prev_h5 = None
                prev_idx_update = load_val

            # Build rounds
            for rnd in range(rounds):
                level = rnd % n_levels
                # Always wait for idx_update to ensure overflow check completes
                val_ready = prev_idx_update

                if level >= 3:
                    prev_end, h5, idx_update = build_scattered(batch, rnd, level, val_ready, prev_idx_update)
                else:
                    prev_end, h5, idx_update = build_non_scattered(batch, rnd, level, val_ready, prev_idx_update)

                prev_h5 = h5
                prev_idx_update = idx_update

            batch_final_ops[batch] = prev_end

        # Build stores
        prev_store_addr_ops = None
        for batch in range(n_batches):
            ctx = contexts[batch]
            if batch == 0:
                addr_idx = dag.add_op(Resource.FLOW, 1, 1, "flow",
                    [("add_imm", ctx["node"], self.scratch["inp_indices_p"], 0)],
                    batch, -2, "STORE_ADDR_IDX")
                addr_val = dag.add_op(Resource.FLOW, 1, 1, "flow",
                    [("add_imm", ctx["node"] + 1, self.scratch["inp_values_p"], 0)],
                    batch, -2, "STORE_ADDR_VAL")
                dag.add_edge(batch_final_ops[batch], addr_idx)
                dag.add_edge(batch_final_ops[batch], addr_val)
            else:
                prev_ctx = contexts[batch - 1]
                prev_addr_idx, prev_addr_val = prev_store_addr_ops
                addr_idx = dag.add_op(Resource.ALU, 1, 1, "alu",
                    [("+", ctx["node"], prev_ctx["node"], c_8)],
                    batch, -2, "STORE_ADDR_IDX")
                dag.add_edge(batch_final_ops[batch], addr_idx)
                dag.add_edge(prev_addr_idx, addr_idx)
                addr_val = dag.add_op(Resource.ALU, 1, 1, "alu",
                    [("+", ctx["node"] + 1, prev_ctx["node"] + 1, c_8)],
                    batch, -2, "STORE_ADDR_VAL")
                dag.add_edge(batch_final_ops[batch], addr_val)
                dag.add_edge(prev_addr_val, addr_val)

            store_idx = dag.add_op(Resource.STORE, 1, 1, "store",
                [("vstore", ctx["node"], ctx["idx"])],
                batch, -2, "STORE_IDX")
            dag.add_edge(addr_idx, store_idx)
            store_val = dag.add_op(Resource.STORE, 1, 1, "store",
                [("vstore", ctx["node"] + 1, ctx["val"])],
                batch, -2, "STORE_VAL")
            dag.add_edge(addr_val, store_val)
            prev_store_addr_ops = (addr_idx, addr_val)

        # === SCHEDULE DAG ===
        dag.analyze()
        colors: Dict[int, List[Op]] = defaultdict(list)
        color_usage: Dict[int, Dict[Resource, int]] = defaultdict(lambda: defaultdict(int))

        def can_schedule(op: Op, cycle: int) -> bool:
            for pred in op.preds:
                if pred.color < 0 or cycle < pred.color + pred.latency:
                    return False
            if color_usage[cycle][op.resource] + op.slots > LIMITS[op.resource]:
                return False
            return True

        def schedule_op(op: Op, cycle: int):
            op.color = cycle
            colors[cycle].append(op)
            color_usage[cycle][op.resource] += op.slots

        def get_stage(round_num):
            if round_num == -1:
                return 0
            elif round_num == -2:
                return 4
            elif round_num <= 5:
                return 1
            elif round_num <= 10:
                return 2
            else:
                return 3

        ready = set()
        scheduled = set()
        for op in dag.ops.values():
            if not op.preds:
                ready.add(op.id)
        cycle = 0
        max_cycles = 10000

        while len(scheduled) < len(dag.ops) and cycle < max_cycles:
            candidates = []
            for op_id in ready:
                op = dag.ops[op_id]
                if not can_schedule(op, cycle):
                    continue
                stage = get_stage(op.round)
                valu_boost = 1000 if op.resource == Resource.VALU else 0
                batch_group = op.batch // 6 if op.batch >= 0 else -1
                batch_in_group = op.batch % 6 if op.batch >= 0 else 0
                if op.round >= 14:
                    priority = (-valu_boost, op.round, batch_group, batch_in_group, stage, op.slack, op.id)
                else:
                    priority = (-valu_boost, stage, batch_group, batch_in_group, op.round, op.slack, op.id)
                candidates.append((priority, op))

            candidates.sort(key=lambda x: x[0])
            for _, op in candidates:
                if op.id in scheduled:
                    continue
                if can_schedule(op, cycle):
                    schedule_op(op, cycle)
                    scheduled.add(op.id)
                    ready.discard(op.id)
                    for succ in op.succs:
                        if succ.id not in scheduled and succ.id not in ready:
                            if all(p.color >= 0 for p in succ.preds):
                                ready.add(succ.id)
            cycle += 1

        # === GENERATE INSTRUCTIONS ===
        if colors:
            max_color = max(colors.keys())
            for color in range(max_color + 1):
                instr = defaultdict(list)
                for op in colors.get(color, []):
                    for data in op.instr_data:
                        instr[op.instr_type].append(data)
                cycle_instr = {}
                if instr["valu"]:
                    cycle_instr["valu"] = instr["valu"][:6]
                if instr["alu"]:
                    cycle_instr["alu"] = instr["alu"][:12]
                if instr["load"]:
                    cycle_instr["load"] = instr["load"][:2]
                if instr["store"]:
                    cycle_instr["store"] = instr["store"][:2]
                if instr["flow"]:
                    cycle_instr["flow"] = instr["flow"][:1]
                self.instrs.append(cycle_instr)
            while self.instrs and not self.instrs[-1]:
                self.instrs.pop()



BASELINE = 147734


def do_kernel_test(forest_height: int, rounds: int, batch_size: int, seed: int = 123, trace: bool = False):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES, value_trace=value_trace, trace=trace)
    for ref_mem in reference_kernel2(mem, value_trace):
        machine.run()

    inp_indices_p = mem[5]
    inp_values_p = mem[6]
    idx_ok = machine.mem[inp_indices_p:inp_indices_p + len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p + len(inp.indices)]
    val_ok = machine.mem[inp_values_p:inp_values_p + len(inp.values)] == ref_mem[inp_values_p:inp_values_p + len(inp.values)]
    if idx_ok and val_ok:
        print("OUTPUT CORRECT!")
    else:
        print("OUTPUT MISMATCH!")
        if not idx_ok:
            print("  IDX mismatch")
        if not val_ok:
            print("  VAL mismatch")

    print("CYCLES:", machine.cycle)
    print("Speedup:", BASELINE / machine.cycle)
    return machine.cycle


if __name__ == "__main__":
    do_kernel_test(10, 16, 256, trace=False)
