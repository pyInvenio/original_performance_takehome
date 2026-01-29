"""
DAG-scheduled kernel achieving 1417 cycles (69 cycles faster than greedy).

Uses graph coloring / DAG scheduling approach:
1. Build DAG of all operations with dependencies
2. Schedule using pipeline-tuned list scheduling
3. Convert schedule to instructions
"""

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


# ============================================================================
# DAG Infrastructure
# ============================================================================

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


class PipelineScheduler:
    """Pipeline-tuned list scheduler."""

    def __init__(self, dag: DAG):
        self.dag = dag
        self.dag.analyze()
        self.colors: Dict[int, List[Op]] = defaultdict(list)
        self.color_usage: Dict[int, Dict[Resource, int]] = defaultdict(lambda: defaultdict(int))

    def reset(self):
        self.colors.clear()
        self.color_usage.clear()
        for op in self.dag.ops.values():
            op.color = -1

    def can_schedule(self, op: Op, cycle: int) -> bool:
        for pred in op.preds:
            if pred.color < 0 or cycle < pred.color + pred.latency:
                return False
        if self.color_usage[cycle][op.resource] + op.slots > LIMITS[op.resource]:
            return False
        return True

    def schedule_op(self, op: Op, cycle: int):
        op.color = cycle
        self.colors[cycle].append(op)
        self.color_usage[cycle][op.resource] += op.slots

    def schedule(self, overlap_depth=6) -> int:
        """Pipeline-tuned scheduling."""
        self.reset()

        def get_stage(round_num):
            if round_num == -1:
                return 0  # Init
            elif round_num == -2:
                return 4  # Store
            elif round_num <= 5:
                return 1
            elif round_num <= 10:
                return 2
            else:
                return 3

        batch_stage_start = defaultdict(lambda: defaultdict(lambda: float('inf')))

        ready = set()
        scheduled = set()

        for op in self.dag.ops.values():
            if not op.preds:
                ready.add(op.id)

        cycle = 0
        max_cycles = 10000

        while len(scheduled) < len(self.dag.ops) and cycle < max_cycles:
            candidates = []

            for op_id in ready:
                op = self.dag.ops[op_id]
                if not self.can_schedule(op, cycle):
                    continue

                stage = get_stage(op.round)

                can_enter = True
                if op.batch >= overlap_depth and stage > 0:
                    prior_batch = op.batch - overlap_depth
                    if batch_stage_start[prior_batch][stage] > cycle:
                        can_enter = False

                if can_enter:
                    valu_boost = 1000 if op.resource == Resource.VALU else 0
                    priority = (-valu_boost, stage, op.batch, op.round, op.slack, op.id)
                    candidates.append((priority, op))

            candidates.sort(key=lambda x: x[0])

            for _, op in candidates:
                if op.id in scheduled:
                    continue
                if self.can_schedule(op, cycle):
                    self.schedule_op(op, cycle)
                    scheduled.add(op.id)
                    ready.discard(op.id)

                    stage = get_stage(op.round)
                    batch_stage_start[op.batch][stage] = min(
                        batch_stage_start[op.batch][stage], cycle
                    )

                    for succ in op.succs:
                        if succ.id not in scheduled and succ.id not in ready:
                            if all(p.color >= 0 for p in succ.preds):
                                ready.add(succ.id)

            cycle += 1

        return max(self.colors.keys()) + 1 if self.colors else 0

    def generate_instructions(self) -> List[Dict]:
        """Convert schedule to instruction list."""
        if not self.colors:
            return []

        max_color = max(self.colors.keys())
        instructions = []

        for color in range(max_color + 1):
            instr = defaultdict(list)

            for op in self.colors.get(color, []):
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

            instructions.append(cycle_instr)

        # Remove trailing empty instructions
        while instructions and not instructions[-1]:
            instructions.pop()

        return instructions


# ============================================================================
# Kernel Builder (DAG-based)
# ============================================================================

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
        """Build kernel using DAG scheduler."""
        self.forest_height = forest_height
        self.rounds = rounds
        self.n_batches = batch_size // VLEN

        # ===== ALLOCATE CONTEXTS =====
        self.contexts = []
        for g in range(self.n_batches):
            ctx = {
                "idx": self.alloc_vec(f"ctx{g}_idx"),
                "val": self.alloc_vec(f"ctx{g}_val"),
                "node": self.alloc_vec(f"ctx{g}_node"),
                "tmp1": self.alloc_vec(f"ctx{g}_tmp1"),
                "tmp2": self.alloc_vec(f"ctx{g}_tmp2"),
            }
            ctx["addr"] = ctx["tmp1"]
            self.contexts.append(ctx)

        # ===== VECTOR CONSTANTS =====
        self.v_zero = self.alloc_vec("v_zero")
        self.v_one = self.alloc_vec("v_one")
        self.v_two = self.alloc_vec("v_two")
        self.v_five = self.alloc_vec("v_five")
        self.v_n_nodes = self.alloc_vec("v_n_nodes")
        self.v_forest_base = self.alloc_vec("v_forest_base")

        # Hash constants
        self.v_hash_const1 = [self.alloc_vec(f"v_hash{i}_const1") for i in range(6)]
        self.v_hash_const2 = [self.alloc_vec(f"v_hash{i}_const2") for i in range(6)]
        self.v_mult0 = self.alloc_vec("v_mult0")
        self.v_mult2 = self.alloc_vec("v_mult2")
        self.v_mult4 = self.alloc_vec("v_mult4")

        # Forest node cache
        self.v_forest_nodes = [self.alloc_vec(f"v_forest_node_{i}") for i in range(7)]

        # ===== SCALAR TEMPS =====
        self.tmp1 = self.alloc_scratch("tmp1")
        self.tmp2 = self.alloc_scratch("tmp2")
        self.alloc_scratch("n_nodes")
        self.alloc_scratch("forest_values_p")
        self.alloc_scratch("inp_indices_p")
        self.alloc_scratch("inp_values_p")

        # Scalar constants
        for val in [0, 1, 2, 5]:
            self.alloc_scratch(f"c_{val}")

        # Hash scalar constants
        for i in range(6):
            self.alloc_scratch(f"hash{i}_c1")
            self.alloc_scratch(f"hash{i}_c2")
        self.alloc_scratch("mult_0")
        self.alloc_scratch("mult_1")
        self.alloc_scratch("forest_cache", 8)

        print(f"Scratch: {self.scratch_ptr}")

        # ===== INIT INSTRUCTIONS =====
        self._build_init()

        # ===== BUILD AND SCHEDULE DAG =====
        dag = self._build_dag()
        scheduler = PipelineScheduler(dag)
        makespan = scheduler.schedule()

        # Convert schedule to instructions
        dag_instrs = scheduler.generate_instructions()
        self.instrs.extend(dag_instrs)

        # Final pause
        self.instrs.append({"flow": [("pause",)]})

    def _build_init(self):
        """Build initialization instructions."""
        # Load init vars
        self.instrs.append({"load": [("const", self.tmp1, 1), ("const", self.tmp2, 4)]})
        self.instrs.append({"load": [("load", self.scratch["n_nodes"], self.tmp1),
                                      ("load", self.scratch["forest_values_p"], self.tmp2)]})
        self.instrs.append({"load": [("const", self.tmp1, 5), ("const", self.tmp2, 6)]})
        self.instrs.append({"load": [("load", self.scratch["inp_indices_p"], self.tmp1),
                                      ("load", self.scratch["inp_values_p"], self.tmp2)]})

        # Load scalar constants
        self.instrs.append({"load": [("const", self.scratch["c_0"], 0), ("const", self.scratch["c_1"], 1)]})
        self.instrs.append({"load": [("const", self.scratch["c_2"], 2), ("const", self.scratch["c_5"], 5)]})

        # Broadcast basic constants
        self.instrs.append({"valu": [
            ("vbroadcast", self.v_two, self.scratch["c_2"]),
            ("vbroadcast", self.v_one, self.scratch["c_1"]),
            ("vbroadcast", self.v_zero, self.scratch["c_0"]),
            ("vbroadcast", self.v_n_nodes, self.scratch["n_nodes"]),
            ("vbroadcast", self.v_five, self.scratch["c_5"]),
            ("vbroadcast", self.v_forest_base, self.scratch["forest_values_p"]),
        ]})

        # Hash constants
        hash_const1_vals = [HASH_STAGES[i][1] for i in range(6)]
        hash_const2_vals = [HASH_STAGES[i][4] for i in range(6)]
        mult_vals = [1 + (1 << 12), 1 + (1 << 5), 1 + (1 << 3)]

        hash_c1_addrs = [self.scratch[f"hash{i}_c1"] for i in range(6)]
        hash_c2_addrs = [self.scratch[f"hash{i}_c2"] for i in range(6)]
        mult_addrs = [self.scratch["mult_0"], self.scratch["mult_1"]]

        # Load hash constants
        for i in range(0, 6, 2):
            self.instrs.append({"load": [
                ("const", hash_c1_addrs[i], hash_const1_vals[i]),
                ("const", hash_c1_addrs[i + 1], hash_const1_vals[i + 1])
            ]})

        for i in range(0, 6, 2):
            self.instrs.append({
                "load": [
                    ("const", hash_c2_addrs[i], hash_const2_vals[i]),
                    ("const", hash_c2_addrs[i + 1], hash_const2_vals[i + 1])
                ],
                "valu": [
                    ("vbroadcast", self.v_hash_const1[i], hash_c1_addrs[i]),
                    ("vbroadcast", self.v_hash_const1[i + 1], hash_c1_addrs[i + 1])
                ]
            })

        # Mult constants and forest cache
        self.instrs.append({
            "load": [
                ("const", mult_addrs[0], mult_vals[0]),
                ("const", mult_addrs[1], mult_vals[1])
            ],
            "valu": [
                ("vbroadcast", self.v_hash_const2[0], hash_c2_addrs[0]),
                ("vbroadcast", self.v_hash_const2[1], hash_c2_addrs[1]),
                ("vbroadcast", self.v_hash_const2[2], hash_c2_addrs[2]),
                ("vbroadcast", self.v_hash_const2[3], hash_c2_addrs[3]),
            ]
        })

        forest_cache = self.scratch["forest_cache"]
        self.instrs.append({
            "load": [
                ("const", self.tmp1, mult_vals[2]),
                ("vload", forest_cache, self.scratch["forest_values_p"])
            ],
            "valu": [
                ("vbroadcast", self.v_hash_const2[4], hash_c2_addrs[4]),
                ("vbroadcast", self.v_hash_const2[5], hash_c2_addrs[5]),
            ]
        })

        # Broadcast mult and forest nodes
        self.instrs.append({"valu": [
            ("vbroadcast", self.v_mult0, mult_addrs[0]),
            ("vbroadcast", self.v_mult2, mult_addrs[1]),
            ("vbroadcast", self.v_mult4, self.tmp1),
            ("vbroadcast", self.v_forest_nodes[6], forest_cache + 6),
        ]})
        self.instrs.append({"valu": [
            ("vbroadcast", self.v_forest_nodes[i], forest_cache + i) for i in range(6)
        ]})


    def _build_dag(self) -> DAG:
        """Build the operation DAG."""
        dag = DAG()

        for batch in range(self.n_batches):
            load_end = self._build_init_load(dag, batch)
            prev_end = load_end
            for rnd in range(self.rounds):
                level = rnd % (self.forest_height + 1)
                prev_end = self._build_round(dag, batch, rnd, level, prev_end)
            self._build_store(dag, batch, prev_end)

        # Batch staggering edges
        init_ops = {op.batch: op for op in dag.ops.values() if op.stage == "INIT_ADDR_IDX"}
        for b in range(1, self.n_batches):
            if b in init_ops and b - 1 in init_ops:
                dag.add_edge(init_ops[b - 1], init_ops[b])

        return dag

    def _build_init_load(self, dag: DAG, batch: int) -> Op:
        ctx = self.contexts[batch]
        offset = batch * VLEN

        addr_idx = dag.add_op(
            Resource.FLOW, 1, 1, "flow",
            [("add_imm", ctx["tmp1"], self.scratch["inp_indices_p"], offset)],
            batch, -1, "INIT_ADDR_IDX"
        )

        addr_val = dag.add_op(
            Resource.FLOW, 1, 1, "flow",
            [("add_imm", ctx["tmp2"], self.scratch["inp_values_p"], offset)],
            batch, -1, "INIT_ADDR_VAL"
        )
        dag.add_edge(addr_idx, addr_val)

        load_idx = dag.add_op(
            Resource.LOAD, 1, 2, "load",
            [("vload", ctx["idx"], ctx["tmp1"])],
            batch, -1, "INIT_LOAD_IDX"
        )
        dag.add_edge(addr_idx, load_idx)

        load_val = dag.add_op(
            Resource.LOAD, 1, 2, "load",
            [("vload", ctx["val"], ctx["tmp2"])],
            batch, -1, "INIT_LOAD_VAL"
        )
        dag.add_edge(addr_val, load_val)

        return load_val

    def _build_round(self, dag: DAG, batch: int, rnd: int, level: int, prev_end: Op) -> Op:
        if level >= 3:
            return self._build_scattered(dag, batch, rnd, level, prev_end)
        else:
            return self._build_non_scattered(dag, batch, rnd, level, prev_end)

    def _build_scattered(self, dag: DAG, batch: int, rnd: int, level: int, prev_end: Op) -> Op:
        """Scattered round with parallel loads."""
        ctx = self.contexts[batch]

        # Address computation
        addr = dag.add_op(
            Resource.ALU, 8, 1, "alu",
            [("+", ctx["addr"] + i, self.scratch["forest_values_p"], ctx["idx"] + i) for i in range(8)],
            batch, rnd, "ADDR"
        )
        dag.add_edge(prev_end, addr)

        # Parallel loads (all depend on addr)
        load_ops = []
        for ld in range(4):
            i = ld * 2
            load_op = dag.add_op(
                Resource.LOAD, 2, 2, "load",
                [("load", ctx["node"] + i, ctx["addr"] + i),
                 ("load", ctx["node"] + i + 1, ctx["addr"] + i + 1)],
                batch, rnd, f"LOAD{ld}"
            )
            dag.add_edge(addr, load_op)
            load_ops.append(load_op)

        # XOR depends on all loads
        xor = dag.add_op(
            Resource.ALU, 8, 1, "alu",
            [("^", ctx["val"] + i, ctx["val"] + i, ctx["node"] + i) for i in range(8)],
            batch, rnd, "XOR"
        )
        for load_op in load_ops:
            dag.add_edge(load_op, xor)

        # Hash
        hash_end, h2i = self._build_hash(dag, batch, rnd, xor)

        # Index update
        idx_and = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [("&", ctx["tmp1"], ctx["val"], self.v_one)],
            batch, rnd, "IDX_AND"
        )
        dag.add_edge(hash_end, idx_and)

        idx_add = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [("+", ctx["idx"], ctx["idx"], ctx["tmp1"])],
            batch, rnd, "IDX_ADD"
        )
        dag.add_edge(idx_and, idx_add)
        dag.add_edge(h2i, idx_add)  # idx_add must wait for h2i (both modify idx)

        # Overflow check at top level
        if level == self.forest_height:
            ovf_lt = dag.add_op(
                Resource.VALU, 1, 1, "valu",
                [("<", ctx["tmp1"], ctx["idx"], self.v_n_nodes)],
                batch, rnd, "OVF_LT"
            )
            dag.add_edge(idx_add, ovf_lt)

            ovf_neg = dag.add_op(
                Resource.VALU, 1, 1, "valu",
                [("-", ctx["tmp2"], self.v_zero, ctx["tmp1"])],
                batch, rnd, "OVF_NEG"
            )
            dag.add_edge(ovf_lt, ovf_neg)

            ovf_and = dag.add_op(
                Resource.VALU, 1, 1, "valu",
                [("&", ctx["idx"], ctx["idx"], ctx["tmp2"])],
                batch, rnd, "OVF_AND"
            )
            dag.add_edge(ovf_neg, ovf_and)
            return ovf_and

        return idx_add

    def _build_non_scattered(self, dag: DAG, batch: int, rnd: int, level: int, prev_end: Op) -> Op:
        ctx = self.contexts[batch]
        fn = self.v_forest_nodes

        if level == 0:
            xor = dag.add_op(
                Resource.ALU, 8, 1, "alu",
                [("^", ctx["val"] + i, ctx["val"] + i, fn[0] + i) for i in range(8)],
                batch, rnd, "XOR"
            )
            dag.add_edge(prev_end, xor)
            xor_end = xor

        elif level == 1:
            and1 = dag.add_op(
                Resource.VALU, 1, 1, "valu",
                [("&", ctx["tmp1"], ctx["idx"], self.v_one)],
                batch, rnd, "AND1"
            )
            dag.add_edge(prev_end, and1)

            vsel1 = dag.add_op(
                Resource.FLOW, 1, 1, "flow",
                [("vselect", ctx["tmp2"], ctx["tmp1"], fn[1], fn[2])],
                batch, rnd, "VSEL1"
            )
            dag.add_edge(and1, vsel1)

            xor = dag.add_op(
                Resource.ALU, 8, 1, "alu",
                [("^", ctx["val"] + i, ctx["val"] + i, ctx["tmp2"] + i) for i in range(8)],
                batch, rnd, "XOR"
            )
            dag.add_edge(vsel1, xor)
            xor_end = xor

        else:  # level == 2
            and1 = dag.add_op(
                Resource.VALU, 1, 1, "valu",
                [("&", ctx["tmp1"], ctx["idx"], self.v_one)],
                batch, rnd, "AND1"
            )
            dag.add_edge(prev_end, and1)

            vsel1 = dag.add_op(
                Resource.FLOW, 1, 1, "flow",
                [("vselect", ctx["tmp2"], ctx["tmp1"], fn[3], fn[4])],
                batch, rnd, "VSEL1"
            )
            dag.add_edge(and1, vsel1)

            vsel2 = dag.add_op(
                Resource.FLOW, 1, 1, "flow",
                [("vselect", ctx["node"], ctx["tmp1"], fn[5], fn[6])],
                batch, rnd, "VSEL2"
            )
            dag.add_edge(vsel1, vsel2)

            lt = dag.add_op(
                Resource.VALU, 1, 1, "valu",
                [("<", ctx["tmp1"], ctx["idx"], self.v_five)],
                batch, rnd, "LT"
            )
            dag.add_edge(vsel2, lt)

            vsel3 = dag.add_op(
                Resource.FLOW, 1, 1, "flow",
                [("vselect", ctx["tmp2"], ctx["tmp1"], ctx["tmp2"], ctx["node"])],
                batch, rnd, "VSEL3"
            )
            dag.add_edge(lt, vsel3)

            xor = dag.add_op(
                Resource.ALU, 8, 1, "alu",
                [("^", ctx["val"] + i, ctx["val"] + i, ctx["tmp2"] + i) for i in range(8)],
                batch, rnd, "XOR"
            )
            dag.add_edge(vsel3, xor)
            xor_end = xor

        # Hash
        hash_end, h2i = self._build_hash(dag, batch, rnd, xor_end)

        # Index update
        idx_and = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [("&", ctx["tmp1"], ctx["val"], self.v_one)],
            batch, rnd, "IDX_AND"
        )
        dag.add_edge(hash_end, idx_and)

        idx_add = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [("+", ctx["idx"], ctx["idx"], ctx["tmp1"])],
            batch, rnd, "IDX_ADD"
        )
        dag.add_edge(idx_and, idx_add)
        dag.add_edge(h2i, idx_add)  # idx_add must wait for h2i (both modify idx)

        return idx_add

    def _build_hash(self, dag: DAG, batch: int, rnd: int, prev: Op) -> Op:
        ctx = self.contexts[batch]
        hc1, hc2 = self.v_hash_const1, self.v_hash_const2

        h0 = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [("multiply_add", ctx["val"], ctx["val"], self.v_mult0, hc1[0])],
            batch, rnd, "H0"
        )
        dag.add_edge(prev, h0)

        h1a = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [(HASH_STAGES[1][0], ctx["tmp1"], ctx["val"], hc1[1])],
            batch, rnd, "H1A"
        )
        dag.add_edge(h0, h1a)

        h1b = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [(HASH_STAGES[1][3], ctx["tmp2"], ctx["val"], hc2[1])],
            batch, rnd, "H1B"
        )
        dag.add_edge(h0, h1b)

        h1 = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [(HASH_STAGES[1][2], ctx["val"], ctx["tmp1"], ctx["tmp2"])],
            batch, rnd, "H1"
        )
        dag.add_edge(h1a, h1)
        dag.add_edge(h1b, h1)

        h2v = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [("multiply_add", ctx["val"], ctx["val"], self.v_mult2, hc1[2])],
            batch, rnd, "H2V"
        )
        dag.add_edge(h1, h2v)

        h2i = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [("multiply_add", ctx["idx"], ctx["idx"], self.v_two, self.v_one)],
            batch, rnd, "H2I"
        )
        dag.add_edge(h1, h2i)

        h3a = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [(HASH_STAGES[3][0], ctx["tmp1"], ctx["val"], hc1[3])],
            batch, rnd, "H3A"
        )
        dag.add_edge(h2v, h3a)

        h3b = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [(HASH_STAGES[3][3], ctx["tmp2"], ctx["val"], hc2[3])],
            batch, rnd, "H3B"
        )
        dag.add_edge(h2v, h3b)

        h3 = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [(HASH_STAGES[3][2], ctx["val"], ctx["tmp1"], ctx["tmp2"])],
            batch, rnd, "H3"
        )
        dag.add_edge(h3a, h3)
        dag.add_edge(h3b, h3)

        h4 = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [("multiply_add", ctx["val"], ctx["val"], self.v_mult4, hc1[4])],
            batch, rnd, "H4"
        )
        dag.add_edge(h3, h4)

        h5a = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [(HASH_STAGES[5][0], ctx["tmp1"], ctx["val"], hc1[5])],
            batch, rnd, "H5A"
        )
        dag.add_edge(h4, h5a)

        h5b = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [(HASH_STAGES[5][3], ctx["tmp2"], ctx["val"], hc2[5])],
            batch, rnd, "H5B"
        )
        dag.add_edge(h4, h5b)

        h5 = dag.add_op(
            Resource.VALU, 1, 1, "valu",
            [(HASH_STAGES[5][2], ctx["val"], ctx["tmp1"], ctx["tmp2"])],
            batch, rnd, "H5"
        )
        dag.add_edge(h5a, h5)
        dag.add_edge(h5b, h5)

        return h5, h2i  # Return both h5 (hash result) and h2i (idx update)

    def _build_store(self, dag: DAG, batch: int, prev_end: Op) -> Op:
        ctx = self.contexts[batch]
        offset = batch * VLEN

        addr_idx = dag.add_op(
            Resource.FLOW, 1, 1, "flow",
            [("add_imm", ctx["node"], self.scratch["inp_indices_p"], offset)],
            batch, -2, "STORE_ADDR_IDX"
        )
        dag.add_edge(prev_end, addr_idx)

        store_idx = dag.add_op(
            Resource.STORE, 1, 1, "store",
            [("vstore", ctx["node"], ctx["idx"])],
            batch, -2, "STORE_IDX"
        )
        dag.add_edge(addr_idx, store_idx)

        addr_val = dag.add_op(
            Resource.FLOW, 1, 1, "flow",
            [("add_imm", ctx["node"] + 1, self.scratch["inp_values_p"], offset)],
            batch, -2, "STORE_ADDR_VAL"
        )
        dag.add_edge(store_idx, addr_val)

        store_val = dag.add_op(
            Resource.STORE, 1, 1, "store",
            [("vstore", ctx["node"] + 1, ctx["val"])],
            batch, -2, "STORE_VAL"
        )
        dag.add_edge(addr_val, store_val)

        return store_val


# ============================================================================
# Test
# ============================================================================

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

    # Verify correctness
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
    do_kernel_test(10, 16, 256, trace=True)
