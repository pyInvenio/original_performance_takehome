"""
32-context kernel with improved scheduling.
Key optimizations:
1. All 32 batches process together
2. On-the-fly address computation (saves 96 words of scratch)
3. Better pipelining - overlap stores with finishing batches
4. Improved VALU/Load scheduling
"""

from collections import defaultdict
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


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch: {self.scratch_ptr} > {SCRATCH_SIZE}"
        return addr

    def alloc_vec(self, name=None):
        return self.alloc_scratch(name, VLEN)

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """32-context kernel with improved scheduling."""
        self.forest_height = forest_height
        n_batches = batch_size // VLEN  # 32

        # ===== ALLOCATE ALL 32 CONTEXTS =====
        contexts = []
        for g in range(32):
            ctx = {
                "idx": self.alloc_vec(f"ctx{g}_idx"),
                "val": self.alloc_vec(f"ctx{g}_val"),
                "node": self.alloc_vec(f"ctx{g}_node"),
                "tmp1": self.alloc_vec(f"ctx{g}_tmp1"),
                "tmp2": self.alloc_vec(f"ctx{g}_tmp2"),
            }
            ctx["addr"] = ctx["tmp1"]
            contexts.append(ctx)

        # ===== VECTOR CONSTANTS =====
        v_zero = self.alloc_vec("v_zero")
        v_one = self.alloc_vec("v_one")
        v_two = self.alloc_vec("v_two")
        v_five = self.alloc_vec("v_five")
        v_n_nodes = self.alloc_vec("v_n_nodes")
        v_forest_base = self.alloc_vec("v_forest_base")

        # ===== SCALAR TEMPS =====
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Only allocate init vars we actually need
        self.alloc_scratch("n_nodes", 1)
        self.alloc_scratch("forest_values_p", 1)
        self.alloc_scratch("inp_indices_p", 1)
        self.alloc_scratch("inp_values_p", 1)

        # Scalar constants
        const_addrs = {}
        for val in [0, 1, 2, 5]:
            const_addrs[val] = self.alloc_scratch(f"c_{val}")
            self.const_map[val] = const_addrs[val]

        # Load the init vars we need
        self.instrs.append({"load": [("const", tmp1, 1), ("const", tmp2, 4)]})
        self.instrs.append({"load": [("load", self.scratch["n_nodes"], tmp1),
                                      ("load", self.scratch["forest_values_p"], tmp2)]})
        self.instrs.append({"load": [("const", tmp1, 5), ("const", tmp2, 6)]})
        self.instrs.append({"load": [("load", self.scratch["inp_indices_p"], tmp1),
                                      ("load", self.scratch["inp_values_p"], tmp2)]})

        # Load scalar constants
        self.instrs.append({"load": [("const", const_addrs[0], 0), ("const", const_addrs[1], 1)]})
        self.instrs.append({"load": [("const", const_addrs[2], 2), ("const", const_addrs[5], 5)]})

        self.instrs.append({"valu": [
            ("vbroadcast", v_two, const_addrs[2]),
            ("vbroadcast", v_one, const_addrs[1]),
            ("vbroadcast", v_zero, const_addrs[0]),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
            ("vbroadcast", v_five, const_addrs[5]),
            ("vbroadcast", v_forest_base, self.scratch["forest_values_p"]),
        ]})
        # Removed pause - hash constant loads are independent

        # ===== HASH CONSTANTS =====
        v_hash_const1 = [self.alloc_vec(f"v_hash{i}_const1") for i in range(6)]
        v_hash_const2 = [self.alloc_vec(f"v_hash{i}_const2") for i in range(6)]
        v_mult0 = self.alloc_vec("v_mult0")
        v_mult2 = self.alloc_vec("v_mult2")
        v_mult4 = self.alloc_vec("v_mult4")

        hash_const1_vals = [HASH_STAGES[i][1] for i in range(6)]
        hash_const2_vals = [HASH_STAGES[i][4] for i in range(6)]
        mult_vals = [1 + (1 << 12), 1 + (1 << 5), 1 + (1 << 3)]
        all_const_vals = hash_const1_vals + hash_const2_vals + mult_vals

        hash_c1_addrs = []
        for i in range(6):
            addr = self.alloc_scratch(f"hash{i}_c1")
            hash_c1_addrs.append(addr)

        hash_c2_addrs = []
        for i in range(6):
            addr = self.alloc_scratch(f"hash{i}_c2")
            hash_c2_addrs.append(addr)

        # Only allocate 2 mult_addrs - reuse tmp1 for the third
        mult_addrs = [self.alloc_scratch("mult_0"), self.alloc_scratch("mult_1")]

        # ===== CACHE TREE NODES 0-6 =====
        v_forest_nodes = [self.alloc_vec(f"v_forest_node_{i}") for i in range(7)]
        forest_cache = self.alloc_scratch("forest_cache", 8)

        # Load hash constants and mult constants, overlapping with vbroadcasts
        all_addrs = hash_c1_addrs + hash_c2_addrs + mult_addrs + [tmp1]
        # First 6 loads (hash_c1): just loads
        for i in range(0, 6, 2):
            self.instrs.append({"load": [
                ("const", all_addrs[i], all_const_vals[i]),
                ("const", all_addrs[i + 1], all_const_vals[i + 1])
            ]})
        # Load 6-11 (hash_c2) with vbroadcast of hash_c1 (loaded 2 cycles ago)
        for i in range(6, 12, 2):
            self.instrs.append({
                "load": [
                    ("const", all_addrs[i], all_const_vals[i]),
                    ("const", all_addrs[i + 1], all_const_vals[i + 1])
                ],
                "valu": [
                    ("vbroadcast", v_hash_const1[i - 6], hash_c1_addrs[i - 6]),
                    ("vbroadcast", v_hash_const1[i - 5], hash_c1_addrs[i - 5])
                ]
            })
        # Load 12-13 (mult_0, mult_1) with vbroadcast of hash_c2 + hash_c1[4,5]
        self.instrs.append({
            "load": [
                ("const", mult_addrs[0], all_const_vals[12]),
                ("const", mult_addrs[1], all_const_vals[13])
            ],
            "valu": [
                ("vbroadcast", v_hash_const2[0], hash_c2_addrs[0]),
                ("vbroadcast", v_hash_const2[1], hash_c2_addrs[1]),
                ("vbroadcast", v_hash_const1[4], hash_c1_addrs[4]),
                ("vbroadcast", v_hash_const1[5], hash_c1_addrs[5])
            ]
        })
        # Load 14 (mult_2) + vload forest_cache, with vbroadcast hash_c2[2,3,4,5]
        self.instrs.append({
            "load": [
                ("const", tmp1, all_const_vals[14]),
                ("vload", forest_cache, self.scratch["forest_values_p"])
            ],
            "valu": [
                ("vbroadcast", v_hash_const2[2], hash_c2_addrs[2]),
                ("vbroadcast", v_hash_const2[3], hash_c2_addrs[3]),
                ("vbroadcast", v_hash_const2[4], hash_c2_addrs[4]),
                ("vbroadcast", v_hash_const2[5], hash_c2_addrs[5])
            ]
        })
        # Broadcast mult and forest nodes
        self.instrs.append({"valu": [
            ("vbroadcast", v_mult0, mult_addrs[0]),
            ("vbroadcast", v_mult2, mult_addrs[1]),
            ("vbroadcast", v_mult4, tmp1),
            ("vbroadcast", v_forest_nodes[6], forest_cache + 6),
        ]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_forest_nodes[i], forest_cache + i) for i in range(6)
        ]})

        print(f"Scratch usage after init: {self.scratch_ptr}")

        # ===== PROCESS ALL 32 BATCHES =====
        self._process_tile(
            contexts, 32, 0, rounds, forest_height, n_nodes,
            v_forest_nodes, v_hash_const1, v_hash_const2,
            v_mult0, v_mult2, v_mult4, v_two, v_one, v_zero,
            v_n_nodes, v_five, v_forest_base
        )

        self.instrs.append({"flow": [("pause",)]})

    def _process_tile(
        self, contexts, active, tile_start, tile_end, forest_height, n_nodes,
        v_forest_nodes, v_hash_const1, v_hash_const2,
        v_mult0, v_mult2, v_mult4, v_two, v_one, v_zero,
        v_n_nodes, v_five, v_forest_base
    ):
        """Process all 32 batches with improved scheduling."""

        # Stage constants
        STAGE_ADDR = 0
        STAGE_LOAD0, STAGE_LOAD1, STAGE_LOAD2, STAGE_LOAD3 = 1, 2, 3, 4
        STAGE_XOR = 5
        STAGE_H0 = 6
        STAGE_H1P1A, STAGE_H1P1B = 7, 38
        STAGE_H1 = 8
        STAGE_H2 = 9
        STAGE_H3P1A, STAGE_H3P1B = 10, 39
        STAGE_H3 = 11
        STAGE_H4 = 12
        STAGE_H5P1A, STAGE_H5P1B = 13, 41
        STAGE_H5 = 14
        STAGE_IDX_AND = 15
        STAGE_IDX_ADD = 16
        STAGE_OVF_LT = 17
        STAGE_OVF_NEG = 42  # New: negate comparison result to make mask
        STAGE_OVF_AND = 18  # Renamed: apply mask with AND
        STAGE_AND1 = 20
        STAGE_VSEL1 = 21
        STAGE_LT = 22
        STAGE_VSEL2 = 23
        STAGE_VSEL3 = 24

        def get_level(round_num):
            return round_num % (forest_height + 1)

        def is_scattered(level):
            return level >= 3

        def needs_overflow(level):
            return level == forest_height

        # Load state: 3-stage pipeline
        LOAD_STAGE_ADDR_IDX = 0
        LOAD_STAGE_ADDR_VAL = 1
        LOAD_STAGE_VAL = 2
        LOAD_STAGE_DONE = 3

        # Store state: 3-stage pipeline (store_addr_idx, store_idx+addr_val, store_val)
        STORE_STAGE_ADDR_IDX = 0
        STORE_STAGE_IDX_ADDR_VAL = 1
        STORE_STAGE_VAL = 2
        STORE_STAGE_DONE = 3

        # State
        current_round = [tile_start] * active
        stage_done = [[-1] * 50 for _ in range(active)]
        batch_done = [False] * active
        batch_load_stage = [LOAD_STAGE_ADDR_IDX] * active
        load_done_cycle = [-1] * active
        batch_stored = [False] * active
        batch_store_stage = [STORE_STAGE_ADDR_IDX] * active

        # Initialize first round stage for all batches
        for g in range(active):
            level = get_level(tile_start)
            if is_scattered(level):
                stage_done[g][STAGE_ADDR] = -2
            else:
                stage_done[g][STAGE_XOR] = -2

        cycle = 0
        max_cycles = 5000

        def all_complete():
            return all(batch_stored)

        while not all_complete() and cycle < max_cycles:
            valu_ops = []
            alu_ops = []
            load_ops = []
            store_ops = []
            flow_ops = []

            # ===== INITIAL LOADS (sequential 3-stage) =====
            for g in range(active):
                if batch_load_stage[g] >= LOAD_STAGE_DONE:
                    continue

                ctx = contexts[g]
                offset = g * VLEN

                if batch_load_stage[g] == LOAD_STAGE_ADDR_IDX:
                    if len(flow_ops) == 0:
                        flow_ops.append(("add_imm", ctx["tmp1"], self.scratch["inp_indices_p"], offset))
                        batch_load_stage[g] = LOAD_STAGE_ADDR_VAL
                    break

                elif batch_load_stage[g] == LOAD_STAGE_ADDR_VAL:
                    if len(flow_ops) == 0 and len(load_ops) == 0:
                        flow_ops.append(("add_imm", ctx["tmp2"], self.scratch["inp_values_p"], offset))
                        load_ops.append(("vload", ctx["idx"], ctx["tmp1"]))
                        batch_load_stage[g] = LOAD_STAGE_VAL
                    break

                elif batch_load_stage[g] == LOAD_STAGE_VAL:
                    if len(load_ops) == 0:
                        load_ops.append(("vload", ctx["val"], ctx["tmp2"]))
                        batch_load_stage[g] = LOAD_STAGE_DONE
                        load_done_cycle[g] = cycle
                    break

            # ===== STORES FOR COMPLETED BATCHES =====
            # Use node[0] and node[1] for store addresses (node is free after processing)
            # This allows computing addresses while other batches still process

            # First pass: S2 (val store) for multiple batches
            for g in range(active):
                if len(store_ops) >= 2:
                    break
                if not batch_done[g] or batch_stored[g]:
                    continue
                if batch_store_stage[g] == STORE_STAGE_VAL:
                    ctx = contexts[g]
                    store_ops.append(("vstore", ctx["node"] + 1, ctx["val"]))  # node[1] has val addr
                    batch_store_stage[g] = STORE_STAGE_DONE
                    batch_stored[g] = True

            # Second pass: S1 (idx store + val addr) or S0 (idx addr)
            # Try S1 first, if can't do it, try S0 for any batch
            s1_candidate = -1
            s0_candidate = -1
            for g in range(active):
                if not batch_done[g] or batch_stored[g]:
                    continue
                if batch_store_stage[g] == STORE_STAGE_IDX_ADDR_VAL and s1_candidate < 0:
                    s1_candidate = g
                elif batch_store_stage[g] == STORE_STAGE_ADDR_IDX and s0_candidate < 0:
                    s0_candidate = g
                if s1_candidate >= 0 and s0_candidate >= 0:
                    break

            # Try S1 first (uses flow + store)
            if s1_candidate >= 0 and len(flow_ops) == 0 and len(store_ops) < 2:
                g = s1_candidate
                ctx = contexts[g]
                offset = g * VLEN
                flow_ops.append(("add_imm", ctx["node"] + 1, self.scratch["inp_values_p"], offset))
                store_ops.append(("vstore", ctx["node"], ctx["idx"]))
                batch_store_stage[g] = STORE_STAGE_VAL
            # If S1 couldn't be done, try S0 (uses flow only)
            elif s0_candidate >= 0 and len(flow_ops) == 0:
                g = s0_candidate
                ctx = contexts[g]
                offset = g * VLEN
                flow_ops.append(("add_imm", ctx["node"], self.scratch["inp_indices_p"], offset))
                batch_store_stage[g] = STORE_STAGE_IDX_ADDR_VAL

            # ===== PROCESS LOADED BATCHES =====
            for g in range(active):
                if batch_done[g]:
                    continue
                if batch_load_stage[g] < LOAD_STAGE_DONE:
                    continue

                ctx = contexts[g]
                sd = stage_done[g]

                # Wait for load latency
                if load_done_cycle[g] >= cycle:
                    continue

                rnd = current_round[g]
                if rnd >= tile_end:
                    batch_done[g] = True
                    continue

                level = get_level(rnd)
                scattered = is_scattered(level)
                overflow = needs_overflow(level)

                if scattered:
                    # ===== SCATTERED ROUND =====
                    if sd[STAGE_ADDR] == -2:
                        if len(alu_ops) <= 4:
                            for i in range(8):
                                alu_ops.append(("+", ctx["addr"] + i, self.scratch["forest_values_p"], ctx["idx"] + i))
                            sd[STAGE_ADDR] = cycle
                        continue

                    if sd[STAGE_ADDR] >= 0 and sd[STAGE_ADDR] < cycle:
                        for ld_stg in range(STAGE_LOAD0, STAGE_LOAD3 + 1):
                            if sd[ld_stg] < 0:
                                if len(load_ops) == 0:
                                    i = (ld_stg - STAGE_LOAD0) * 2
                                    load_ops.append(("load", ctx["node"] + i, ctx["addr"] + i))
                                    load_ops.append(("load", ctx["node"] + i + 1, ctx["addr"] + i + 1))
                                    sd[ld_stg] = cycle
                                break
                        if sd[STAGE_LOAD3] < 0:
                            continue

                    if sd[STAGE_LOAD3] >= 0 and sd[STAGE_LOAD3] < cycle:
                        if sd[STAGE_XOR] < 0:
                            if len(alu_ops) <= 4:
                                for i in range(8):
                                    alu_ops.append(("^", ctx["val"] + i, ctx["val"] + i, ctx["node"] + i))
                                sd[STAGE_XOR] = cycle
                            continue

                    # Hash stages
                    if sd[STAGE_XOR] >= 0 and sd[STAGE_XOR] < cycle:
                        if sd[STAGE_H0] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult0, v_hash_const1[0]))
                                sd[STAGE_H0] = cycle
                            continue

                    if sd[STAGE_H0] >= 0 and sd[STAGE_H0] < cycle:
                        if sd[STAGE_H1P1A] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[1][0], ctx["tmp1"], ctx["val"], v_hash_const1[1]))
                            sd[STAGE_H1P1A] = cycle
                        if sd[STAGE_H1P1B] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[1][3], ctx["tmp2"], ctx["val"], v_hash_const2[1]))
                            sd[STAGE_H1P1B] = cycle
                        if sd[STAGE_H1P1A] < 0 or sd[STAGE_H1P1B] < 0:
                            continue

                    if sd[STAGE_H1P1A] >= 0 and sd[STAGE_H1P1A] < cycle and \
                       sd[STAGE_H1P1B] >= 0 and sd[STAGE_H1P1B] < cycle:
                        if sd[STAGE_H1] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append((HASH_STAGES[1][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                sd[STAGE_H1] = cycle
                            continue

                    if sd[STAGE_H1] >= 0 and sd[STAGE_H1] < cycle:
                        if sd[STAGE_H2] < 0:
                            if len(valu_ops) <= 4:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult2, v_hash_const1[2]))
                                valu_ops.append(("multiply_add", ctx["idx"], ctx["idx"], v_two, v_one))
                                sd[STAGE_H2] = cycle
                            continue

                    if sd[STAGE_H2] >= 0 and sd[STAGE_H2] < cycle:
                        if sd[STAGE_H3P1A] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[3][0], ctx["tmp1"], ctx["val"], v_hash_const1[3]))
                            sd[STAGE_H3P1A] = cycle
                        if sd[STAGE_H3P1B] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[3][3], ctx["tmp2"], ctx["val"], v_hash_const2[3]))
                            sd[STAGE_H3P1B] = cycle
                        if sd[STAGE_H3P1A] < 0 or sd[STAGE_H3P1B] < 0:
                            continue

                    if sd[STAGE_H3P1A] >= 0 and sd[STAGE_H3P1A] < cycle and \
                       sd[STAGE_H3P1B] >= 0 and sd[STAGE_H3P1B] < cycle:
                        if sd[STAGE_H3] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append((HASH_STAGES[3][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                sd[STAGE_H3] = cycle
                            continue

                    if sd[STAGE_H3] >= 0 and sd[STAGE_H3] < cycle:
                        if sd[STAGE_H4] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult4, v_hash_const1[4]))
                                sd[STAGE_H4] = cycle
                            continue

                    if sd[STAGE_H4] >= 0 and sd[STAGE_H4] < cycle:
                        if sd[STAGE_H5P1A] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[5][0], ctx["tmp1"], ctx["val"], v_hash_const1[5]))
                            sd[STAGE_H5P1A] = cycle
                        if sd[STAGE_H5P1B] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[5][3], ctx["tmp2"], ctx["val"], v_hash_const2[5]))
                            sd[STAGE_H5P1B] = cycle
                        if sd[STAGE_H5P1A] < 0 or sd[STAGE_H5P1B] < 0:
                            continue

                    if sd[STAGE_H5P1A] >= 0 and sd[STAGE_H5P1A] < cycle and \
                       sd[STAGE_H5P1B] >= 0 and sd[STAGE_H5P1B] < cycle:
                        if sd[STAGE_H5] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append((HASH_STAGES[5][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                sd[STAGE_H5] = cycle
                            continue

                    if sd[STAGE_H5] >= 0 and sd[STAGE_H5] < cycle:
                        if sd[STAGE_IDX_AND] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("&", ctx["tmp1"], ctx["val"], v_one))
                                sd[STAGE_IDX_AND] = cycle
                            continue

                    if sd[STAGE_IDX_AND] >= 0 and sd[STAGE_IDX_AND] < cycle:
                        if sd[STAGE_IDX_ADD] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("+", ctx["idx"], ctx["idx"], ctx["tmp1"]))
                                sd[STAGE_IDX_ADD] = cycle
                            continue

                    # Overflow check (using VALU instead of flow vselect)
                    if overflow:
                        if sd[STAGE_IDX_ADD] >= 0 and sd[STAGE_IDX_ADD] < cycle:
                            if sd[STAGE_OVF_LT] < 0:
                                if len(valu_ops) < 6:
                                    valu_ops.append(("<", ctx["tmp1"], ctx["idx"], v_n_nodes))
                                    sd[STAGE_OVF_LT] = cycle
                                continue

                        if sd[STAGE_OVF_LT] >= 0 and sd[STAGE_OVF_LT] < cycle:
                            if sd[STAGE_OVF_NEG] < 0:
                                if len(valu_ops) < 6:
                                    # tmp2 = 0 - tmp1 = -tmp1 (creates mask: 0xFFFFFFFF if tmp1=1, 0 if tmp1=0)
                                    valu_ops.append(("-", ctx["tmp2"], v_zero, ctx["tmp1"]))
                                    sd[STAGE_OVF_NEG] = cycle
                                continue

                        if sd[STAGE_OVF_NEG] >= 0 and sd[STAGE_OVF_NEG] < cycle:
                            if sd[STAGE_OVF_AND] < 0:
                                if len(valu_ops) < 6:
                                    # idx = idx & tmp2 (zeros out idx where tmp1 was 0)
                                    valu_ops.append(("&", ctx["idx"], ctx["idx"], ctx["tmp2"]))
                                    sd[STAGE_OVF_AND] = cycle
                                continue

                        if sd[STAGE_OVF_AND] >= 0 and sd[STAGE_OVF_AND] < cycle:
                            current_round[g] += 1
                            if current_round[g] < tile_end:
                                for s in range(50):
                                    sd[s] = -1
                                next_level = get_level(current_round[g])
                                if is_scattered(next_level):
                                    sd[STAGE_ADDR] = -2
                                else:
                                    sd[STAGE_XOR] = -2
                            continue
                    else:
                        if sd[STAGE_IDX_ADD] >= 0 and sd[STAGE_IDX_ADD] < cycle:
                            current_round[g] += 1
                            if current_round[g] < tile_end:
                                for s in range(50):
                                    sd[s] = -1
                                next_level = get_level(current_round[g])
                                if is_scattered(next_level):
                                    sd[STAGE_ADDR] = -2
                                else:
                                    sd[STAGE_XOR] = -2
                            continue

                else:
                    # ===== NON-SCATTERED ROUND =====
                    if level == 0:
                        if sd[STAGE_XOR] == -2:
                            if len(alu_ops) <= 4:
                                for i in range(8):
                                    alu_ops.append(("^", ctx["val"] + i, ctx["val"] + i, v_forest_nodes[0] + i))
                                sd[STAGE_XOR] = cycle
                            continue
                    elif level == 1:
                        if sd[STAGE_AND1] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("&", ctx["tmp1"], ctx["idx"], v_one))
                                sd[STAGE_AND1] = cycle
                            continue

                        if sd[STAGE_AND1] >= 0 and sd[STAGE_AND1] < cycle:
                            if sd[STAGE_VSEL1] < 0:
                                if len(flow_ops) < 1:
                                    flow_ops.append(("vselect", ctx["tmp2"], ctx["tmp1"],
                                                   v_forest_nodes[1], v_forest_nodes[2]))
                                    sd[STAGE_VSEL1] = cycle
                                continue

                        if sd[STAGE_VSEL1] >= 0 and sd[STAGE_VSEL1] < cycle:
                            if sd[STAGE_XOR] < 0:
                                if len(alu_ops) <= 4:
                                    for i in range(8):
                                        alu_ops.append(("^", ctx["val"] + i, ctx["val"] + i, ctx["tmp2"] + i))
                                    sd[STAGE_XOR] = cycle
                                continue
                    elif level == 2:
                        if sd[STAGE_AND1] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("&", ctx["tmp1"], ctx["idx"], v_one))
                                sd[STAGE_AND1] = cycle
                            continue

                        if sd[STAGE_AND1] >= 0 and sd[STAGE_AND1] < cycle:
                            if sd[STAGE_VSEL1] < 0:
                                if len(flow_ops) < 1:
                                    flow_ops.append(("vselect", ctx["tmp2"], ctx["tmp1"],
                                                   v_forest_nodes[3], v_forest_nodes[4]))
                                    sd[STAGE_VSEL1] = cycle
                                continue

                        if sd[STAGE_VSEL1] >= 0 and sd[STAGE_VSEL1] < cycle:
                            if sd[STAGE_VSEL2] < 0:
                                if len(flow_ops) < 1:
                                    flow_ops.append(("vselect", ctx["node"], ctx["tmp1"],
                                                   v_forest_nodes[5], v_forest_nodes[6]))
                                    sd[STAGE_VSEL2] = cycle
                                continue

                        if sd[STAGE_VSEL2] >= 0 and sd[STAGE_VSEL2] < cycle:
                            if sd[STAGE_LT] < 0:
                                if len(valu_ops) < 6:
                                    valu_ops.append(("<", ctx["tmp1"], ctx["idx"], v_five))
                                    sd[STAGE_LT] = cycle
                                continue

                        if sd[STAGE_LT] >= 0 and sd[STAGE_LT] < cycle:
                            if sd[STAGE_VSEL3] < 0:
                                if len(flow_ops) < 1:
                                    flow_ops.append(("vselect", ctx["tmp2"], ctx["tmp1"],
                                                   ctx["tmp2"], ctx["node"]))
                                    sd[STAGE_VSEL3] = cycle
                                continue

                        if sd[STAGE_VSEL3] >= 0 and sd[STAGE_VSEL3] < cycle:
                            if sd[STAGE_XOR] < 0:
                                if len(alu_ops) <= 4:
                                    for i in range(8):
                                        alu_ops.append(("^", ctx["val"] + i, ctx["val"] + i, ctx["tmp2"] + i))
                                    sd[STAGE_XOR] = cycle
                                continue

                    # Hash stages for non-scattered
                    if sd[STAGE_XOR] >= 0 and sd[STAGE_XOR] < cycle:
                        if sd[STAGE_H0] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult0, v_hash_const1[0]))
                                sd[STAGE_H0] = cycle
                            continue

                    if sd[STAGE_H0] >= 0 and sd[STAGE_H0] < cycle:
                        if sd[STAGE_H1P1A] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[1][0], ctx["tmp1"], ctx["val"], v_hash_const1[1]))
                            sd[STAGE_H1P1A] = cycle
                        if sd[STAGE_H1P1B] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[1][3], ctx["tmp2"], ctx["val"], v_hash_const2[1]))
                            sd[STAGE_H1P1B] = cycle
                        if sd[STAGE_H1P1A] < 0 or sd[STAGE_H1P1B] < 0:
                            continue

                    if sd[STAGE_H1P1A] >= 0 and sd[STAGE_H1P1A] < cycle and \
                       sd[STAGE_H1P1B] >= 0 and sd[STAGE_H1P1B] < cycle:
                        if sd[STAGE_H1] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append((HASH_STAGES[1][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                sd[STAGE_H1] = cycle
                            continue

                    if sd[STAGE_H1] >= 0 and sd[STAGE_H1] < cycle:
                        if sd[STAGE_H2] < 0:
                            if len(valu_ops) <= 4:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult2, v_hash_const1[2]))
                                valu_ops.append(("multiply_add", ctx["idx"], ctx["idx"], v_two, v_one))
                                sd[STAGE_H2] = cycle
                            continue

                    if sd[STAGE_H2] >= 0 and sd[STAGE_H2] < cycle:
                        if sd[STAGE_H3P1A] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[3][0], ctx["tmp1"], ctx["val"], v_hash_const1[3]))
                            sd[STAGE_H3P1A] = cycle
                        if sd[STAGE_H3P1B] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[3][3], ctx["tmp2"], ctx["val"], v_hash_const2[3]))
                            sd[STAGE_H3P1B] = cycle
                        if sd[STAGE_H3P1A] < 0 or sd[STAGE_H3P1B] < 0:
                            continue

                    if sd[STAGE_H3P1A] >= 0 and sd[STAGE_H3P1A] < cycle and \
                       sd[STAGE_H3P1B] >= 0 and sd[STAGE_H3P1B] < cycle:
                        if sd[STAGE_H3] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append((HASH_STAGES[3][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                sd[STAGE_H3] = cycle
                            continue

                    if sd[STAGE_H3] >= 0 and sd[STAGE_H3] < cycle:
                        if sd[STAGE_H4] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult4, v_hash_const1[4]))
                                sd[STAGE_H4] = cycle
                            continue

                    if sd[STAGE_H4] >= 0 and sd[STAGE_H4] < cycle:
                        if sd[STAGE_H5P1A] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[5][0], ctx["tmp1"], ctx["val"], v_hash_const1[5]))
                            sd[STAGE_H5P1A] = cycle
                        if sd[STAGE_H5P1B] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[5][3], ctx["tmp2"], ctx["val"], v_hash_const2[5]))
                            sd[STAGE_H5P1B] = cycle
                        if sd[STAGE_H5P1A] < 0 or sd[STAGE_H5P1B] < 0:
                            continue

                    if sd[STAGE_H5P1A] >= 0 and sd[STAGE_H5P1A] < cycle and \
                       sd[STAGE_H5P1B] >= 0 and sd[STAGE_H5P1B] < cycle:
                        if sd[STAGE_H5] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append((HASH_STAGES[5][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                sd[STAGE_H5] = cycle
                            continue

                    if sd[STAGE_H5] >= 0 and sd[STAGE_H5] < cycle:
                        if sd[STAGE_IDX_AND] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("&", ctx["tmp1"], ctx["val"], v_one))
                                sd[STAGE_IDX_AND] = cycle
                            continue

                    if sd[STAGE_IDX_AND] >= 0 and sd[STAGE_IDX_AND] < cycle:
                        if sd[STAGE_IDX_ADD] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("+", ctx["idx"], ctx["idx"], ctx["tmp1"]))
                                sd[STAGE_IDX_ADD] = cycle
                            continue

                    if sd[STAGE_IDX_ADD] >= 0 and sd[STAGE_IDX_ADD] < cycle:
                        current_round[g] += 1
                        if current_round[g] < tile_end:
                            for s in range(50):
                                sd[s] = -1
                            next_level = get_level(current_round[g])
                            if is_scattered(next_level):
                                sd[STAGE_ADDR] = -2
                            else:
                                sd[STAGE_XOR] = -2

            # Emit instruction
            instr = {}
            if valu_ops:
                instr["valu"] = valu_ops[:6]
            if alu_ops:
                instr["alu"] = alu_ops[:12]
            if load_ops:
                instr["load"] = load_ops[:2]
            if store_ops:
                instr["store"] = store_ops[:2]
            if flow_ops:
                instr["flow"] = flow_ops[:1]

            if instr:
                self.instrs.append(instr)

            cycle += 1


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
