"""
Round-tiled kernel implementation.
GROUP_SIZE=17, ROUND_TILE=13 with diagonal scheduling.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
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

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space: {self.scratch_ptr} > {SCRATCH_SIZE}"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def alloc_vec(self, name=None):
        return self.alloc_scratch(name, VLEN)

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Round-tiled kernel with diagonal scheduling.
        GROUP_SIZE=17, ROUND_TILE=13
        """
        self.forest_height = forest_height
        n_batches = batch_size // VLEN
        GROUP_SIZE = 17
        ROUND_TILE = 13

        def emit_valu_chunked(ops):
            """Emit VALU ops in chunks of 6 (max VALU slots)"""
            for i in range(0, len(ops), 6):
                self.instrs.append({"valu": ops[i:i+6]})

        # ===== ALLOCATE CONTEXTS =====
        # Reduced context: idx, val, node, tmp1, tmp2 (no addrs - compute inline)
        # Also add addr vector for VALU address computation
        contexts = []
        for g in range(GROUP_SIZE):
            ctx = {
                "idx": self.alloc_vec(f"ctx{g}_idx"),
                "val": self.alloc_vec(f"ctx{g}_val"),
                "node": self.alloc_vec(f"ctx{g}_node"),
                "tmp1": self.alloc_vec(f"ctx{g}_tmp1"),
                "tmp2": self.alloc_vec(f"ctx{g}_tmp2"),
                "addr": self.alloc_vec(f"ctx{g}_addr"),  # Vector addresses
            }
            contexts.append(ctx)

        # Constants
        v_zero = self.alloc_vec("v_zero")
        v_one = self.alloc_vec("v_one")
        v_two = self.alloc_vec("v_two")
        v_five = self.alloc_vec("v_five")
        v_n_nodes = self.alloc_vec("v_n_nodes")
        v_forest_base = self.alloc_vec("v_forest_base")  # For VALU addr calc

        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Pre-allocate init vars
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Pre-allocate scalar constants
        const_addrs = {}
        for val in [0, 1, 2, 5]:
            const_addrs[val] = self.alloc_scratch(f"c_{val}")
            self.const_map[val] = const_addrs[val]

        # Batch the const loads for init_vars
        const_vals = [0, 1, 2, 5]
        const_idx = 0
        for i in range(0, len(init_vars), 2):
            loads = [("const", tmp1, i)]
            if i + 1 < len(init_vars):
                loads.append(("const", tmp2, i + 1))
            else:
                loads.append(("const", const_addrs[const_vals[const_idx]], const_vals[const_idx]))
                const_idx += 1
            self.instrs.append({"load": loads})
            loads2 = [("load", self.scratch[init_vars[i]], tmp1)]
            if i + 1 < len(init_vars):
                loads2.append(("load", self.scratch[init_vars[i + 1]], tmp2))
            else:
                loads2.append(("const", const_addrs[const_vals[const_idx]], const_vals[const_idx]))
                const_idx += 1
            self.instrs.append({"load": loads2})

        while const_idx < len(const_vals):
            loads = [("const", const_addrs[const_vals[const_idx]], const_vals[const_idx])]
            const_idx += 1
            if const_idx < len(const_vals):
                loads.append(("const", const_addrs[const_vals[const_idx]], const_vals[const_idx]))
                const_idx += 1
            self.instrs.append({"load": loads})

        zero_const = const_addrs[0]
        one_const = const_addrs[1]
        two_const = const_addrs[2]
        five_const = const_addrs[5]

        self.instrs.append({"valu": [
            ("vbroadcast", v_two, two_const),
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_zero, zero_const),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
            ("vbroadcast", v_five, five_const),
            ("vbroadcast", v_forest_base, self.scratch["forest_values_p"]),
        ]})

        self.instrs.append({"flow": [("pause",)], "debug": [("comment", "Starting loop")]})

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
            self.const_map[hash_const1_vals[i]] = addr

        hash_c2_addrs = []
        for i in range(6):
            addr = self.alloc_scratch(f"hash{i}_c2")
            hash_c2_addrs.append(addr)
            self.const_map[hash_const2_vals[i]] = addr

        mult_addrs = []
        for i, val in enumerate(mult_vals):
            addr = self.alloc_scratch(f"mult_{i}")
            mult_addrs.append(addr)
            self.const_map[val] = addr

        # ===== CACHE TREE NODES 0-6 =====
        v_forest_nodes = [self.alloc_vec(f"v_forest_node_{i}") for i in range(7)]
        forest_cache = self.alloc_scratch("forest_cache", 8)

        all_addrs = hash_c1_addrs + hash_c2_addrs + mult_addrs
        for i in range(0, len(all_addrs), 2):
            loads = [("const", all_addrs[i], all_const_vals[i])]
            if i + 1 < len(all_addrs):
                loads.append(("const", all_addrs[i + 1], all_const_vals[i + 1]))
            else:
                loads.append(("vload", forest_cache, self.scratch["forest_values_p"]))
            self.instrs.append({"load": loads})

        self.instrs.append({"valu": [
            ("vbroadcast", v_hash_const1[i], hash_c1_addrs[i])
            for i in range(6)
        ]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_hash_const2[i], hash_c2_addrs[i])
            for i in range(6)
        ]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_mult0, mult_addrs[0]),
            ("vbroadcast", v_mult2, mult_addrs[1]),
            ("vbroadcast", v_mult4, mult_addrs[2]),
            ("vbroadcast", v_forest_nodes[6], forest_cache + 6),
        ]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_forest_nodes[i], forest_cache + i) for i in range(6)
        ]})

        # ===== PRECOMPUTE BATCH ADDRESSES =====
        idx_addrs = [self.alloc_scratch(f"idx_addr_{i}") for i in range(n_batches)]
        val_addrs = [self.alloc_scratch(f"val_addr_{i}") for i in range(n_batches)]

        offset_vals = [i * VLEN for i in range(n_batches)]
        offset_addrs = []
        for i in range(n_batches):
            addr = self.alloc_scratch(f"off_{i}")
            offset_addrs.append(addr)
            self.const_map[offset_vals[i]] = addr

        for i in range(0, n_batches, 2):
            loads = [("const", offset_addrs[i], offset_vals[i])]
            if i + 1 < n_batches:
                loads.append(("const", offset_addrs[i + 1], offset_vals[i + 1]))
            self.instrs.append({"load": loads})

        for i in range(0, n_batches, 6):
            alu_ops = []
            for j in range(min(6, n_batches - i)):
                alu_ops.append(("+", idx_addrs[i+j], self.scratch["inp_indices_p"], offset_addrs[i+j]))
                alu_ops.append(("+", val_addrs[i+j], self.scratch["inp_values_p"], offset_addrs[i+j]))
            self.instrs.append({"alu": alu_ops})

        print(f"Scratch usage after init: {self.scratch_ptr}")

        # ===== MAIN KERNEL WITH ROUND TILING =====
        for group_start in range(0, n_batches, GROUP_SIZE):
            group_end = min(group_start + GROUP_SIZE, n_batches)
            active = group_end - group_start

            # Process rounds in tiles
            # Loads and stores are integrated into the first/last tile processing
            for tile_start in range(0, rounds, ROUND_TILE):
                tile_end = min(tile_start + ROUND_TILE, rounds)

                is_first_tile = (tile_start == 0)
                is_last_tile = (tile_end >= rounds)

                # Pass load addresses for first tile
                load_addrs = [(idx_addrs[group_start + g], val_addrs[group_start + g])
                              for g in range(active)] if is_first_tile else None
                # Pass store addresses for last tile
                store_addrs = [(idx_addrs[group_start + g], val_addrs[group_start + g])
                               for g in range(active)] if is_last_tile else None

                self._process_tile_diagonal(
                    contexts, active, tile_start, tile_end, forest_height, n_nodes,
                    v_forest_nodes, v_hash_const1, v_hash_const2,
                    v_mult0, v_mult2, v_mult4, v_two, v_one, v_zero,
                    v_n_nodes, v_five, v_forest_base,
                    load_addrs=load_addrs, store_addrs=store_addrs
                )

        self.instrs.append({"flow": [("pause",)]})

    def _process_tile_diagonal(
        self, contexts, active, tile_start, tile_end, forest_height, n_nodes,
        v_forest_nodes, v_hash_const1, v_hash_const2,
        v_mult0, v_mult2, v_mult4, v_two, v_one, v_zero,
        v_n_nodes, v_five, v_forest_base,
        load_addrs=None, store_addrs=None
    ):
        """
        Process a tile of rounds using diagonal scheduling.
        Each batch can be at a different round, allowing better VALU packing.
        Optionally handles initial loads and final stores integrated into processing.
        """
        n_rounds = tile_end - tile_start

        # Track batch loading/storing state
        batch_loaded = [False] * active if load_addrs else [True] * active
        batch_load_cycle = [-1] * active  # Cycle when load was issued
        batch_stored = [False] * active

        # State tracking: current_round[g] = which round batch g is processing
        current_round = [tile_start] * active

        # Per-batch, per-round stage tracking
        # Stages: 0=need_xor, 1=need_h0, 2=need_h1p1, 3=need_h1, 4=need_h2,
        #         5=need_h3p1, 6=need_h3, 7=need_h4, 8=need_h5p1, 9=need_h5,
        #         10=need_idx_and, 11=need_idx_add, 12=need_ovf_lt, 13=need_ovf_sel, 14=done
        # For scattered rounds, add stages for addr_calc (0) and loads (1-4)
        STAGE_ADDR = 0
        STAGE_LOAD0 = 1
        STAGE_LOAD1 = 2
        STAGE_LOAD2 = 3
        STAGE_LOAD3 = 4
        STAGE_XOR = 5
        STAGE_H0 = 6
        STAGE_H1P1 = 7
        STAGE_H1 = 8
        STAGE_H2 = 9
        STAGE_H3P1 = 10
        STAGE_H3 = 11
        STAGE_H4 = 12
        STAGE_H5P1 = 13
        STAGE_H5 = 14
        STAGE_IDX_AND = 15
        STAGE_IDX_ADD = 16
        STAGE_OVF_LT = 17
        STAGE_OVF_SEL = 18
        STAGE_DONE = 19

        # For non-scattered with vselect (levels 1, 2)
        STAGE_AND1 = 0
        STAGE_VSEL1 = 1
        STAGE_LT = 2  # for level 2
        STAGE_VSEL2 = 3  # for level 2
        STAGE_VSEL3 = 4  # for level 2

        # Track stage completion cycle for each batch
        stage_done = [[-1] * 20 for _ in range(active)]  # stage_done[batch][stage]
        batch_done = [False] * active

        def get_level(round_num):
            return round_num % (forest_height + 1)

        def is_scattered(level):
            return level >= 3

        def needs_overflow(level):
            return level == forest_height

        # Initialize: all batches start at tile_start
        for g in range(active):
            level = get_level(tile_start)
            if is_scattered(level):
                stage_done[g][STAGE_ADDR] = -2  # needs doing
            else:
                stage_done[g][STAGE_XOR] = -2  # needs doing

        cycle = 0
        max_cycles = 5000  # Safety limit

        while not all(batch_done) and cycle < max_cycles:
            instr = {}
            valu_ops = []
            alu_ops = []
            load_ops = []
            store_ops = []
            flow_ops = []

            # Issue loads for batches that haven't been loaded yet
            if load_addrs:
                for g in range(active):
                    if not batch_loaded[g] and len(load_ops) < 2:
                        idx_addr, val_addr = load_addrs[g]
                        ctx = contexts[g]
                        load_ops.append(("vload", ctx["idx"], idx_addr))
                        if len(load_ops) < 2:
                            load_ops.append(("vload", ctx["val"], val_addr))
                            batch_loaded[g] = True
                            batch_load_cycle[g] = cycle

            # Issue stores for completed batches
            if store_addrs:
                for g in range(active):
                    if batch_done[g] and not batch_stored[g] and len(store_ops) < 2:
                        idx_addr, val_addr = store_addrs[g]
                        ctx = contexts[g]
                        store_ops.append(("vstore", idx_addr, ctx["idx"]))
                        if len(store_ops) < 2:
                            store_ops.append(("vstore", val_addr, ctx["val"]))
                            batch_stored[g] = True

            # Process all batches, trying to fill slots
            for g in range(active):
                if batch_done[g]:
                    continue

                # Skip if batch hasn't been loaded yet (need 1 cycle latency)
                if load_addrs and (not batch_loaded[g] or batch_load_cycle[g] >= cycle):
                    continue

                ctx = contexts[g]
                rnd = current_round[g]
                if rnd >= tile_end:
                    batch_done[g] = True
                    continue

                level = get_level(rnd)
                need_scattered = is_scattered(level)
                need_overflow = needs_overflow(level)

                if need_scattered:
                    # Scattered round: addr -> loads -> xor -> hash -> idx
                    # Address calculation (8 ALU ops - keeps load as bottleneck)
                    if stage_done[g][STAGE_ADDR] == -2:
                        if len(alu_ops) <= 4:  # Need space for 8 ops
                            for i in range(8):
                                alu_ops.append(("+", ctx["addr"] + i, self.scratch["forest_values_p"], ctx["idx"] + i))
                            stage_done[g][STAGE_ADDR] = cycle
                        continue

                    # Loads (8 loads = 4 pairs, but we do 2 per cycle)
                    if stage_done[g][STAGE_ADDR] >= 0 and stage_done[g][STAGE_ADDR] < cycle:
                        for load_stage in range(STAGE_LOAD0, STAGE_LOAD3 + 1):
                            if stage_done[g][load_stage] < 0:
                                if len(load_ops) < 2:
                                    i = (load_stage - STAGE_LOAD0) * 2
                                    # Need to use scalar loads since addresses are in a vector
                                    load_ops.append(("load", ctx["node"] + i, ctx["addr"] + i))
                                    load_ops.append(("load", ctx["node"] + i + 1, ctx["addr"] + i + 1))
                                    stage_done[g][load_stage] = cycle
                                break
                        if stage_done[g][STAGE_LOAD3] < 0:
                            continue

                    # XOR - use ALU instead of VALU (8 scalar ops, ALU has 12 slots)
                    if stage_done[g][STAGE_LOAD3] >= 0 and stage_done[g][STAGE_LOAD3] < cycle:
                        if stage_done[g][STAGE_XOR] < 0:
                            if len(alu_ops) <= 4:  # Need 8 slots
                                for i in range(8):
                                    alu_ops.append(("^", ctx["val"] + i, ctx["val"] + i, ctx["node"] + i))
                                stage_done[g][STAGE_XOR] = cycle
                            continue

                    # Hash stages (same pattern as before)
                    if stage_done[g][STAGE_XOR] >= 0 and stage_done[g][STAGE_XOR] < cycle:
                        if stage_done[g][STAGE_H0] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult0, v_hash_const1[0]))
                                stage_done[g][STAGE_H0] = cycle
                            continue

                    if stage_done[g][STAGE_H0] >= 0 and stage_done[g][STAGE_H0] < cycle:
                        if stage_done[g][STAGE_H1P1] < 0:
                            if len(valu_ops) <= 4:
                                valu_ops.append((HASH_STAGES[1][0], ctx["tmp1"], ctx["val"], v_hash_const1[1]))
                                valu_ops.append((HASH_STAGES[1][3], ctx["tmp2"], ctx["val"], v_hash_const2[1]))
                                stage_done[g][STAGE_H1P1] = cycle
                            continue

                    if stage_done[g][STAGE_H1P1] >= 0 and stage_done[g][STAGE_H1P1] < cycle:
                        if stage_done[g][STAGE_H1] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append((HASH_STAGES[1][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                stage_done[g][STAGE_H1] = cycle
                            continue

                    if stage_done[g][STAGE_H1] >= 0 and stage_done[g][STAGE_H1] < cycle:
                        if stage_done[g][STAGE_H2] < 0:
                            if len(valu_ops) <= 4:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult2, v_hash_const1[2]))
                                valu_ops.append(("multiply_add", ctx["idx"], ctx["idx"], v_two, v_one))
                                stage_done[g][STAGE_H2] = cycle
                            continue

                    if stage_done[g][STAGE_H2] >= 0 and stage_done[g][STAGE_H2] < cycle:
                        if stage_done[g][STAGE_H3P1] < 0:
                            if len(valu_ops) <= 4:
                                valu_ops.append((HASH_STAGES[3][0], ctx["tmp1"], ctx["val"], v_hash_const1[3]))
                                valu_ops.append((HASH_STAGES[3][3], ctx["tmp2"], ctx["val"], v_hash_const2[3]))
                                stage_done[g][STAGE_H3P1] = cycle
                            continue

                    if stage_done[g][STAGE_H3P1] >= 0 and stage_done[g][STAGE_H3P1] < cycle:
                        if stage_done[g][STAGE_H3] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append((HASH_STAGES[3][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                stage_done[g][STAGE_H3] = cycle
                            continue

                    if stage_done[g][STAGE_H3] >= 0 and stage_done[g][STAGE_H3] < cycle:
                        if stage_done[g][STAGE_H4] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult4, v_hash_const1[4]))
                                stage_done[g][STAGE_H4] = cycle
                            continue

                    if stage_done[g][STAGE_H4] >= 0 and stage_done[g][STAGE_H4] < cycle:
                        if stage_done[g][STAGE_H5P1] < 0:
                            if len(valu_ops) <= 4:
                                valu_ops.append((HASH_STAGES[5][0], ctx["tmp1"], ctx["val"], v_hash_const1[5]))
                                valu_ops.append((HASH_STAGES[5][3], ctx["tmp2"], ctx["val"], v_hash_const2[5]))
                                stage_done[g][STAGE_H5P1] = cycle
                            continue

                    if stage_done[g][STAGE_H5P1] >= 0 and stage_done[g][STAGE_H5P1] < cycle:
                        if stage_done[g][STAGE_H5] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append((HASH_STAGES[5][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                stage_done[g][STAGE_H5] = cycle
                            continue

                    if stage_done[g][STAGE_H5] >= 0 and stage_done[g][STAGE_H5] < cycle:
                        if stage_done[g][STAGE_IDX_AND] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("&", ctx["tmp1"], ctx["val"], v_one))
                                stage_done[g][STAGE_IDX_AND] = cycle
                            continue

                    if stage_done[g][STAGE_IDX_AND] >= 0 and stage_done[g][STAGE_IDX_AND] < cycle:
                        if stage_done[g][STAGE_IDX_ADD] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("+", ctx["idx"], ctx["idx"], ctx["tmp1"]))
                                stage_done[g][STAGE_IDX_ADD] = cycle
                            continue

                    # Overflow check
                    if need_overflow:
                        if stage_done[g][STAGE_IDX_ADD] >= 0 and stage_done[g][STAGE_IDX_ADD] < cycle:
                            if stage_done[g][STAGE_OVF_LT] < 0:
                                if len(valu_ops) < 6:
                                    valu_ops.append(("<", ctx["tmp1"], ctx["idx"], v_n_nodes))
                                    stage_done[g][STAGE_OVF_LT] = cycle
                                continue

                        if stage_done[g][STAGE_OVF_LT] >= 0 and stage_done[g][STAGE_OVF_LT] < cycle:
                            if stage_done[g][STAGE_OVF_SEL] < 0:
                                if len(flow_ops) < 1:
                                    flow_ops.append(("vselect", ctx["idx"], ctx["tmp1"], ctx["idx"], v_zero))
                                    stage_done[g][STAGE_OVF_SEL] = cycle
                                continue

                        if stage_done[g][STAGE_OVF_SEL] >= 0 and stage_done[g][STAGE_OVF_SEL] < cycle:
                            # Round done, advance to next
                            current_round[g] += 1
                            if current_round[g] < tile_end:
                                # Reset stages for next round
                                for s in range(20):
                                    stage_done[g][s] = -1
                                next_level = get_level(current_round[g])
                                if is_scattered(next_level):
                                    stage_done[g][STAGE_ADDR] = -2
                                else:
                                    stage_done[g][STAGE_XOR] = -2
                            continue
                    else:
                        if stage_done[g][STAGE_IDX_ADD] >= 0 and stage_done[g][STAGE_IDX_ADD] < cycle:
                            # Round done, advance to next
                            current_round[g] += 1
                            if current_round[g] < tile_end:
                                for s in range(20):
                                    stage_done[g][s] = -1
                                next_level = get_level(current_round[g])
                                if is_scattered(next_level):
                                    stage_done[g][STAGE_ADDR] = -2
                                else:
                                    stage_done[g][STAGE_XOR] = -2
                            continue

                else:
                    # Non-scattered: use cached nodes
                    # Level 0: direct XOR with v_forest_nodes[0]
                    # Level 1: & -> vselect -> XOR
                    # Level 2: & -> vselect1,2 -> < -> vselect3 -> XOR

                    if level == 0:
                        # Direct XOR - use ALU (8 scalar ops)
                        if stage_done[g][STAGE_XOR] == -2:
                            if len(alu_ops) <= 4:  # Need 8 slots
                                for i in range(8):
                                    alu_ops.append(("^", ctx["val"] + i, ctx["val"] + i, v_forest_nodes[0] + i))
                                stage_done[g][STAGE_XOR] = cycle
                            continue
                    elif level == 1:
                        # & -> vselect -> XOR
                        if stage_done[g][STAGE_AND1] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("&", ctx["tmp1"], ctx["idx"], v_one))
                                stage_done[g][STAGE_AND1] = cycle
                            continue

                        if stage_done[g][STAGE_AND1] >= 0 and stage_done[g][STAGE_AND1] < cycle:
                            if stage_done[g][STAGE_VSEL1] < 0:
                                if len(flow_ops) < 1:
                                    flow_ops.append(("vselect", ctx["tmp2"], ctx["tmp1"],
                                                   v_forest_nodes[1], v_forest_nodes[2]))
                                    stage_done[g][STAGE_VSEL1] = cycle
                                continue

                        if stage_done[g][STAGE_VSEL1] >= 0 and stage_done[g][STAGE_VSEL1] < cycle:
                            if stage_done[g][STAGE_XOR] < 0:
                                if len(alu_ops) <= 4:  # Need 8 slots for scalar XOR
                                    for i in range(8):
                                        alu_ops.append(("^", ctx["val"] + i, ctx["val"] + i, ctx["tmp2"] + i))
                                    stage_done[g][STAGE_XOR] = cycle
                                continue
                    elif level == 2:
                        # & -> vsel1,vsel2 -> < -> vsel3 -> XOR
                        if stage_done[g][STAGE_AND1] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("&", ctx["tmp1"], ctx["idx"], v_one))
                                stage_done[g][STAGE_AND1] = cycle
                            continue

                        if stage_done[g][STAGE_AND1] >= 0 and stage_done[g][STAGE_AND1] < cycle:
                            if stage_done[g][STAGE_VSEL1] < 0:
                                if len(flow_ops) < 1:
                                    flow_ops.append(("vselect", ctx["tmp2"], ctx["tmp1"],
                                                   v_forest_nodes[3], v_forest_nodes[4]))
                                    stage_done[g][STAGE_VSEL1] = cycle
                                continue

                        if stage_done[g][STAGE_VSEL1] >= 0 and stage_done[g][STAGE_VSEL1] < cycle:
                            if stage_done[g][STAGE_VSEL2] < 0:
                                if len(flow_ops) < 1:
                                    flow_ops.append(("vselect", ctx["node"], ctx["tmp1"],
                                                   v_forest_nodes[5], v_forest_nodes[6]))
                                    stage_done[g][STAGE_VSEL2] = cycle
                                continue

                        if stage_done[g][STAGE_VSEL2] >= 0 and stage_done[g][STAGE_VSEL2] < cycle:
                            if stage_done[g][STAGE_LT] < 0:
                                if len(valu_ops) < 6:
                                    valu_ops.append(("<", ctx["tmp1"], ctx["idx"], v_five))
                                    stage_done[g][STAGE_LT] = cycle
                                continue

                        if stage_done[g][STAGE_LT] >= 0 and stage_done[g][STAGE_LT] < cycle:
                            if stage_done[g][STAGE_VSEL3] < 0:
                                if len(flow_ops) < 1:
                                    flow_ops.append(("vselect", ctx["tmp2"], ctx["tmp1"],
                                                   ctx["tmp2"], ctx["node"]))
                                    stage_done[g][STAGE_VSEL3] = cycle
                                continue

                        if stage_done[g][STAGE_VSEL3] >= 0 and stage_done[g][STAGE_VSEL3] < cycle:
                            if stage_done[g][STAGE_XOR] < 0:
                                if len(alu_ops) <= 4:  # Need 8 slots for scalar XOR
                                    for i in range(8):
                                        alu_ops.append(("^", ctx["val"] + i, ctx["val"] + i, ctx["tmp2"] + i))
                                    stage_done[g][STAGE_XOR] = cycle
                                continue

                    # Hash stages (same for all non-scattered)
                    if stage_done[g][STAGE_XOR] >= 0 and stage_done[g][STAGE_XOR] < cycle:
                        if stage_done[g][STAGE_H0] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult0, v_hash_const1[0]))
                                stage_done[g][STAGE_H0] = cycle
                            continue

                    if stage_done[g][STAGE_H0] >= 0 and stage_done[g][STAGE_H0] < cycle:
                        if stage_done[g][STAGE_H1P1] < 0:
                            if len(valu_ops) <= 4:
                                valu_ops.append((HASH_STAGES[1][0], ctx["tmp1"], ctx["val"], v_hash_const1[1]))
                                valu_ops.append((HASH_STAGES[1][3], ctx["tmp2"], ctx["val"], v_hash_const2[1]))
                                stage_done[g][STAGE_H1P1] = cycle
                            continue

                    if stage_done[g][STAGE_H1P1] >= 0 and stage_done[g][STAGE_H1P1] < cycle:
                        if stage_done[g][STAGE_H1] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append((HASH_STAGES[1][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                stage_done[g][STAGE_H1] = cycle
                            continue

                    if stage_done[g][STAGE_H1] >= 0 and stage_done[g][STAGE_H1] < cycle:
                        if stage_done[g][STAGE_H2] < 0:
                            if len(valu_ops) <= 4:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult2, v_hash_const1[2]))
                                valu_ops.append(("multiply_add", ctx["idx"], ctx["idx"], v_two, v_one))
                                stage_done[g][STAGE_H2] = cycle
                            continue

                    if stage_done[g][STAGE_H2] >= 0 and stage_done[g][STAGE_H2] < cycle:
                        if stage_done[g][STAGE_H3P1] < 0:
                            if len(valu_ops) <= 4:
                                valu_ops.append((HASH_STAGES[3][0], ctx["tmp1"], ctx["val"], v_hash_const1[3]))
                                valu_ops.append((HASH_STAGES[3][3], ctx["tmp2"], ctx["val"], v_hash_const2[3]))
                                stage_done[g][STAGE_H3P1] = cycle
                            continue

                    if stage_done[g][STAGE_H3P1] >= 0 and stage_done[g][STAGE_H3P1] < cycle:
                        if stage_done[g][STAGE_H3] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append((HASH_STAGES[3][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                stage_done[g][STAGE_H3] = cycle
                            continue

                    if stage_done[g][STAGE_H3] >= 0 and stage_done[g][STAGE_H3] < cycle:
                        if stage_done[g][STAGE_H4] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult4, v_hash_const1[4]))
                                stage_done[g][STAGE_H4] = cycle
                            continue

                    if stage_done[g][STAGE_H4] >= 0 and stage_done[g][STAGE_H4] < cycle:
                        if stage_done[g][STAGE_H5P1] < 0:
                            if len(valu_ops) <= 4:
                                valu_ops.append((HASH_STAGES[5][0], ctx["tmp1"], ctx["val"], v_hash_const1[5]))
                                valu_ops.append((HASH_STAGES[5][3], ctx["tmp2"], ctx["val"], v_hash_const2[5]))
                                stage_done[g][STAGE_H5P1] = cycle
                            continue

                    if stage_done[g][STAGE_H5P1] >= 0 and stage_done[g][STAGE_H5P1] < cycle:
                        if stage_done[g][STAGE_H5] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append((HASH_STAGES[5][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                stage_done[g][STAGE_H5] = cycle
                            continue

                    if stage_done[g][STAGE_H5] >= 0 and stage_done[g][STAGE_H5] < cycle:
                        if stage_done[g][STAGE_IDX_AND] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("&", ctx["tmp1"], ctx["val"], v_one))
                                stage_done[g][STAGE_IDX_AND] = cycle
                            continue

                    if stage_done[g][STAGE_IDX_AND] >= 0 and stage_done[g][STAGE_IDX_AND] < cycle:
                        if stage_done[g][STAGE_IDX_ADD] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("+", ctx["idx"], ctx["idx"], ctx["tmp1"]))
                                stage_done[g][STAGE_IDX_ADD] = cycle
                            continue

                    if stage_done[g][STAGE_IDX_ADD] >= 0 and stage_done[g][STAGE_IDX_ADD] < cycle:
                        # Round done, advance
                        current_round[g] += 1
                        if current_round[g] < tile_end:
                            for s in range(20):
                                stage_done[g][s] = -1
                            next_level = get_level(current_round[g])
                            if is_scattered(next_level):
                                stage_done[g][STAGE_ADDR] = -2
                            else:
                                stage_done[g][STAGE_XOR] = -2

            # Emit instruction
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

        # Emit any remaining stores for batches that weren't stored during processing
        if store_addrs:
            for g in range(active):
                if not batch_stored[g]:
                    ctx = contexts[g]
                    idx_addr, val_addr = store_addrs[g]
                    self.instrs.append({"store": [
                        ("vstore", idx_addr, ctx["idx"]),
                        ("vstore", val_addr, ctx["val"]),
                    ]})


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


if __name__ == "__main__":
    do_kernel_test(10, 16, 256, trace=True)
