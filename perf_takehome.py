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
        # Use max contexts that fit (29) - small remainder (3) can be interweaved
        GROUP_SIZE = min(n_batches, 29)
        ROUND_TILE = 16  # Process all rounds in one tile

        def emit_valu_chunked(ops):
            """Emit VALU ops in chunks of 6 (max VALU slots)"""
            for i in range(0, len(ops), 6):
                self.instrs.append({"valu": ops[i:i+6]})

        # ===== ALLOCATE CONTEXTS =====
        # Minimal context: idx, val, node, tmp1, tmp2 (5 vectors)
        # tmp1 is reused as addr during scattered stages
        # Note: Can't alias node to tmp2 because level 2 needs both
        contexts = []
        for g in range(GROUP_SIZE):  # One context per batch for flat scheduling
            ctx = {
                "idx": self.alloc_vec(f"ctx{g}_idx"),
                "val": self.alloc_vec(f"ctx{g}_val"),
                "node": self.alloc_vec(f"ctx{g}_node"),
                "tmp1": self.alloc_vec(f"ctx{g}_tmp1"),  # Also used as addr in scattered
                "tmp2": self.alloc_vec(f"ctx{g}_tmp2"),
            }
            # Alias for clarity - addr reuses tmp1's storage
            ctx["addr"] = ctx["tmp1"]
            contexts.append(ctx)

        # Constants
        v_zero = self.alloc_vec("v_zero")
        v_one = self.alloc_vec("v_one")
        v_two = self.alloc_vec("v_two")
        v_five = self.alloc_vec("v_five")
        v_n_nodes = self.alloc_vec("v_n_nodes")
        v_forest_base = self.alloc_vec("v_forest_base")  # For VALU addr calc
        # Placeholders for removed level 3 constants (to avoid breaking references)
        v_four = v_zero  # Not used
        v_seven = v_zero  # Not used

        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Pre-allocate init vars
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Pre-allocate scalar constants
        const_addrs = {}
        for val in [0, 1, 2, 5]:  # Removed 4, 7 (level 3 not used)
            const_addrs[val] = self.alloc_scratch(f"c_{val}")
            self.const_map[val] = const_addrs[val]

        # Batch the const loads for init_vars
        const_vals = [0, 1, 2, 5]  # Removed 4, 7
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

        # ===== CACHE TREE NODES 0-6 (levels 0-2) =====
        # Level 3+ is scattered, so we only need nodes 0-6
        v_forest_nodes = [self.alloc_vec(f"v_forest_node_{i}") for i in range(7)]
        forest_cache = self.alloc_scratch("forest_cache", 8)  # Nodes 0-7

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

        # Level 3+ nodes are loaded via scattered path, no caching needed

        # ===== PRECOMPUTE BATCH ADDRESSES =====
        idx_addrs = [self.alloc_scratch(f"idx_addr_{i}") for i in range(n_batches)]
        val_addrs = [self.alloc_scratch(f"val_addr_{i}") for i in range(n_batches)]

        offset_vals = [i * VLEN for i in range(n_batches)]
        offset_addrs = []
        for i in range(n_batches):
            addr = self.alloc_scratch(f"off_{i}")
            offset_addrs.append(addr)
            self.const_map[offset_vals[i]] = addr

        # Load all offset constants first (2 per cycle)
        for i in range(0, n_batches, 2):
            loads = [("const", offset_addrs[i], offset_vals[i])]
            if i + 1 < n_batches:
                loads.append(("const", offset_addrs[i + 1], offset_vals[i + 1]))
            self.instrs.append({"load": loads})

        # Then compute all addresses (12 ALU ops = 6 batches per cycle)
        for i in range(0, n_batches, 6):
            alu_ops = []
            for j in range(min(6, n_batches - i)):
                alu_ops.append(("+", idx_addrs[i+j], self.scratch["inp_indices_p"], offset_addrs[i+j]))
                alu_ops.append(("+", val_addrs[i+j], self.scratch["inp_values_p"], offset_addrs[i+j]))
            self.instrs.append({"alu": alu_ops})

        print(f"Scratch usage after init: {self.scratch_ptr}")

        # ===== MAIN KERNEL WITH ROUND TILING =====
        batches_processed = 0
        while batches_processed < n_batches:
            group_start = batches_processed
            group_end = min(group_start + GROUP_SIZE, n_batches)
            active = group_end - group_start

            # Check if there's a next group to preload
            next_group_start = group_start + GROUP_SIZE
            next_group_end = min(next_group_start + GROUP_SIZE, n_batches)
            next_active = next_group_end - next_group_start if next_group_start < n_batches else 0

            # Only interweave if next group is small (fits in tail of current group)
            should_interweave = next_active > 0 and next_active <= active // 4

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
                # Pass next group's load addresses for overlapping during tail
                next_load_addrs = [(idx_addrs[next_group_start + g], val_addrs[next_group_start + g])
                                   for g in range(next_active)] if is_last_tile and should_interweave else None

                self._process_tile_diagonal(
                    contexts, active, tile_start, tile_end, forest_height, n_nodes,
                    v_forest_nodes, v_hash_const1, v_hash_const2,
                    v_mult0, v_mult2, v_mult4, v_two, v_one, v_zero,
                    v_n_nodes, v_four, v_five, v_seven, v_forest_base,
                    load_addrs=load_addrs, store_addrs=store_addrs,
                    next_load_addrs=next_load_addrs
                )

            # Update processed count - include interweaved batches if any
            batches_processed = group_end
            if should_interweave:
                batches_processed += next_active  # Skip next group since we processed it

        self.instrs.append({"flow": [("pause",)]})

    def _process_tile_diagonal(
        self, contexts, active, tile_start, tile_end, forest_height, n_nodes,
        v_forest_nodes, v_hash_const1, v_hash_const2,
        v_mult0, v_mult2, v_mult4, v_two, v_one, v_zero,
        v_n_nodes, v_four, v_five, v_seven, v_forest_base,
        load_addrs=None, store_addrs=None, next_load_addrs=None
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

        # Track next group - these will be interleaved as current batches finish
        next_active = len(next_load_addrs) if next_load_addrs else 0
        # Map: context slot -> next-group batch index (or -1 if not assigned)
        next_batch_for_ctx = [-1] * active
        # Track next-group batch state
        next_batch_loaded = [False] * next_active
        next_batch_load_cycle = [-1] * next_active
        next_batch_done = [False] * next_active
        next_batch_stored = [False] * next_active
        next_batch_assigned = [False] * next_active  # Which next-group batches are assigned to contexts
        # Next-group batches use same stage tracking structure
        next_stage_done = [[-1] * 45 for _ in range(next_active)]
        next_current_round = [0] * next_active  # Start at round 0, not tile_start

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
        STAGE_H1P1A = 7   # First op of H1 prep (independent)
        STAGE_H1P1B = 38  # Second op of H1 prep (independent, can be same cycle)
        STAGE_H1 = 8
        STAGE_H2 = 9
        STAGE_H3P1A = 10
        STAGE_H3P1B = 39
        STAGE_H3 = 11
        STAGE_H4 = 12
        STAGE_H5P1A = 13
        STAGE_H5P1B = 41
        STAGE_H5 = 14
        STAGE_IDX_AND = 15
        STAGE_IDX_ADD = 16
        STAGE_OVF_LT = 17
        STAGE_OVF_SEL = 18
        STAGE_DONE = 19

        # For non-scattered with vselect (levels 1, 2, 3)
        # These use stages 20-39 to avoid overlap with scattered stages 0-19
        STAGE_AND1 = 20
        STAGE_VSEL1 = 21
        STAGE_LT = 22  # for level 2
        STAGE_VSEL2 = 23  # for level 2
        STAGE_VSEL3 = 24  # for level 2
        # Level 3 additional stages (7 vselects total)
        STAGE_L3_OFFSET = 25  # compute offset = idx - 7
        STAGE_L3_BIT0 = 26    # compute bit0 = offset & 1
        STAGE_L3_VSEL1A = 27
        STAGE_L3_VSEL1B = 28
        STAGE_L3_VSEL1C = 29
        STAGE_L3_VSEL1D = 30
        STAGE_L3_BIT1 = 31    # bit1_ind = offset & 2 (no shift needed)
        STAGE_L3_VSEL2A = 32
        STAGE_L3_VSEL2B = 33
        STAGE_L3_BIT2 = 34    # bit2_ind = offset & 4 (no shift needed)
        STAGE_L3_VSEL3 = 35

        # Track stage completion cycle for each batch
        stage_done = [[-1] * 45 for _ in range(active)]  # stage_done[batch][stage]
        batch_done = [False] * active

        def get_level(round_num):
            return round_num % (forest_height + 1)

        def is_scattered(level):
            return level >= 3  # Levels 0-2 cached, 3+ scattered

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

        # Continue until all current AND next-group batches are done
        def all_done():
            if not all(batch_done):
                return False
            if next_active > 0 and not all(next_batch_stored):
                return False
            return True

        while not all_done() and cycle < max_cycles:
            instr = {}
            valu_ops = []
            alu_ops = []
            load_ops = []
            store_ops = []
            flow_ops = []

            # Issue loads for batches that haven't been loaded yet - need space for BOTH
            if load_addrs:
                for g in range(active):
                    if not batch_loaded[g] and len(load_ops) == 0:
                        idx_addr, val_addr = load_addrs[g]
                        ctx = contexts[g]
                        load_ops.append(("vload", ctx["idx"], idx_addr))
                        load_ops.append(("vload", ctx["val"], val_addr))
                        batch_loaded[g] = True
                        batch_load_cycle[g] = cycle

            # Issue stores for completed batches - need space for BOTH stores
            if store_addrs:
                for g in range(active):
                    if batch_done[g] and not batch_stored[g] and len(store_ops) == 0:
                        idx_addr, val_addr = store_addrs[g]
                        ctx = contexts[g]
                        store_ops.append(("vstore", idx_addr, ctx["idx"]))
                        store_ops.append(("vstore", val_addr, ctx["val"]))
                        batch_stored[g] = True

            # Assign and load next-group batches into freed context slots
            if next_load_addrs:
                next_to_assign = 0  # Next unassigned next-group batch
                for g in range(active):
                    # Find next unassigned next-group batch
                    while next_to_assign < next_active and next_batch_assigned[next_to_assign]:
                        next_to_assign += 1
                    if next_to_assign >= next_active:
                        break
                    # Assign to freed context - need space for BOTH loads (idx and val)
                    if batch_stored[g] and next_batch_for_ctx[g] < 0 and len(load_ops) == 0:
                        idx_addr, val_addr = next_load_addrs[next_to_assign]
                        ctx = contexts[g]
                        load_ops.append(("vload", ctx["idx"], idx_addr))
                        load_ops.append(("vload", ctx["val"], val_addr))
                        next_batch_for_ctx[g] = next_to_assign
                        next_batch_assigned[next_to_assign] = True
                        next_batch_load_cycle[next_to_assign] = cycle
                        next_batch_loaded[next_to_assign] = True
                        # Initialize next-group batch stage tracking (starts at round 0)
                        init_level = get_level(0)  # Round 0 -> level 0
                        if is_scattered(init_level):
                            next_stage_done[next_to_assign][STAGE_ADDR] = -2
                        else:
                            next_stage_done[next_to_assign][STAGE_XOR] = -2
                        next_to_assign += 1

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
                                if len(load_ops) == 0:  # Need space for 2 loads
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

                    # Hash stages for scattered rounds
                    if stage_done[g][STAGE_XOR] >= 0 and stage_done[g][STAGE_XOR] < cycle:
                        if stage_done[g][STAGE_H0] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult0, v_hash_const1[0]))
                                stage_done[g][STAGE_H0] = cycle
                            continue

                    # H1P1: Two independent ops - can execute in same cycle or across cycles
                    if stage_done[g][STAGE_H0] >= 0 and stage_done[g][STAGE_H0] < cycle:
                        # Try op A if not done and slot available
                        if stage_done[g][STAGE_H1P1A] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[1][0], ctx["tmp1"], ctx["val"], v_hash_const1[1]))
                            stage_done[g][STAGE_H1P1A] = cycle
                        # Try op B if not done and slot available
                        if stage_done[g][STAGE_H1P1B] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[1][3], ctx["tmp2"], ctx["val"], v_hash_const2[1]))
                            stage_done[g][STAGE_H1P1B] = cycle
                        # Wait if either op still needs doing
                        if stage_done[g][STAGE_H1P1A] < 0 or stage_done[g][STAGE_H1P1B] < 0:
                            continue

                    # H1 proceeds when BOTH prep ops are done
                    if stage_done[g][STAGE_H1P1A] >= 0 and stage_done[g][STAGE_H1P1A] < cycle and \
                       stage_done[g][STAGE_H1P1B] >= 0 and stage_done[g][STAGE_H1P1B] < cycle:
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

                    # H3P1: Two independent ops - can execute in same cycle or across cycles
                    if stage_done[g][STAGE_H2] >= 0 and stage_done[g][STAGE_H2] < cycle:
                        # Try op A if not done and slot available
                        if stage_done[g][STAGE_H3P1A] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[3][0], ctx["tmp1"], ctx["val"], v_hash_const1[3]))
                            stage_done[g][STAGE_H3P1A] = cycle
                        # Try op B if not done and slot available
                        if stage_done[g][STAGE_H3P1B] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[3][3], ctx["tmp2"], ctx["val"], v_hash_const2[3]))
                            stage_done[g][STAGE_H3P1B] = cycle
                        # Wait if either op still needs doing
                        if stage_done[g][STAGE_H3P1A] < 0 or stage_done[g][STAGE_H3P1B] < 0:
                            continue

                    # H3 proceeds when BOTH prep ops are done
                    if stage_done[g][STAGE_H3P1A] >= 0 and stage_done[g][STAGE_H3P1A] < cycle and \
                       stage_done[g][STAGE_H3P1B] >= 0 and stage_done[g][STAGE_H3P1B] < cycle:
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

                    # H5P1: Two independent ops - can execute in same cycle or across cycles
                    if stage_done[g][STAGE_H4] >= 0 and stage_done[g][STAGE_H4] < cycle:
                        # Try op A if not done and slot available
                        if stage_done[g][STAGE_H5P1A] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[5][0], ctx["tmp1"], ctx["val"], v_hash_const1[5]))
                            stage_done[g][STAGE_H5P1A] = cycle
                        # Try op B if not done and slot available
                        if stage_done[g][STAGE_H5P1B] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[5][3], ctx["tmp2"], ctx["val"], v_hash_const2[5]))
                            stage_done[g][STAGE_H5P1B] = cycle
                        # Wait if either op still needs doing
                        if stage_done[g][STAGE_H5P1A] < 0 or stage_done[g][STAGE_H5P1B] < 0:
                            continue

                    # H5 proceeds when BOTH prep ops are done
                    if stage_done[g][STAGE_H5P1A] >= 0 and stage_done[g][STAGE_H5P1A] < cycle and \
                       stage_done[g][STAGE_H5P1B] >= 0 and stage_done[g][STAGE_H5P1B] < cycle:
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
                                for s in range(45):
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
                                for s in range(45):
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

                    elif level == 3:
                        # Level 3: vselect tree for 8 nodes (7-14)
                        # idx is in [7,14], compute offset = idx - 7 to get [0,7]
                        # Use bits of offset to select from the 8 nodes

                        # Step 1a: offset = idx - 7 (store in tmp4)
                        if stage_done[g][STAGE_L3_OFFSET] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("-", ctx["tmp4"], ctx["idx"], v_seven))
                                stage_done[g][STAGE_L3_OFFSET] = cycle
                            continue

                        # Step 1b: bit0 = offset & 1 (need to wait for offset)
                        if stage_done[g][STAGE_L3_OFFSET] >= 0 and stage_done[g][STAGE_L3_OFFSET] < cycle:
                            if stage_done[g][STAGE_L3_BIT0] < 0:
                                if len(valu_ops) < 6:
                                    valu_ops.append(("&", ctx["tmp1"], ctx["tmp4"], v_one))
                                    stage_done[g][STAGE_L3_BIT0] = cycle
                                continue

                        # Step 2: Four vselects for pairs based on bit0
                        # vselect semantics: if cond != 0, pick val_true, else val_false
                        # For pair (node7, node8): if bit0=1 → node8, else → node7
                        if stage_done[g][STAGE_L3_BIT0] >= 0 and stage_done[g][STAGE_L3_BIT0] < cycle:
                            if stage_done[g][STAGE_L3_VSEL1A] < 0:
                                if len(flow_ops) < 1:
                                    # pair0: nodes 7,8 - offsets 0,1
                                    flow_ops.append(("vselect", ctx["tmp2"], ctx["tmp1"],
                                                   v_forest_nodes[8], v_forest_nodes[7]))
                                    stage_done[g][STAGE_L3_VSEL1A] = cycle
                                continue

                        if stage_done[g][STAGE_L3_VSEL1A] >= 0 and stage_done[g][STAGE_L3_VSEL1A] < cycle:
                            if stage_done[g][STAGE_L3_VSEL1B] < 0:
                                if len(flow_ops) < 1:
                                    # pair1: nodes 9,10 - offsets 2,3
                                    flow_ops.append(("vselect", ctx["node"], ctx["tmp1"],
                                                   v_forest_nodes[10], v_forest_nodes[9]))
                                    stage_done[g][STAGE_L3_VSEL1B] = cycle
                                continue

                        if stage_done[g][STAGE_L3_VSEL1B] >= 0 and stage_done[g][STAGE_L3_VSEL1B] < cycle:
                            if stage_done[g][STAGE_L3_VSEL1C] < 0:
                                if len(flow_ops) < 1:
                                    # pair2: nodes 11,12 - offsets 4,5
                                    flow_ops.append(("vselect", ctx["tmp3"], ctx["tmp1"],
                                                   v_forest_nodes[12], v_forest_nodes[11]))
                                    stage_done[g][STAGE_L3_VSEL1C] = cycle
                                continue

                        if stage_done[g][STAGE_L3_VSEL1C] >= 0 and stage_done[g][STAGE_L3_VSEL1C] < cycle:
                            if stage_done[g][STAGE_L3_VSEL1D] < 0:
                                if len(flow_ops) < 1:
                                    # pair3: nodes 13,14 - offsets 6,7
                                    # Store in addr (unused in non-scattered path)
                                    flow_ops.append(("vselect", ctx["addr"], ctx["tmp1"],
                                                   v_forest_nodes[14], v_forest_nodes[13]))
                                    stage_done[g][STAGE_L3_VSEL1D] = cycle
                                continue

                        # Step 3: bit1_ind = offset & 2 (non-zero when bit1 is set)
                        # No shift needed - vselect just checks for non-zero
                        if stage_done[g][STAGE_L3_VSEL1D] >= 0 and stage_done[g][STAGE_L3_VSEL1D] < cycle:
                            if stage_done[g][STAGE_L3_BIT1] < 0:
                                if len(valu_ops) < 6:
                                    valu_ops.append(("&", ctx["tmp1"], ctx["tmp4"], v_two))
                                    stage_done[g][STAGE_L3_BIT1] = cycle
                                continue

                        # Step 4: Two vselects for quads based on bit1
                        # quad0: pairs 0,1 (offsets 0-3) - if bit1=1 → pair1(node), else → pair0(tmp2)
                        if stage_done[g][STAGE_L3_BIT1] >= 0 and stage_done[g][STAGE_L3_BIT1] < cycle:
                            if stage_done[g][STAGE_L3_VSEL2A] < 0:
                                if len(flow_ops) < 1:
                                    flow_ops.append(("vselect", ctx["tmp2"], ctx["tmp1"],
                                                   ctx["node"], ctx["tmp2"]))
                                    stage_done[g][STAGE_L3_VSEL2A] = cycle
                                continue

                        # quad1: pairs 2,3 (offsets 4-7) - if bit1=1 → pair3(addr), else → pair2(tmp3)
                        if stage_done[g][STAGE_L3_VSEL2A] >= 0 and stage_done[g][STAGE_L3_VSEL2A] < cycle:
                            if stage_done[g][STAGE_L3_VSEL2B] < 0:
                                if len(flow_ops) < 1:
                                    flow_ops.append(("vselect", ctx["tmp3"], ctx["tmp1"],
                                                   ctx["addr"], ctx["tmp3"]))
                                    stage_done[g][STAGE_L3_VSEL2B] = cycle
                                continue

                        # Step 5: bit2_ind = offset & 4 (non-zero when bit2 is set)
                        # No shift needed - vselect just checks for non-zero
                        if stage_done[g][STAGE_L3_VSEL2B] >= 0 and stage_done[g][STAGE_L3_VSEL2B] < cycle:
                            if stage_done[g][STAGE_L3_BIT2] < 0:
                                if len(valu_ops) < 6:
                                    valu_ops.append(("&", ctx["tmp1"], ctx["tmp4"], v_four))
                                    stage_done[g][STAGE_L3_BIT2] = cycle
                                continue

                        # Step 6: Final vselect - if bit2=1 → quad1(tmp3), else → quad0(tmp2)
                        if stage_done[g][STAGE_L3_BIT2] >= 0 and stage_done[g][STAGE_L3_BIT2] < cycle:
                            if stage_done[g][STAGE_L3_VSEL3] < 0:
                                if len(flow_ops) < 1:
                                    flow_ops.append(("vselect", ctx["tmp2"], ctx["tmp1"],
                                                   ctx["tmp3"], ctx["tmp2"]))
                                    stage_done[g][STAGE_L3_VSEL3] = cycle
                                continue

                        # Step 7: XOR
                        if stage_done[g][STAGE_L3_VSEL3] >= 0 and stage_done[g][STAGE_L3_VSEL3] < cycle:
                            if stage_done[g][STAGE_XOR] < 0:
                                if len(alu_ops) <= 4:
                                    for i in range(8):
                                        alu_ops.append(("^", ctx["val"] + i, ctx["val"] + i, ctx["tmp2"] + i))
                                    stage_done[g][STAGE_XOR] = cycle
                                continue

                    # Hash stages for non-scattered rounds
                    if stage_done[g][STAGE_XOR] >= 0 and stage_done[g][STAGE_XOR] < cycle:
                        if stage_done[g][STAGE_H0] < 0:
                            if len(valu_ops) < 6:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult0, v_hash_const1[0]))
                                stage_done[g][STAGE_H0] = cycle
                            continue

                    # H1P1: Two independent ops - can execute in same cycle or across cycles
                    if stage_done[g][STAGE_H0] >= 0 and stage_done[g][STAGE_H0] < cycle:
                        # Try op A if not done and slot available
                        if stage_done[g][STAGE_H1P1A] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[1][0], ctx["tmp1"], ctx["val"], v_hash_const1[1]))
                            stage_done[g][STAGE_H1P1A] = cycle
                        # Try op B if not done and slot available
                        if stage_done[g][STAGE_H1P1B] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[1][3], ctx["tmp2"], ctx["val"], v_hash_const2[1]))
                            stage_done[g][STAGE_H1P1B] = cycle
                        # Wait if either op still needs doing
                        if stage_done[g][STAGE_H1P1A] < 0 or stage_done[g][STAGE_H1P1B] < 0:
                            continue

                    # H1 proceeds when BOTH prep ops are done
                    if stage_done[g][STAGE_H1P1A] >= 0 and stage_done[g][STAGE_H1P1A] < cycle and \
                       stage_done[g][STAGE_H1P1B] >= 0 and stage_done[g][STAGE_H1P1B] < cycle:
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

                    # H3P1: Two independent ops - can execute in same cycle or across cycles
                    if stage_done[g][STAGE_H2] >= 0 and stage_done[g][STAGE_H2] < cycle:
                        # Try op A if not done and slot available
                        if stage_done[g][STAGE_H3P1A] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[3][0], ctx["tmp1"], ctx["val"], v_hash_const1[3]))
                            stage_done[g][STAGE_H3P1A] = cycle
                        # Try op B if not done and slot available
                        if stage_done[g][STAGE_H3P1B] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[3][3], ctx["tmp2"], ctx["val"], v_hash_const2[3]))
                            stage_done[g][STAGE_H3P1B] = cycle
                        # Wait if either op still needs doing
                        if stage_done[g][STAGE_H3P1A] < 0 or stage_done[g][STAGE_H3P1B] < 0:
                            continue

                    # H3 proceeds when BOTH prep ops are done
                    if stage_done[g][STAGE_H3P1A] >= 0 and stage_done[g][STAGE_H3P1A] < cycle and \
                       stage_done[g][STAGE_H3P1B] >= 0 and stage_done[g][STAGE_H3P1B] < cycle:
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

                    # H5P1: Two independent ops - can execute in same cycle or across cycles
                    if stage_done[g][STAGE_H4] >= 0 and stage_done[g][STAGE_H4] < cycle:
                        # Try op A if not done and slot available
                        if stage_done[g][STAGE_H5P1A] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[5][0], ctx["tmp1"], ctx["val"], v_hash_const1[5]))
                            stage_done[g][STAGE_H5P1A] = cycle
                        # Try op B if not done and slot available
                        if stage_done[g][STAGE_H5P1B] < 0 and len(valu_ops) < 6:
                            valu_ops.append((HASH_STAGES[5][3], ctx["tmp2"], ctx["val"], v_hash_const2[5]))
                            stage_done[g][STAGE_H5P1B] = cycle
                        # Wait if either op still needs doing
                        if stage_done[g][STAGE_H5P1A] < 0 or stage_done[g][STAGE_H5P1B] < 0:
                            continue

                    # H5 proceeds when BOTH prep ops are done
                    if stage_done[g][STAGE_H5P1A] >= 0 and stage_done[g][STAGE_H5P1A] < cycle and \
                       stage_done[g][STAGE_H5P1B] >= 0 and stage_done[g][STAGE_H5P1B] < cycle:
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
                            for s in range(45):
                                stage_done[g][s] = -1
                            next_level = get_level(current_round[g])
                            if is_scattered(next_level):
                                stage_done[g][STAGE_ADDR] = -2
                            else:
                                stage_done[g][STAGE_XOR] = -2

            # Process next-group batches that are assigned to freed contexts
            for g in range(active):
                nb = next_batch_for_ctx[g]
                if nb < 0:
                    continue
                if next_batch_done[nb]:
                    # Store if not stored yet
                    if not next_batch_stored[nb] and len(store_ops) < 2:
                        idx_addr, val_addr = next_load_addrs[nb]
                        ctx = contexts[g]
                        store_ops.append(("vstore", idx_addr, ctx["idx"]))
                        if len(store_ops) < 2:
                            store_ops.append(("vstore", val_addr, ctx["val"]))
                            next_batch_stored[nb] = True
                    continue

                # Skip if batch hasn't been loaded yet (need 1 cycle latency)
                if not next_batch_loaded[nb] or next_batch_load_cycle[nb] >= cycle:
                    continue

                ctx = contexts[g]
                rnd = next_current_round[nb]
                n_rounds_next = 16  # Process all 16 rounds

                if rnd >= n_rounds_next:
                    next_batch_done[nb] = True
                    continue

                level = get_level(rnd)
                need_scattered = is_scattered(level)
                need_overflow = needs_overflow(level)

                # Use same processing logic but with next_stage_done array
                sd = next_stage_done[nb]

                if need_scattered:
                    # Scattered round processing for next-group batch
                    if sd[STAGE_ADDR] == -2:
                        if len(alu_ops) <= 4:
                            for i in range(8):
                                alu_ops.append(("+", ctx["addr"] + i, self.scratch["forest_values_p"], ctx["idx"] + i))
                            sd[STAGE_ADDR] = cycle
                        continue

                    if sd[STAGE_ADDR] >= 0 and sd[STAGE_ADDR] < cycle:
                        for load_stage in range(STAGE_LOAD0, STAGE_LOAD3 + 1):
                            if sd[load_stage] < 0:
                                if len(load_ops) == 0:  # Need space for 2 loads
                                    i = (load_stage - STAGE_LOAD0) * 2
                                    load_ops.append(("load", ctx["node"] + i, ctx["addr"] + i))
                                    load_ops.append(("load", ctx["node"] + i + 1, ctx["addr"] + i + 1))
                                    sd[load_stage] = cycle
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

                    # Hash stages H0-H5
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

                    if sd[STAGE_H1P1A] >= 0 and sd[STAGE_H1P1A] < cycle and sd[STAGE_H1P1B] >= 0 and sd[STAGE_H1P1B] < cycle:
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

                    if sd[STAGE_H3P1A] >= 0 and sd[STAGE_H3P1A] < cycle and sd[STAGE_H3P1B] >= 0 and sd[STAGE_H3P1B] < cycle:
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

                    if sd[STAGE_H5P1A] >= 0 and sd[STAGE_H5P1A] < cycle and sd[STAGE_H5P1B] >= 0 and sd[STAGE_H5P1B] < cycle:
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

                    if need_overflow:
                        if sd[STAGE_IDX_ADD] >= 0 and sd[STAGE_IDX_ADD] < cycle:
                            if sd[STAGE_OVF_LT] < 0:
                                if len(valu_ops) < 6:
                                    valu_ops.append(("<", ctx["tmp1"], ctx["idx"], v_n_nodes))
                                    sd[STAGE_OVF_LT] = cycle
                                continue
                        if sd[STAGE_OVF_LT] >= 0 and sd[STAGE_OVF_LT] < cycle:
                            if sd[STAGE_OVF_SEL] < 0:
                                if len(flow_ops) < 1:
                                    flow_ops.append(("vselect", ctx["idx"], ctx["tmp1"], ctx["idx"], v_zero))
                                    sd[STAGE_OVF_SEL] = cycle
                                continue
                        if sd[STAGE_OVF_SEL] >= 0 and sd[STAGE_OVF_SEL] < cycle:
                            next_current_round[nb] += 1
                            if next_current_round[nb] < n_rounds_next:
                                for s in range(45):
                                    sd[s] = -1
                                next_level = get_level(next_current_round[nb])
                                if is_scattered(next_level):
                                    sd[STAGE_ADDR] = -2
                                else:
                                    sd[STAGE_XOR] = -2
                            continue
                    else:
                        if sd[STAGE_IDX_ADD] >= 0 and sd[STAGE_IDX_ADD] < cycle:
                            next_current_round[nb] += 1
                            if next_current_round[nb] < n_rounds_next:
                                for s in range(45):
                                    sd[s] = -1
                                next_level = get_level(next_current_round[nb])
                                if is_scattered(next_level):
                                    sd[STAGE_ADDR] = -2
                                else:
                                    sd[STAGE_XOR] = -2
                            continue

                else:
                    # Non-scattered round processing for next-group batch
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

                    if sd[STAGE_H1P1A] >= 0 and sd[STAGE_H1P1A] < cycle and sd[STAGE_H1P1B] >= 0 and sd[STAGE_H1P1B] < cycle:
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

                    if sd[STAGE_H3P1A] >= 0 and sd[STAGE_H3P1A] < cycle and sd[STAGE_H3P1B] >= 0 and sd[STAGE_H3P1B] < cycle:
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

                    if sd[STAGE_H5P1A] >= 0 and sd[STAGE_H5P1A] < cycle and sd[STAGE_H5P1B] >= 0 and sd[STAGE_H5P1B] < cycle:
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
                        next_current_round[nb] += 1
                        if next_current_round[nb] < n_rounds_next:
                            for s in range(45):
                                sd[s] = -1
                            next_level = get_level(next_current_round[nb])
                            if is_scattered(next_level):
                                sd[STAGE_ADDR] = -2
                            else:
                                sd[STAGE_XOR] = -2

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

        # Emit any remaining stores for next-group batches
        if next_load_addrs:
            for g in range(active):
                nb = next_batch_for_ctx[g]
                if nb >= 0 and not next_batch_stored[nb]:
                    ctx = contexts[g]
                    idx_addr, val_addr = next_load_addrs[nb]
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
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


if __name__ == "__main__":
    do_kernel_test(10, 16, 256, trace=True)
