"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
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
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
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
        Optimized kernel with scattered load pipelining.
        """
        self.forest_height = forest_height
        n_batches = batch_size // VLEN
        GROUP_SIZE = 24  # Process batches in groups (limited by scratch space)

        def emit_valu_chunked(ops):
            """Emit VALU ops in chunks of 6 (max VALU slots)"""
            for i in range(0, len(ops), 6):
                self.instrs.append({"valu": ops[i:i+6]})

        # ===== ALLOCATE CONTEXTS =====
        contexts = []
        for g in range(GROUP_SIZE):
            ctx = {
                "idx": self.alloc_vec(f"ctx{g}_idx"),
                "val": self.alloc_vec(f"ctx{g}_val"),
                "node": self.alloc_vec(f"ctx{g}_node"),
                "tmp1": self.alloc_vec(f"ctx{g}_tmp1"),
                "tmp2": self.alloc_vec(f"ctx{g}_tmp2"),
                "addrs": [self.alloc_scratch(f"ctx{g}_addr_{i}") for i in range(VLEN)],
            }
            contexts.append(ctx)

        # Constants
        v_zero = self.alloc_vec("v_zero")
        v_one = self.alloc_vec("v_one")
        v_two = self.alloc_vec("v_two")
        v_five = self.alloc_vec("v_five")
        v_n_nodes = self.alloc_vec("v_n_nodes")

        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Pre-allocate init vars
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Pre-allocate scalar constants (without emitting loads yet)
        const_addrs = {}
        for val in [0, 1, 2, 5]:
            const_addrs[val] = self.alloc_scratch(f"c_{val}")
            self.const_map[val] = const_addrs[val]

        # Batch the const loads for init_vars, interleave scalar consts with last partial batch
        const_vals = [0, 1, 2, 5]
        const_idx = 0
        for i in range(0, len(init_vars), 2):
            loads = [("const", tmp1, i)]
            if i + 1 < len(init_vars):
                loads.append(("const", tmp2, i + 1))
            else:
                # Last iteration has only 1 init_var - pack with first scalar const
                loads.append(("const", const_addrs[const_vals[const_idx]], const_vals[const_idx]))
                const_idx += 1
            self.instrs.append({"load": loads})
            # Now load from memory
            loads2 = [("load", self.scratch[init_vars[i]], tmp1)]
            if i + 1 < len(init_vars):
                loads2.append(("load", self.scratch[init_vars[i + 1]], tmp2))
            else:
                # Pack with second scalar const
                loads2.append(("const", const_addrs[const_vals[const_idx]], const_vals[const_idx]))
                const_idx += 1
            self.instrs.append({"load": loads2})

        # Load remaining scalar constants
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
        ]})

        self.instrs.append({"flow": [("pause",)], "debug": [("comment", "Starting loop")]})

        # ===== HASH CONSTANTS =====
        v_hash_const1 = [self.alloc_vec(f"v_hash{i}_const1") for i in range(6)]
        v_hash_const2 = [self.alloc_vec(f"v_hash{i}_const2") for i in range(6)]

        v_mult0 = self.alloc_vec("v_mult0")
        v_mult2 = self.alloc_vec("v_mult2")
        v_mult4 = self.alloc_vec("v_mult4")

        # Pre-allocate all hash and mult constants, then batch load them
        hash_const1_vals = [HASH_STAGES[i][1] for i in range(6)]
        hash_const2_vals = [HASH_STAGES[i][4] for i in range(6)]
        mult_vals = [1 + (1 << 12), 1 + (1 << 5), 1 + (1 << 3)]
        all_const_vals = hash_const1_vals + hash_const2_vals + mult_vals

        # Pre-allocate addresses for all constants
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
        # Allocate early so we can overlap vload with hash const loads
        v_forest_nodes = [self.alloc_vec(f"v_forest_node_{i}") for i in range(7)]
        forest_cache = self.alloc_scratch("forest_cache", 8)

        # Batch load all 15 constants (2 per cycle), overlap vload with last one
        all_addrs = hash_c1_addrs + hash_c2_addrs + mult_addrs
        for i in range(0, len(all_addrs), 2):
            loads = [("const", all_addrs[i], all_const_vals[i])]
            if i + 1 < len(all_addrs):
                loads.append(("const", all_addrs[i + 1], all_const_vals[i + 1]))
            else:
                # Last iteration has only 1 const - pair with forest_cache vload
                loads.append(("vload", forest_cache, self.scratch["forest_values_p"]))
            self.instrs.append({"load": loads})

        # Vbroadcasts (6 per cycle) - vload completes during these 3 cycles
        self.instrs.append({"valu": [
            ("vbroadcast", v_hash_const1[i], hash_c1_addrs[i])
            for i in range(6)
        ]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_hash_const2[i], hash_c2_addrs[i])
            for i in range(6)
        ]})
        # Combine mult vbroadcasts with forest_node[6] (4 ops, within 6 VALU limit)
        self.instrs.append({"valu": [
            ("vbroadcast", v_mult0, mult_addrs[0]),
            ("vbroadcast", v_mult2, mult_addrs[1]),
            ("vbroadcast", v_mult4, mult_addrs[2]),
            ("vbroadcast", v_forest_nodes[6], forest_cache + 6),
        ]})
        # forest_nodes 0-5 (after vload has completed)
        self.instrs.append({"valu": [
            ("vbroadcast", v_forest_nodes[i], forest_cache + i) for i in range(6)
        ]})

        # ===== PRECOMPUTE BATCH ADDRESSES =====
        idx_addrs = [self.alloc_scratch(f"idx_addr_{i}") for i in range(n_batches)]
        val_addrs = [self.alloc_scratch(f"val_addr_{i}") for i in range(n_batches)]

        # Pre-allocate offset constants and batch load them
        offset_vals = [i * VLEN for i in range(n_batches)]
        offset_addrs = []
        for i in range(n_batches):
            addr = self.alloc_scratch(f"off_{i}")
            offset_addrs.append(addr)
            self.const_map[offset_vals[i]] = addr

        # Batch load offset constants (2 per cycle)
        for i in range(0, n_batches, 2):
            loads = [("const", offset_addrs[i], offset_vals[i])]
            if i + 1 < n_batches:
                loads.append(("const", offset_addrs[i + 1], offset_vals[i + 1]))
            self.instrs.append({"load": loads})

        # Now compute addresses using pre-loaded offsets
        for i in range(0, n_batches, 6):
            alu_ops = []
            for j in range(min(6, n_batches - i)):
                alu_ops.append(("+", idx_addrs[i+j], self.scratch["inp_indices_p"], offset_addrs[i+j]))
                alu_ops.append(("+", val_addrs[i+j], self.scratch["inp_values_p"], offset_addrs[i+j]))
            self.instrs.append({"alu": alu_ops})

        # ===== MAIN KERNEL =====
        for group_start in range(0, n_batches, GROUP_SIZE):
            group_end = min(group_start + GROUP_SIZE, n_batches)
            active = group_end - group_start

            # Load batches - overlap last few with round 0 XOR
            # We need ceil(active/6) cycles to XOR all batches
            # So load active - ceil(active/6) batches first, then overlap the rest
            xor_cycles_needed = (active + 5) // 6  # ceil division
            load_overlap_start = max(0, active - xor_cycles_needed)

            for g in range(load_overlap_start):
                batch = group_start + g
                ctx = contexts[g]
                self.instrs.append({"load": [
                    ("vload", ctx["idx"], idx_addrs[batch]),
                    ("vload", ctx["val"], val_addrs[batch]),
                ]})

            # Overlap remaining loads with round 0 XOR (level 0)
            xor_idx = 0
            for g in range(load_overlap_start, active):
                batch = group_start + g
                ctx = contexts[g]
                instr = {"load": [
                    ("vload", ctx["idx"], idx_addrs[batch]),
                    ("vload", ctx["val"], val_addrs[batch]),
                ]}
                # Add XOR ops for earlier batches (up to 6 per cycle)
                if xor_idx < load_overlap_start:
                    xor_ops = []
                    for _ in range(min(6, load_overlap_start - xor_idx)):
                        xor_ops.append(("^", contexts[xor_idx]["val"],
                                       contexts[xor_idx]["val"], v_forest_nodes[0]))
                        xor_idx += 1
                    if xor_ops:
                        instr["valu"] = xor_ops
                self.instrs.append(instr)

            # Track how many batches were XORed during loading
            round0_xor_done = xor_idx

            # Process rounds with cross-round pipelining for consecutive scattered rounds
            round_num = 0
            # Track addr calc done from previous round (for pipelining)
            prev_round_addr_done = None
            prev_round_loads_done = None
            while round_num < rounds:
                level = round_num % (forest_height + 1)
                need_scattered = level >= 3
                need_overflow_check = (level == forest_height)

                # Check if next round is also scattered (for pipelining)
                next_level = (round_num + 1) % (forest_height + 1) if round_num + 1 < rounds else -1
                next_scattered = next_level >= 3 if round_num + 1 < rounds else False

                # Scattered loads with interleaved hash computation
                # Pack loads with hash stages from earlier batches
                if need_scattered:
                    # Comprehensive scheduler for all operations
                    # Operations per batch: addr_calc -> 4 loads -> xor -> hash[0-5] -> idx_update -> overflow_check

                    # Track completion cycles (using arrays instead of object attributes)
                    loads_done = [-1] * active
                    xor_done = [-1] * active
                    h1_final = [-1] * active
                    h3_p1 = [-1] * active
                    h5_p1 = [-1] * active
                    idx_and = [-1] * active
                    ovf_lt = [-1] * active
                    hash_done = [[-1] * 6 for _ in range(active)]  # hash_done[batch][stage]
                    idx_update_done = [-1] * active
                    all_done = [False] * active

                    # For cross-round pipelining: track next round's addr calc
                    next_round_addr_done = [-1] * active if next_scattered else None
                    next_round_addr_idx = 0

                    load_queue = []  # (batch, pair_idx)
                    cycle = 0

                    # Use pre-computed addr calc and loads from previous round if available
                    addr_done = [-1] * active
                    # Interleave batch order: 0, 12, 1, 13, 2, 14, ... for better pipeline utilization
                    interleaved = []
                    for i in range(active // 2):
                        interleaved.append(i)
                        if i + active // 2 < active:
                            interleaved.append(i + active // 2)
                    addr_calc_queue = interleaved[:]  # Batches needing addr calc
                    if prev_round_addr_done is not None:
                        # Mark pre-computed batches as done and queue their loads (if not pre-loaded)
                        for g in range(active):
                            if prev_round_addr_done[g] >= 0:
                                addr_done[g] = -1  # "done before this round"
                                # Check if loads were also pre-done
                                if prev_round_loads_done is not None and prev_round_loads_done[g] >= 0:
                                    loads_done[g] = -1  # "done before this round"
                                else:
                                    for pair in range(4):
                                        load_queue.append((g, pair))
                                if g in addr_calc_queue:
                                    addr_calc_queue.remove(g)  # Don't need addr calc
                        prev_round_addr_done = None
                        prev_round_loads_done = None

                    while not all(all_done):
                        instr = {}
                        valu_ops = []

                        # 1. Schedule addr calc (8 ALU ops) - one per cycle
                        # First priority: current round addr calc
                        if addr_calc_queue:
                            g = addr_calc_queue.pop(0)
                            ctx = contexts[g]
                            instr["alu"] = [
                                ("+", ctx["addrs"][i], self.scratch["forest_values_p"], ctx["idx"] + i)
                                for i in range(8)
                            ]
                            addr_done[g] = cycle
                            for pair in range(4):
                                load_queue.append((g, pair))
                        # Second priority: next round addr calc (during hash tail)
                        elif next_scattered and next_round_addr_idx < active:
                            # Find a batch whose idx is ready for next round
                            for g in range(next_round_addr_idx, active):
                                if idx_update_done[g] >= 0 and idx_update_done[g] < cycle:
                                    ctx = contexts[g]
                                    instr["alu"] = [
                                        ("+", ctx["addrs"][i], self.scratch["forest_values_p"], ctx["idx"] + i)
                                        for i in range(8)
                                    ]
                                    next_round_addr_done[g] = cycle
                                    next_round_addr_idx = g + 1
                                    break

                        # 2. Schedule loads (1 pair = 2 ops per cycle)
                        if load_queue:
                            batch, pair = load_queue[0]
                            if addr_done[batch] < cycle:  # addr must be done in previous cycle
                                load_queue.pop(0)
                                ctx = contexts[batch]
                                i = pair * 2
                                instr["load"] = [
                                    ("load", ctx["node"] + i, ctx["addrs"][i]),
                                    ("load", ctx["node"] + i + 1, ctx["addrs"][i + 1]),
                                ]
                                if pair == 3:
                                    loads_done[batch] = cycle

                        # 3. Schedule VALU ops: XOR, hash stages, idx update
                        # Process batches in interleaved order for better VALU packing
                        for g in interleaved:
                            if len(valu_ops) >= 6:
                                break
                            ctx = contexts[g]

                            # XOR needs loads done in previous cycle
                            if xor_done[g] < 0 and loads_done[g] >= 0 and loads_done[g] < cycle:
                                valu_ops.append(("^", ctx["val"], ctx["val"], ctx["node"]))
                                xor_done[g] = cycle
                                continue

                            # Hash stage 0
                            if hash_done[g][0] < 0 and xor_done[g] >= 0 and xor_done[g] < cycle:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult0, v_hash_const1[0]))
                                hash_done[g][0] = cycle
                                continue

                            # Hash stage 1 part 1 - need 2 VALUs
                            if hash_done[g][1] < 0 and hash_done[g][0] >= 0 and hash_done[g][0] < cycle:
                                if len(valu_ops) <= 4:  # Need space for 2 ops
                                    valu_ops.append((HASH_STAGES[1][0], ctx["tmp1"], ctx["val"], v_hash_const1[1]))
                                    valu_ops.append((HASH_STAGES[1][3], ctx["tmp2"], ctx["val"], v_hash_const2[1]))
                                    hash_done[g][1] = cycle
                                continue

                            # Hash stage 1 final XOR
                            if h1_final[g] < 0 and hash_done[g][1] >= 0 and hash_done[g][1] < cycle:
                                valu_ops.append((HASH_STAGES[1][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                h1_final[g] = cycle
                                continue

                            # Stage 2 + idx update
                            if hash_done[g][2] < 0 and h1_final[g] >= 0 and h1_final[g] < cycle:
                                if len(valu_ops) <= 4:
                                    valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult2, v_hash_const1[2]))
                                    valu_ops.append(("multiply_add", ctx["idx"], ctx["idx"], v_two, v_one))
                                    hash_done[g][2] = cycle
                                continue

                            # Stage 3 part 1
                            if h3_p1[g] < 0 and hash_done[g][2] >= 0 and hash_done[g][2] < cycle:
                                if len(valu_ops) <= 4:
                                    valu_ops.append((HASH_STAGES[3][0], ctx["tmp1"], ctx["val"], v_hash_const1[3]))
                                    valu_ops.append((HASH_STAGES[3][3], ctx["tmp2"], ctx["val"], v_hash_const2[3]))
                                    h3_p1[g] = cycle
                                continue

                            # Stage 3 final
                            if hash_done[g][3] < 0 and h3_p1[g] >= 0 and h3_p1[g] < cycle:
                                valu_ops.append((HASH_STAGES[3][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                hash_done[g][3] = cycle
                                continue

                            # Stage 4
                            if hash_done[g][4] < 0 and hash_done[g][3] >= 0 and hash_done[g][3] < cycle:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult4, v_hash_const1[4]))
                                hash_done[g][4] = cycle
                                continue

                            # Stage 5 part 1
                            if h5_p1[g] < 0 and hash_done[g][4] >= 0 and hash_done[g][4] < cycle:
                                if len(valu_ops) <= 4:
                                    valu_ops.append((HASH_STAGES[5][0], ctx["tmp1"], ctx["val"], v_hash_const1[5]))
                                    valu_ops.append((HASH_STAGES[5][3], ctx["tmp2"], ctx["val"], v_hash_const2[5]))
                                    h5_p1[g] = cycle
                                continue

                            # Stage 5 final
                            if hash_done[g][5] < 0 and h5_p1[g] >= 0 and h5_p1[g] < cycle:
                                valu_ops.append((HASH_STAGES[5][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                hash_done[g][5] = cycle
                                continue

                            # Index update: & then +
                            if idx_and[g] < 0 and hash_done[g][5] >= 0 and hash_done[g][5] < cycle:
                                valu_ops.append(("&", ctx["tmp1"], ctx["val"], v_one))
                                idx_and[g] = cycle
                                continue

                            if idx_update_done[g] < 0 and idx_and[g] >= 0 and idx_and[g] < cycle:
                                valu_ops.append(("+", ctx["idx"], ctx["idx"], ctx["tmp1"]))
                                idx_update_done[g] = cycle
                                continue

                            # Overflow check (only at level 10)
                            if need_overflow_check and idx_update_done[g] >= 0 and idx_update_done[g] < cycle:
                                if ovf_lt[g] < 0:
                                    valu_ops.append(("<", ctx["tmp1"], ctx["idx"], v_n_nodes))
                                    ovf_lt[g] = cycle
                                    continue

                            # Mark done if no overflow check needed
                            if not need_overflow_check and idx_update_done[g] >= 0 and idx_update_done[g] < cycle:
                                all_done[g] = True

                        if valu_ops:
                            instr["valu"] = valu_ops[:6]

                        # Handle vselect for overflow check
                        if need_overflow_check:
                            for g in range(active):
                                ctx = contexts[g]
                                if ovf_lt[g] >= 0 and ovf_lt[g] < cycle and not all_done[g]:
                                    if "flow" not in instr:
                                        instr["flow"] = [("vselect", ctx["idx"], ctx["tmp1"], ctx["idx"], v_zero)]
                                        all_done[g] = True
                                        break

                        if instr:
                            self.instrs.append(instr)
                        cycle += 1

                        if cycle > 200:
                            break

                    # Pass pre-computed addr calc to next round
                    prev_round_addr_done = next_round_addr_done

                    # Skip the separate hash section for scattered rounds
                    round_num += 1
                    continue

                elif level == 0:
                    # For round 0, some batches were XORed during loading (overlap optimization)
                    start_batch = round0_xor_done if round_num == 0 else 0
                    if start_batch < active:
                        emit_valu_chunked([
                            ("^", contexts[g]["val"], contexts[g]["val"], v_forest_nodes[0])
                            for g in range(start_batch, active)
                        ])
                elif level == 1:
                    # Interleaved scheduling for level 1: & -> vselect -> ^ -> hash
                    # Track when each stage completes
                    and_cycle = [-1] * active
                    vsel_cycle = [-1] * active
                    xor_cycle = [-1] * active
                    h0_cycle = [-1] * active
                    h1p1_cycle = [-1] * active
                    h1_cycle = [-1] * active
                    h2_cycle = [-1] * active
                    h3p1_cycle = [-1] * active
                    h3_cycle = [-1] * active
                    h4_cycle = [-1] * active
                    h5p1_cycle = [-1] * active
                    h5_cycle = [-1] * active
                    idx_and_cycle = [-1] * active
                    all_done = [False] * active

                    cycle = 0
                    and_idx = 0
                    while not all(all_done):
                        instr = {}
                        valu_ops = []

                        # Schedule vselect FIRST (1 per cycle) to keep pipeline flowing
                        for g in range(active):
                            if and_cycle[g] >= 0 and and_cycle[g] < cycle and vsel_cycle[g] < 0:
                                ctx = contexts[g]
                                instr["flow"] = [("vselect", ctx["tmp2"], ctx["tmp1"],
                                                  v_forest_nodes[1], v_forest_nodes[2])]
                                vsel_cycle[g] = cycle
                                break

                        # Fill VALU: prioritize XOR and hash from completed batches, then &
                        for g in range(active):
                            if len(valu_ops) >= 6:
                                break
                            ctx = contexts[g]

                            # XOR needs vselect done previous cycle
                            if vsel_cycle[g] >= 0 and vsel_cycle[g] < cycle and xor_cycle[g] < 0:
                                valu_ops.append(("^", ctx["val"], ctx["val"], ctx["tmp2"]))
                                xor_cycle[g] = cycle
                                continue

                            # Hash stage 0
                            if xor_cycle[g] >= 0 and xor_cycle[g] < cycle and h0_cycle[g] < 0:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult0, v_hash_const1[0]))
                                h0_cycle[g] = cycle
                                continue

                            # Hash stage 1 part 1
                            if h0_cycle[g] >= 0 and h0_cycle[g] < cycle and h1p1_cycle[g] < 0:
                                if len(valu_ops) <= 4:
                                    valu_ops.append((HASH_STAGES[1][0], ctx["tmp1"], ctx["val"], v_hash_const1[1]))
                                    valu_ops.append((HASH_STAGES[1][3], ctx["tmp2"], ctx["val"], v_hash_const2[1]))
                                    h1p1_cycle[g] = cycle
                                continue

                            # Hash stage 1 final
                            if h1p1_cycle[g] >= 0 and h1p1_cycle[g] < cycle and h1_cycle[g] < 0:
                                valu_ops.append((HASH_STAGES[1][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                h1_cycle[g] = cycle
                                continue

                            # Hash stage 2 + idx update
                            if h1_cycle[g] >= 0 and h1_cycle[g] < cycle and h2_cycle[g] < 0:
                                if len(valu_ops) <= 4:
                                    valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult2, v_hash_const1[2]))
                                    valu_ops.append(("multiply_add", ctx["idx"], ctx["idx"], v_two, v_one))
                                    h2_cycle[g] = cycle
                                continue

                            # Hash stage 3 part 1
                            if h2_cycle[g] >= 0 and h2_cycle[g] < cycle and h3p1_cycle[g] < 0:
                                if len(valu_ops) <= 4:
                                    valu_ops.append((HASH_STAGES[3][0], ctx["tmp1"], ctx["val"], v_hash_const1[3]))
                                    valu_ops.append((HASH_STAGES[3][3], ctx["tmp2"], ctx["val"], v_hash_const2[3]))
                                    h3p1_cycle[g] = cycle
                                continue

                            # Hash stage 3 final
                            if h3p1_cycle[g] >= 0 and h3p1_cycle[g] < cycle and h3_cycle[g] < 0:
                                valu_ops.append((HASH_STAGES[3][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                h3_cycle[g] = cycle
                                continue

                            # Hash stage 4
                            if h3_cycle[g] >= 0 and h3_cycle[g] < cycle and h4_cycle[g] < 0:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult4, v_hash_const1[4]))
                                h4_cycle[g] = cycle
                                continue

                            # Hash stage 5 part 1
                            if h4_cycle[g] >= 0 and h4_cycle[g] < cycle and h5p1_cycle[g] < 0:
                                if len(valu_ops) <= 4:
                                    valu_ops.append((HASH_STAGES[5][0], ctx["tmp1"], ctx["val"], v_hash_const1[5]))
                                    valu_ops.append((HASH_STAGES[5][3], ctx["tmp2"], ctx["val"], v_hash_const2[5]))
                                    h5p1_cycle[g] = cycle
                                continue

                            # Hash stage 5 final
                            if h5p1_cycle[g] >= 0 and h5p1_cycle[g] < cycle and h5_cycle[g] < 0:
                                valu_ops.append((HASH_STAGES[5][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                h5_cycle[g] = cycle
                                continue

                            # Index update part 1 (&)
                            if h5_cycle[g] >= 0 and h5_cycle[g] < cycle and idx_and_cycle[g] < 0:
                                valu_ops.append(("&", ctx["tmp1"], ctx["val"], v_one))
                                idx_and_cycle[g] = cycle
                                continue

                            # Index update part 2 (+)
                            if idx_and_cycle[g] >= 0 and idx_and_cycle[g] < cycle and not all_done[g]:
                                valu_ops.append(("+", ctx["idx"], ctx["idx"], ctx["tmp1"]))
                                all_done[g] = True

                        # Fill remaining slots with & for new batches
                        while len(valu_ops) < 6 and and_idx < active:
                            ctx = contexts[and_idx]
                            valu_ops.append(("&", ctx["tmp1"], ctx["idx"], v_one))
                            and_cycle[and_idx] = cycle
                            and_idx += 1

                        if valu_ops:
                            instr["valu"] = valu_ops[:6]
                        if instr:
                            self.instrs.append(instr)
                        cycle += 1
                        if cycle > 300:
                            break

                    # Skip the separate hash section for level 1
                    round_num += 1
                    continue
                elif level == 2:
                    # Interleaved scheduling for level 2
                    # Stages: & -> vsel1,vsel2 -> < -> vsel3 -> ^ -> hash
                    and1_cycle = [-1] * active
                    vsel1_cycle = [-1] * active
                    vsel2_cycle = [-1] * active
                    lt_cycle = [-1] * active
                    vsel3_cycle = [-1] * active
                    xor_cycle = [-1] * active
                    h0_cycle = [-1] * active
                    h1p1_cycle = [-1] * active
                    h1_cycle = [-1] * active
                    h2_cycle = [-1] * active
                    h3p1_cycle = [-1] * active
                    h3_cycle = [-1] * active
                    h4_cycle = [-1] * active
                    h5p1_cycle = [-1] * active
                    h5_cycle = [-1] * active
                    idx_and_cycle = [-1] * active
                    all_done = [False] * active

                    cycle = 0
                    and1_idx = 0
                    while not all(all_done):
                        instr = {}
                        valu_ops = []

                        # Schedule vselects FIRST (priority: vsel1, vsel2, then vsel3)
                        for g in range(active):
                            if and1_cycle[g] >= 0 and and1_cycle[g] < cycle and vsel1_cycle[g] < 0:
                                ctx = contexts[g]
                                instr["flow"] = [("vselect", ctx["tmp2"], ctx["tmp1"],
                                                  v_forest_nodes[3], v_forest_nodes[4])]
                                vsel1_cycle[g] = cycle
                                break
                            if vsel1_cycle[g] >= 0 and vsel1_cycle[g] < cycle and vsel2_cycle[g] < 0:
                                ctx = contexts[g]
                                instr["flow"] = [("vselect", ctx["node"], ctx["tmp1"],
                                                  v_forest_nodes[5], v_forest_nodes[6])]
                                vsel2_cycle[g] = cycle
                                break
                            if lt_cycle[g] >= 0 and lt_cycle[g] < cycle and vsel3_cycle[g] < 0:
                                ctx = contexts[g]
                                instr["flow"] = [("vselect", ctx["tmp2"], ctx["tmp1"],
                                                  ctx["tmp2"], ctx["node"])]
                                vsel3_cycle[g] = cycle
                                break

                        # Fill VALU slots with <, ^, and hash stages
                        for g in range(active):
                            if len(valu_ops) >= 6:
                                break
                            ctx = contexts[g]

                            # < needs vsel2 done (so tmp1 is free to reuse)
                            if vsel2_cycle[g] >= 0 and vsel2_cycle[g] < cycle and lt_cycle[g] < 0:
                                valu_ops.append(("<", ctx["tmp1"], ctx["idx"], v_five))
                                lt_cycle[g] = cycle
                                continue

                            # XOR needs vsel3 done
                            if vsel3_cycle[g] >= 0 and vsel3_cycle[g] < cycle and xor_cycle[g] < 0:
                                valu_ops.append(("^", ctx["val"], ctx["val"], ctx["tmp2"]))
                                xor_cycle[g] = cycle
                                continue

                            # Hash stages (same as level 1)
                            if xor_cycle[g] >= 0 and xor_cycle[g] < cycle and h0_cycle[g] < 0:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult0, v_hash_const1[0]))
                                h0_cycle[g] = cycle
                                continue
                            if h0_cycle[g] >= 0 and h0_cycle[g] < cycle and h1p1_cycle[g] < 0:
                                if len(valu_ops) <= 4:
                                    valu_ops.append((HASH_STAGES[1][0], ctx["tmp1"], ctx["val"], v_hash_const1[1]))
                                    valu_ops.append((HASH_STAGES[1][3], ctx["tmp2"], ctx["val"], v_hash_const2[1]))
                                    h1p1_cycle[g] = cycle
                                continue
                            if h1p1_cycle[g] >= 0 and h1p1_cycle[g] < cycle and h1_cycle[g] < 0:
                                valu_ops.append((HASH_STAGES[1][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                h1_cycle[g] = cycle
                                continue
                            if h1_cycle[g] >= 0 and h1_cycle[g] < cycle and h2_cycle[g] < 0:
                                if len(valu_ops) <= 4:
                                    valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult2, v_hash_const1[2]))
                                    valu_ops.append(("multiply_add", ctx["idx"], ctx["idx"], v_two, v_one))
                                    h2_cycle[g] = cycle
                                continue
                            if h2_cycle[g] >= 0 and h2_cycle[g] < cycle and h3p1_cycle[g] < 0:
                                if len(valu_ops) <= 4:
                                    valu_ops.append((HASH_STAGES[3][0], ctx["tmp1"], ctx["val"], v_hash_const1[3]))
                                    valu_ops.append((HASH_STAGES[3][3], ctx["tmp2"], ctx["val"], v_hash_const2[3]))
                                    h3p1_cycle[g] = cycle
                                continue
                            if h3p1_cycle[g] >= 0 and h3p1_cycle[g] < cycle and h3_cycle[g] < 0:
                                valu_ops.append((HASH_STAGES[3][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                h3_cycle[g] = cycle
                                continue
                            if h3_cycle[g] >= 0 and h3_cycle[g] < cycle and h4_cycle[g] < 0:
                                valu_ops.append(("multiply_add", ctx["val"], ctx["val"], v_mult4, v_hash_const1[4]))
                                h4_cycle[g] = cycle
                                continue
                            if h4_cycle[g] >= 0 and h4_cycle[g] < cycle and h5p1_cycle[g] < 0:
                                if len(valu_ops) <= 4:
                                    valu_ops.append((HASH_STAGES[5][0], ctx["tmp1"], ctx["val"], v_hash_const1[5]))
                                    valu_ops.append((HASH_STAGES[5][3], ctx["tmp2"], ctx["val"], v_hash_const2[5]))
                                    h5p1_cycle[g] = cycle
                                continue
                            if h5p1_cycle[g] >= 0 and h5p1_cycle[g] < cycle and h5_cycle[g] < 0:
                                valu_ops.append((HASH_STAGES[5][2], ctx["val"], ctx["tmp1"], ctx["tmp2"]))
                                h5_cycle[g] = cycle
                                continue
                            if h5_cycle[g] >= 0 and h5_cycle[g] < cycle and idx_and_cycle[g] < 0:
                                valu_ops.append(("&", ctx["tmp1"], ctx["val"], v_one))
                                idx_and_cycle[g] = cycle
                                continue
                            if idx_and_cycle[g] >= 0 and idx_and_cycle[g] < cycle and not all_done[g]:
                                valu_ops.append(("+", ctx["idx"], ctx["idx"], ctx["tmp1"]))
                                all_done[g] = True

                        # Fill remaining slots with first & for new batches
                        while len(valu_ops) < 6 and and1_idx < active:
                            ctx = contexts[and1_idx]
                            valu_ops.append(("&", ctx["tmp1"], ctx["idx"], v_one))
                            and1_cycle[and1_idx] = cycle
                            and1_idx += 1

                        if valu_ops:
                            instr["valu"] = valu_ops[:6]
                        if instr:
                            self.instrs.append(instr)
                        cycle += 1
                        if cycle > 400:
                            break

                    # Skip the separate hash section for level 2
                    round_num += 1
                    continue
                else:
                    emit_valu_chunked([
                        ("^", contexts[g]["val"], contexts[g]["val"], contexts[g]["node"])
                        for g in range(active)
                    ])

                # Hash computation - pack parallel ops together
                # Stage 0
                emit_valu_chunked([
                    ("multiply_add", contexts[g]["val"], contexts[g]["val"], v_mult0, v_hash_const1[0])
                    for g in range(active)
                ])

                # Stage 1: pack ^ and >> together
                ops1 = []
                for g in range(active):
                    ops1.append((HASH_STAGES[1][0], contexts[g]["tmp1"], contexts[g]["val"], v_hash_const1[1]))
                    ops1.append((HASH_STAGES[1][3], contexts[g]["tmp2"], contexts[g]["val"], v_hash_const2[1]))
                # Split into 6-op chunks
                emit_valu_chunked(ops1)
                emit_valu_chunked([
                    (HASH_STAGES[1][2], contexts[g]["val"], contexts[g]["tmp1"], contexts[g]["tmp2"])
                    for g in range(active)
                ])

                # Stage 2 + index update - pack together
                ops2 = []
                for g in range(active):
                    ops2.append(("multiply_add", contexts[g]["val"], contexts[g]["val"], v_mult2, v_hash_const1[2]))
                    ops2.append(("multiply_add", contexts[g]["idx"], contexts[g]["idx"], v_two, v_one))
                emit_valu_chunked(ops2)

                # Stage 3: pack + and << together
                ops3 = []
                for g in range(active):
                    ops3.append((HASH_STAGES[3][0], contexts[g]["tmp1"], contexts[g]["val"], v_hash_const1[3]))
                    ops3.append((HASH_STAGES[3][3], contexts[g]["tmp2"], contexts[g]["val"], v_hash_const2[3]))
                emit_valu_chunked(ops3)
                emit_valu_chunked([
                    (HASH_STAGES[3][2], contexts[g]["val"], contexts[g]["tmp1"], contexts[g]["tmp2"])
                    for g in range(active)
                ])

                # Stage 4
                emit_valu_chunked([
                    ("multiply_add", contexts[g]["val"], contexts[g]["val"], v_mult4, v_hash_const1[4])
                    for g in range(active)
                ])

                # Stage 5: pack ^ and >> together
                ops5 = []
                for g in range(active):
                    ops5.append((HASH_STAGES[5][0], contexts[g]["tmp1"], contexts[g]["val"], v_hash_const1[5]))
                    ops5.append((HASH_STAGES[5][3], contexts[g]["tmp2"], contexts[g]["val"], v_hash_const2[5]))
                emit_valu_chunked(ops5)
                emit_valu_chunked([
                    (HASH_STAGES[5][2], contexts[g]["val"], contexts[g]["tmp1"], contexts[g]["tmp2"])
                    for g in range(active)
                ])

                # Index update
                emit_valu_chunked([
                    ("&", contexts[g]["tmp1"], contexts[g]["val"], v_one)
                    for g in range(active)
                ])
                emit_valu_chunked([
                    ("+", contexts[g]["idx"], contexts[g]["idx"], contexts[g]["tmp1"])
                    for g in range(active)
                ])

                if need_overflow_check:
                    emit_valu_chunked([
                        ("<", contexts[g]["tmp1"], contexts[g]["idx"], v_n_nodes)
                        for g in range(active)
                    ])
                    for g in range(active):
                        self.instrs.append({"flow": [
                            ("vselect", contexts[g]["idx"], contexts[g]["tmp1"],
                             contexts[g]["idx"], v_zero)
                        ]})

                round_num += 1

            # Store batches
            for g in range(active):
                batch = group_start + g
                ctx = contexts[g]
                self.instrs.append({"store": [
                    ("vstore", idx_addrs[batch], ctx["idx"]),
                    ("vstore", val_addrs[batch], ctx["val"]),
                ]})

        self.instrs.append({"flow": [("pause",)]})

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


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
