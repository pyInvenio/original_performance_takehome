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

        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        five_const = self.scratch_const(5)

        self.instrs.append({"valu": [
            ("vbroadcast", v_two, two_const),
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_zero, zero_const),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
            ("vbroadcast", v_five, five_const),
        ]})

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        # ===== HASH CONSTANTS =====
        v_hash_const1 = [self.alloc_vec(f"v_hash{i}_const1") for i in range(6)]
        v_hash_const2 = [self.alloc_vec(f"v_hash{i}_const2") for i in range(6)]

        self.instrs.append({"valu": [
            ("vbroadcast", v_hash_const1[i], self.scratch_const(HASH_STAGES[i][1]))
            for i in range(6)
        ]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_hash_const2[i], self.scratch_const(HASH_STAGES[i][4]))
            for i in range(6)
        ]})

        v_mult0 = self.alloc_vec("v_mult0")
        v_mult2 = self.alloc_vec("v_mult2")
        v_mult4 = self.alloc_vec("v_mult4")
        self.instrs.append({"valu": [
            ("vbroadcast", v_mult0, self.scratch_const(1 + (1 << 12))),
            ("vbroadcast", v_mult2, self.scratch_const(1 + (1 << 5))),
            ("vbroadcast", v_mult4, self.scratch_const(1 + (1 << 3))),
        ]})

        # ===== CACHE TREE NODES 0-6 =====
        v_forest_nodes = [self.alloc_vec(f"v_forest_node_{i}") for i in range(7)]
        forest_cache = self.alloc_scratch("forest_cache", 8)
        self.instrs.append({"load": [("vload", forest_cache, self.scratch["forest_values_p"])]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_forest_nodes[i], forest_cache + i) for i in range(6)
        ]})
        self.instrs.append({"valu": [("vbroadcast", v_forest_nodes[6], forest_cache + 6)]})

        # ===== PRECOMPUTE BATCH ADDRESSES =====
        idx_addrs = [self.alloc_scratch(f"idx_addr_{i}") for i in range(n_batches)]
        val_addrs = [self.alloc_scratch(f"val_addr_{i}") for i in range(n_batches)]

        for i in range(0, n_batches, 6):
            alu_ops = []
            for j in range(min(6, n_batches - i)):
                offset = (i + j) * VLEN
                off_const = self.scratch_const(offset)
                alu_ops.append(("+", idx_addrs[i+j], self.scratch["inp_indices_p"], off_const))
                alu_ops.append(("+", val_addrs[i+j], self.scratch["inp_values_p"], off_const))
            self.instrs.append({"alu": alu_ops})

        # ===== MAIN KERNEL =====
        for group_start in range(0, n_batches, GROUP_SIZE):
            group_end = min(group_start + GROUP_SIZE, n_batches)
            active = group_end - group_start

            # Load batches
            for g in range(active):
                batch = group_start + g
                ctx = contexts[g]
                self.instrs.append({"load": [
                    ("vload", ctx["idx"], idx_addrs[batch]),
                    ("vload", ctx["val"], val_addrs[batch]),
                ]})

            # Process rounds with cross-round pipelining for consecutive scattered rounds
            round_num = 0
            # Track addr calc done from previous round (for pipelining)
            prev_round_addr_done = None
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

                    # Use pre-computed addr calc from previous round if available
                    addr_done = [-1] * active
                    addr_calc_queue = list(range(active))  # Batches needing addr calc
                    if prev_round_addr_done is not None:
                        # Mark pre-computed batches as done and queue their loads
                        for g in range(active):
                            if prev_round_addr_done[g] >= 0:
                                addr_done[g] = -1  # "done before this round"
                                for pair in range(4):
                                    load_queue.append((g, pair))
                                addr_calc_queue.remove(g)  # Don't need addr calc
                        prev_round_addr_done = None

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
                        # Process batches in order, greedily fill 6 VALU slots
                        for g in range(active):
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
                    emit_valu_chunked([
                        ("^", contexts[g]["val"], contexts[g]["val"], v_forest_nodes[0])
                        for g in range(active)
                    ])
                elif level == 1:
                    for g in range(active):
                        ctx = contexts[g]
                        self.instrs.append({"valu": [("&", ctx["tmp1"], ctx["idx"], v_one)]})
                        self.instrs.append({"flow": [("vselect", ctx["tmp2"], ctx["tmp1"],
                                                      v_forest_nodes[1], v_forest_nodes[2])]})
                        self.instrs.append({"valu": [("^", ctx["val"], ctx["val"], ctx["tmp2"])]})
                elif level == 2:
                    for g in range(active):
                        ctx = contexts[g]
                        self.instrs.append({"valu": [("&", ctx["tmp1"], ctx["idx"], v_one)]})
                        self.instrs.append({"flow": [("vselect", ctx["tmp2"], ctx["tmp1"],
                                                      v_forest_nodes[3], v_forest_nodes[4])]})
                        self.instrs.append({"flow": [("vselect", ctx["node"], ctx["tmp1"],
                                                      v_forest_nodes[5], v_forest_nodes[6])]})
                        self.instrs.append({"valu": [("<", ctx["tmp1"], ctx["idx"], v_five)]})
                        self.instrs.append({"flow": [("vselect", ctx["tmp2"], ctx["tmp1"],
                                                      ctx["tmp2"], ctx["node"])]})
                        self.instrs.append({"valu": [("^", ctx["val"], ctx["val"], ctx["tmp2"])]})
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
