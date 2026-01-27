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

    # def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
    #     # Simple slot packing that just uses one slot per instruction bundle
    #     instrs = []
    #     for engine, slot in slots:
    #         instrs.append({engine: [slot]})
    #     return instrs

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

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        # for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):

        #     # need to broadcast const
        #     slots.append({"valu": [
        #               ("vbroadcast", v_const1, self.scratch_const(val1)),
        #               ("vbroadcast", v_const2, self.scratch_const(val3)),
        #           ]})
        #     slots.append({"valu": [
        #       (op1, tmp1, val_hash_addr, v_const1),
        #       (op3, tmp2, val_hash_addr, v_const2)
        #     ]})
        #     slots.append({"valu": [(op2, val_hash_addr, tmp1, tmp2)]})
        #     slots.append({"debug": [("compare", val_hash_addr, (round, i, "hash_stage", hi))]})

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """

        v_idx_a = self.alloc_scratch("v_idx_a", VLEN)
        v_val_a = self.alloc_scratch("v_val_a", VLEN)
        v_node_val_a = self.alloc_scratch("v_node_val_a", VLEN)
        v_idx_b = self.alloc_scratch("v_idx_b", VLEN)
        v_val_b = self.alloc_scratch("v_val_b", VLEN)
        v_node_val_b = self.alloc_scratch("v_node_val_b", VLEN)

        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)

        idx_base = self.alloc_scratch("idx_base")
        val_base = self.alloc_scratch("val_base")

        v_two = self.alloc_scratch("v_two", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_zero = self.alloc_scratch("v_zero", VLEN)

        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        self.instrs.append({"valu": [("vbroadcast", v_two, two_const),
                                     ("vbroadcast", v_one, one_const),
                                     ("vbroadcast", v_zero, zero_const),
                                     ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])]})

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr_1 = self.alloc_scratch("tmp_addr_1")
        tmp_addr_2 = self.alloc_scratch("tmp_addr_2")
        tmp_addr_3 = self.alloc_scratch("tmp_addr_3")
        tmp_addrs_a = [self.alloc_scratch(f"tmp_addr_a_{i}") for i in range(VLEN)]
        tmp_addrs_b = [self.alloc_scratch(f"tmp_addr_b_{i}") for i in range(VLEN)]

        buffers = [
            (v_idx_a, v_val_a, v_node_val_a, tmp_addrs_a),
            (v_idx_b, v_val_b, v_node_val_b, tmp_addrs_b),
        ]

        v_hash0_const1 = self.alloc_scratch(f"v_hash0_const1", VLEN)
        v_hash0_const2 = self.alloc_scratch(f"v_hash0_const2", VLEN)
        v_hash1_const1 = self.alloc_scratch(f"v_hash1_const1", VLEN)
        v_hash1_const2 = self.alloc_scratch(f"v_hash1_const2", VLEN)
        v_hash2_const1 = self.alloc_scratch(f"v_hash2_const1", VLEN)
        v_hash2_const2 = self.alloc_scratch(f"v_hash2_const2", VLEN)
        v_hash3_const1 = self.alloc_scratch(f"v_hash3_const1", VLEN)
        v_hash3_const2 = self.alloc_scratch(f"v_hash3_const2", VLEN)
        v_hash4_const1 = self.alloc_scratch(f"v_hash4_const1", VLEN)
        v_hash4_const2 = self.alloc_scratch(f"v_hash4_const2", VLEN)
        v_hash5_const1 = self.alloc_scratch(f"v_hash5_const1", VLEN)
        v_hash5_const2 = self.alloc_scratch(f"v_hash5_const2", VLEN)

        self.instrs.append({"valu": [
                ("vbroadcast", v_hash0_const1, self.scratch_const(HASH_STAGES[0][1])),
                ("vbroadcast", v_hash0_const2, self.scratch_const(HASH_STAGES[0][4])),
                ("vbroadcast", v_hash1_const1, self.scratch_const(HASH_STAGES[1][1])),
                ("vbroadcast", v_hash1_const2, self.scratch_const(HASH_STAGES[1][4])),
                ("vbroadcast", v_hash2_const1, self.scratch_const(HASH_STAGES[2][1])),
                ("vbroadcast", v_hash2_const2, self.scratch_const(HASH_STAGES[2][4])),
        ]})
        self.instrs.append({"valu": [
                ("vbroadcast", v_hash3_const1, self.scratch_const(HASH_STAGES[3][1])),
                ("vbroadcast", v_hash3_const2, self.scratch_const(HASH_STAGES[3][4])),
                ("vbroadcast", v_hash4_const1, self.scratch_const(HASH_STAGES[4][1])),
                ("vbroadcast", v_hash4_const2, self.scratch_const(HASH_STAGES[4][4])),
                ("vbroadcast", v_hash5_const1, self.scratch_const(HASH_STAGES[5][1])),
                ("vbroadcast", v_hash5_const2, self.scratch_const(HASH_STAGES[5][4])),
        ]})


        def do_address_calc(offset_const, tmp_addr_1, tmp_addr_2):
            self.instrs.append({"alu": [
                ("+", tmp_addr_1, self.scratch["inp_indices_p"], offset_const),
                ("+", tmp_addr_2, self.scratch["inp_values_p"], offset_const),
            ]})
        def do_vloads(v_idx, v_val, tmp_addr_1, tmp_addr_2):
            self.instrs.append({"load": [
                ("vload", v_idx, tmp_addr_1),
                ("vload", v_val, tmp_addr_2),
            ]})
        def do_scattered_address_calc(v_idx, tmp_addrs):
            self.instrs.append({"alu": [
                ("+", tmp_addrs[0], self.scratch["forest_values_p"], v_idx + 0),
                ("+", tmp_addrs[1], self.scratch["forest_values_p"], v_idx + 1),
                ("+", tmp_addrs[2], self.scratch["forest_values_p"], v_idx + 2),
                ("+", tmp_addrs[3], self.scratch["forest_values_p"], v_idx + 3),
                ("+", tmp_addrs[4], self.scratch["forest_values_p"], v_idx + 4),
                ("+", tmp_addrs[5], self.scratch["forest_values_p"], v_idx + 5),
                ("+", tmp_addrs[6], self.scratch["forest_values_p"], v_idx + 6),
                ("+", tmp_addrs[7], self.scratch["forest_values_p"], v_idx + 7),
            ]})

        def do_scattered_loads(v_node_val, tmp_addrs):
            self.instrs.append({"load": [
                ("load", v_node_val + 0, tmp_addrs[0]),
                ("load", v_node_val + 1, tmp_addrs[1]),
            ]})
            self.instrs.append({"load": [
                ("load", v_node_val + 2, tmp_addrs[2]),
                ("load", v_node_val + 3, tmp_addrs[3]),
            ]})
            self.instrs.append({"load": [
                ("load", v_node_val + 4, tmp_addrs[4]),
                ("load", v_node_val + 5, tmp_addrs[5]),
            ]})
            self.instrs.append({"load": [
                ("load", v_node_val + 6, tmp_addrs[6]),
                ("load", v_node_val + 7, tmp_addrs[7]),
            ]})

        for round in range(rounds):
            offset_const_0 = self.scratch_const(0)
            v_idx_cur, v_val_cur, v_node_val_cur, tmp_addrs_cur = buffers[0]
            # first batch
            do_address_calc(offset_const_0, tmp_addr_1, tmp_addr_2)
            do_vloads(v_idx_cur, v_val_cur, tmp_addr_1, tmp_addr_2)
            do_scattered_address_calc(v_idx_cur, tmp_addrs_cur)
            do_scattered_loads(v_node_val_cur, tmp_addrs_cur)

            for offset in range(0, batch_size - VLEN, VLEN):
                offset_const = self.scratch_const(offset)
                next_offset_const = self.scratch_const(offset + VLEN)

                cur_buf = (offset // VLEN) % 2
                next_buf = 1 - cur_buf

                v_idx_cur, v_val_cur, v_node_val_cur, tmp_addrs_cur = buffers[cur_buf]
                v_idx_next, v_val_next, v_node_val_next, tmp_addrs_next = buffers[next_buf]

                self.instrs.append({"valu": [("^", v_val_cur, v_val_cur, v_node_val_cur)]})

                self.instrs.append({"valu": [
                                        (HASH_STAGES[0][0], v_tmp1, v_val_cur, v_hash0_const1),
                                        (HASH_STAGES[0][3], v_tmp2, v_val_cur, v_hash0_const2)
                                    ],
                                    "alu": [("+", tmp_addr_1, self.scratch["inp_indices_p"], next_offset_const),
                                            ("+", tmp_addr_2, self.scratch["inp_values_p"], next_offset_const)]})

                self.instrs.append({"valu":
                                    [(HASH_STAGES[0][2], v_val_cur, v_tmp1, v_tmp2)],
                                    "load": [
                                    ("vload", v_idx_next, tmp_addr_1),
                                    ("vload", v_val_next, tmp_addr_2)]})

                self.instrs.append({"valu": [
                    (HASH_STAGES[1][0], v_tmp1, v_val_cur, v_hash1_const1),
                    (HASH_STAGES[1][3], v_tmp2, v_val_cur, v_hash1_const2)
                    ],
                    "alu": [
                    ("+", tmp_addrs_next[0], self.scratch["forest_values_p"], v_idx_next + 0),
                    ("+", tmp_addrs_next[1], self.scratch["forest_values_p"], v_idx_next + 1),
                    ("+", tmp_addrs_next[2], self.scratch["forest_values_p"], v_idx_next + 2),
                    ("+", tmp_addrs_next[3], self.scratch["forest_values_p"], v_idx_next + 3),
                    ("+", tmp_addrs_next[4], self.scratch["forest_values_p"], v_idx_next + 4),
                    ("+", tmp_addrs_next[5], self.scratch["forest_values_p"], v_idx_next + 5),
                    ("+", tmp_addrs_next[6], self.scratch["forest_values_p"], v_idx_next + 6),
                    ("+", tmp_addrs_next[7], self.scratch["forest_values_p"], v_idx_next + 7),
                ]})

                self.instrs.append({
                    "valu": [(HASH_STAGES[1][2], v_val_cur, v_tmp1, v_tmp2),
                             ("*", v_idx_cur, v_idx_cur, v_two)],
                    "load": [
                    ("load", v_node_val_next + 0, tmp_addrs_next[0]),
                    ("load", v_node_val_next + 1, tmp_addrs_next[1]),
                ]})

                self.instrs.append({
                    "valu": [
                    (HASH_STAGES[2][0], v_tmp1, v_val_cur, v_hash2_const1),
                    (HASH_STAGES[2][3], v_tmp2, v_val_cur, v_hash2_const2),
                    ("+", v_idx_cur, v_idx_cur, v_one)
                    ],
                    "load": [
                    ("load", v_node_val_next + 2, tmp_addrs_next[2]),
                    ("load", v_node_val_next + 3, tmp_addrs_next[3]),
                ]})

                self.instrs.append({"valu":
                                    [(HASH_STAGES[2][2], v_val_cur, v_tmp1, v_tmp2)],
                                    "load": [
                                    ("load", v_node_val_next + 4, tmp_addrs_next[4]),
                                    ("load", v_node_val_next + 5, tmp_addrs_next[5]),
                ]})

                self.instrs.append({"valu": [
                    (HASH_STAGES[3][0], v_tmp1, v_val_cur, v_hash3_const1),
                    (HASH_STAGES[3][3], v_tmp2, v_val_cur, v_hash3_const2)
                    ],
                    "load": [
                    ("load", v_node_val_next + 6, tmp_addrs_next[6]),
                    ("load", v_node_val_next + 7, tmp_addrs_next[7]),
                ]})

                # Remaining hash stages, no loads for overlap
                self.instrs.append({"valu": [(HASH_STAGES[3][2], v_val_cur, v_tmp1, v_tmp2)]})
                self.instrs.append({"valu": [
                    (HASH_STAGES[4][0], v_tmp1, v_val_cur, v_hash4_const1),
                    (HASH_STAGES[4][3], v_tmp2, v_val_cur, v_hash4_const2)
                    ]})
                self.instrs.append({"valu": [(HASH_STAGES[4][2], v_val_cur, v_tmp1, v_tmp2)]})
                self.instrs.append({"valu": [
                    (HASH_STAGES[5][0], v_tmp1, v_val_cur, v_hash5_const1),
                    (HASH_STAGES[5][3], v_tmp2, v_val_cur, v_hash5_const2)
                    ]})
                self.instrs.append({"valu": [(HASH_STAGES[5][2], v_val_cur, v_tmp1, v_tmp2)]})

                # Index update
                self.instrs.append({"valu": [("&", v_tmp1, v_val_cur, v_one)]})
                self.instrs.append({"valu": [("+", v_idx_cur, v_idx_cur, v_tmp1)]})
                self.instrs.append({"valu": [("<", v_tmp1, v_idx_cur, v_n_nodes)]})
                self.instrs.append({"flow": [("vselect", v_idx_cur, v_tmp1, v_idx_cur, v_zero)]})

                # Store current batch
                self.instrs.append({"alu": [("+", tmp_addr_1, self.scratch["inp_indices_p"], offset_const),
                                            ("+", tmp_addr_2, self.scratch["inp_values_p"], offset_const)]})
                self.instrs.append({"store": [("vstore", tmp_addr_1, v_idx_cur),
                                              ("vstore", tmp_addr_2, v_val_cur)]})

            # last batch
            last_offset = batch_size - VLEN
            last_offset_const = self.scratch_const(last_offset)
            last_buf = ((batch_size // VLEN) - 1) % 2
            v_idx_cur, v_val_cur, v_node_val_cur, tmp_addrs_cur = buffers[last_buf]

            self.instrs.append({"valu": [("^", v_val_cur, v_val_cur, v_node_val_cur)]})

            # Hash stages
            self.instrs.append({"valu": [
                (HASH_STAGES[0][0], v_tmp1, v_val_cur, v_hash0_const1),
                (HASH_STAGES[0][3], v_tmp2, v_val_cur, v_hash0_const2)]})
            self.instrs.append({"valu": [(HASH_STAGES[0][2], v_val_cur, v_tmp1, v_tmp2)]})
            self.instrs.append({"valu": [
                (HASH_STAGES[1][0], v_tmp1, v_val_cur, v_hash1_const1),
                (HASH_STAGES[1][3], v_tmp2, v_val_cur, v_hash1_const2)]})
            self.instrs.append({"valu": [(HASH_STAGES[1][2], v_val_cur, v_tmp1, v_tmp2)]})
            self.instrs.append({"valu": [
                (HASH_STAGES[2][0], v_tmp1, v_val_cur, v_hash2_const1),
                (HASH_STAGES[2][3], v_tmp2, v_val_cur, v_hash2_const2)]})
            self.instrs.append({"valu": [(HASH_STAGES[2][2], v_val_cur, v_tmp1, v_tmp2)]})
            self.instrs.append({"valu": [
                (HASH_STAGES[3][0], v_tmp1, v_val_cur, v_hash3_const1),
                (HASH_STAGES[3][3], v_tmp2, v_val_cur, v_hash3_const2)]})
            self.instrs.append({"valu": [(HASH_STAGES[3][2], v_val_cur, v_tmp1, v_tmp2)]})
            self.instrs.append({"valu": [
                (HASH_STAGES[4][0], v_tmp1, v_val_cur, v_hash4_const1),
                (HASH_STAGES[4][3], v_tmp2, v_val_cur, v_hash4_const2)]})
            self.instrs.append({"valu": [(HASH_STAGES[4][2], v_val_cur, v_tmp1, v_tmp2)]})
            self.instrs.append({"valu": [
                (HASH_STAGES[5][0], v_tmp1, v_val_cur, v_hash5_const1),
                (HASH_STAGES[5][3], v_tmp2, v_val_cur, v_hash5_const2)]})
            self.instrs.append({"valu": [(HASH_STAGES[5][2], v_val_cur, v_tmp1, v_tmp2)]})

            # Index update
            self.instrs.append({"valu": [("*", v_idx_cur, v_idx_cur, v_two)]})
            self.instrs.append({"valu": [("+", v_idx_cur, v_idx_cur, v_one)]})
            self.instrs.append({"valu": [("&", v_tmp1, v_val_cur, v_one)]})
            self.instrs.append({"valu": [("+", v_idx_cur, v_idx_cur, v_tmp1)]})
            self.instrs.append({"valu": [("<", v_tmp1, v_idx_cur, v_n_nodes)]})
            self.instrs.append({"flow": [("vselect", v_idx_cur, v_tmp1, v_idx_cur, v_zero)]})

            self.instrs.append({"alu": [("+", tmp_addr_1, self.scratch["inp_indices_p"], last_offset_const),
                                        ("+", tmp_addr_2, self.scratch["inp_values_p"], last_offset_const)]})
            self.instrs.append({"store": [("vstore", tmp_addr_1, v_idx_cur),
                                          ("vstore", tmp_addr_2, v_val_cur)]})

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
    # print(kb.instrs)

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
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
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
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
