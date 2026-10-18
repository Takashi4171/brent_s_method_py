"""Tests for brent_s_method_py.core.brent.

Only behaviours the implementation actually claims are tested. The return type
is BrentResult in every case (converged root, bad bracket, max iterations),
so every assertion goes through that type.
"""

from __future__ import annotations

import math
import unittest

from brent_s_method_py import BrentResult, brent


class TestBrentHappyPath(unittest.TestCase):
    def test_simple_root_polynomial(self):
        r = brent(lambda x: x * x - 4.0, 0.0, 10.0)
        self.assertTrue(r.converged)
        self.assertLess(r.f_root, 1e-9)
        self.assertLess(abs(r.root - 2.0), 1e-6)

    def test_root_at_origin(self):
        r = brent(lambda x: x, -1.0, 1.0)
        self.assertTrue(r.converged)
        self.assertLess(abs(r.root), 1e-6)

    def test_root_in_left_half(self):
        r = brent(lambda x: x + 3.0, -10.0, 10.0)
        self.assertTrue(r.converged)
        self.assertLess(abs(r.root - (-3.0)), 1e-6)

    def test_endpoint_root_a(self):
        # f(a) == 0 exactly; implementation short-circuits.
        r = brent(lambda x: x - 5.0, 5.0, 7.0)
        self.assertTrue(r.converged)
        self.assertEqual(r.iterations, 0)
        self.assertEqual(r.reason, "converged")

    def test_endpoint_root_b(self):
        # f(b) == 0 exactly; implementation short-circuits.
        r = brent(lambda x: x - 7.0, 5.0, 7.0)
        self.assertTrue(r.converged)
        self.assertEqual(r.iterations, 0)
        self.assertEqual(r.reason, "converged")


class TestBrentBracketInvariants(unittest.TestCase):
    def test_returned_bracket_has_opposite_signs(self):
        f = lambda x: math.sin(x)
        r = brent(f, 3.0, 4.0)
        a, b = r.bracket
        self.assertLessEqual(f(a) * f(b), 0.0)

    def test_bracket_contains_root(self):
        f = lambda x: x * x - 2.0
        r = brent(f, 0.0, 5.0)
        a, b = r.bracket
        lo, hi = (a, b) if a <= b else (b, a)
        self.assertLessEqual(lo, r.root)
        self.assertLessEqual(r.root, hi)


class TestBrentBadInputs(unittest.TestCase):
    def test_bad_bracket_same_sign(self):
        r = brent(lambda x: x * x + 1.0, -1.0, 1.0)
        self.assertFalse(r.converged)
        self.assertEqual(r.reason, "bad_bracket")
        self.assertEqual(r.iterations, 0)

    def test_nan_endpoint_function_value(self):
        # f returns NaN at one endpoint; the bracket check rejects it.
        def f(x: float) -> float:
            if x == -1.0:
                return float("nan")
            return x * x
        r = brent(f, -1.0, 2.0)
        self.assertFalse(r.converged)
        self.assertEqual(r.reason, "bad_bracket")


class TestBrentMaxIter(unittest.TestCase):
    def test_zero_iterations_returns_max_iterations(self):
        # max_iter=0 means the loop body never runs; we should report
        # max_iterations (not converged) because no convergence test ran.
        r = brent(lambda x: x - 0.5, 0.0, 1.0, max_iter=0)
        self.assertFalse(r.converged)
        self.assertEqual(r.reason, "max_iterations")
        self.assertEqual(r.iterations, 0)


class TestBrentTrickyFunctions(unittest.TestCase):
    def test_smooth_steep_near_root(self):
        # Cubic with a steep slope at the root. Tests tolerance vs slope.
        f = lambda x: (x - 0.3) ** 3 + (x - 0.3)
        r = brent(f, -1.0, 1.0)
        self.assertTrue(r.converged)
        self.assertLess(abs(r.root - 0.3), 1e-6)

    def test_asymmetric_bracket(self):
        # Root near one endpoint; bracket width asymmetric.
        f = lambda x: x - 0.99
        r = brent(f, 0.0, 1.0)
        self.assertTrue(r.converged)
        self.assertLess(abs(r.root - 0.99), 1e-6)


class TestBrentResultType(unittest.TestCase):
    def test_result_is_brent_result(self):
        r = brent(lambda x: x, -1.0, 1.0)
        self.assertIsInstance(r, BrentResult)

    def test_bad_bracket_returns_brent_result(self):
        r = brent(lambda x: x * x + 1.0, -1.0, 1.0)
        self.assertIsInstance(r, BrentResult)


if __name__ == "__main__":
    unittest.main()
