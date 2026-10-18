"""Brent's method for root finding on a bracketing interval.

Design choices worth stating plainly:

- No exceptions, Result object only. The return type is BrentResult in every
  case. Callers branch on BrentResult.converged. This keeps the happy path
  (a root) and the two failure modes (bad bracket, no convergence in budget)
  uniform and easy to inspect.

- Tolerance is absolute. Two bounds, |b - a| and 2*|f(b)|, must both fall
  below xtol for convergence. Relative tolerance across wildly different
  magnitudes is the caller's problem; the brief said "a root of a scalar
  function" and absolute is the honest reading.

- Two bracket checks: at entry, f(a) and f(b) must be finite and have opposite
  sign (or one must be zero). Inside the loop, if a newly chosen candidate
  lands outside the current bracket or is not finite, we fall back to a plain
  bisection step. Brent's method is only safe because the bracket is an
  invariant; every fast step here is guarded by that invariant.

- Default tolerances are 1e-12. That is below float64 epsilon (~2.2e-16) at
  unit scale, so the loop terminates on the bracket-width bound rather than
  the function-value bound when f'(root) is order 1. For functions with large
  slope or with roots very near zero, pass xtol explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class BrentResult:
    """Outcome of a Brent's method run.

    Attributes:
        root: best estimate of the root. Equal to the b at termination.
        f_root: f(root). Should be small when converged is True.
        iterations: number of loop iterations actually performed.
        converged: True iff the loop met its tolerance before max_iter.
        bracket: (a, b) at termination. Always a valid bracket: f values at
            the two endpoints have opposite sign (unless one is exactly zero).
        reason: short human description of why the loop stopped. One of:
            "converged", "bad_bracket", or "max_iterations".
    """

    root: float
    f_root: float
    iterations: int
    converged: bool
    bracket: tuple
    reason: str


def _sign(x: float) -> int:
    """Return -1, 0, or +1 without relying on math.copysign.

    copysign treats +0.0 and -0.0 differently, which is noise here. We only
    care which side of zero f lands on for the bracket invariant.
    """
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    xtol: float = 1e-12,
    max_iter: int = 100,
) -> BrentResult:
    """Find a root of f in [a, b] using Brent's method.

    f must be continuous on [a, b] and f(a), f(b) must bracket a root (opposite
    signs, or one zero). No derivatives are taken.

    Returns:
        BrentResult. Check .converged before trusting .root.
    """
    fa = f(a)
    fb = f(b)

    # --- Entry validation -------------------------------------------------
    # A bracket is required. We also refuse NaNs because the sign test below
    # would happily accept a NaN bracket and then loop forever.
    if not (math.isfinite(a) and math.isfinite(b)):
        return BrentResult(
            root=b, f_root=fb, iterations=0, converged=False,
            bracket=(a, b), reason="bad_bracket",
        )
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return BrentResult(
            root=b, f_root=fb, iterations=0, converged=False,
            bracket=(a, b), reason="bad_bracket",
        )

    sa = _sign(fa)
    sb = _sign(fb)

    # Exact endpoint root: short-circuit before the loop.
    if sa == 0:
        return BrentResult(
            root=a, f_root=fa, iterations=0, converged=True,
            bracket=(a, b), reason="converged",
        )
    if sb == 0:
        return BrentResult(
            root=b, f_root=fb, iterations=0, converged=True,
            bracket=(a, b), reason="converged",
        )
    if sa == sb:
        # Same strict sign on both ends is not a bracket.
        return BrentResult(
            root=b, f_root=fb, iterations=0, converged=False,
            bracket=(a, b), reason="bad_bracket",
        )

    # --- Brent's loop -----------------------------------------------------
    # Naming follows the standard presentation: a/b are the current bracket,
    # c is the previous iterate (the "contrapoint" with opposite sign), and
    # b is always the best current estimate. fa/fc have opposite sign.
    c = a
    fc = fa
    mflag = True  # True while bisection is the fall-back choice
    iter_count = 0

    while iter_count < max_iter:
        iter_count += 1

        # Convergence test. |b - a| bounds the bracket size; 2*|fb| covers
        # the function-value side. Both must hold for a clean stop.
        if abs(b - a) < xtol and 2.0 * abs(fb) < xtol:
            return BrentResult(
                root=b, f_root=fb, iterations=iter_count, converged=True,
                bracket=(a, b), reason="converged",
            )

        # --- Choose the next iterate -----------------------------------
        # Three interpolation strategies, best to worst, with bisection as
        # the unconditional fall-back whenever interpolation misbehaves.
        use_bisection = False
        s_candidate: Optional[float] = None

        same_fa_fc = (fa == fc)  # exact equality is correct here
        same_fb_fc = (fb == fc)

        if not same_fa_fc and not same_fb_fc:
            # Inverse quadratic interpolation through (a,f(a)), (b,f(b)), (c,f(c)). Only when all three are distinct.
            try:
                t1 = (a - b) * (fb - fc)
                t2 = (a - c) * (fb - fa)
                num = (a - c) * (a - b) * (fb - fc) * (fb - fa)
                denom = t1 - t2
            except (OverflowError, ValueError):
                denom = 0.0  # force fall-through to secant / bisection
            if denom != 0.0 and math.isfinite(num) and math.isfinite(denom):
                s_candidate = b - num / denom
        if s_candidate is None and not same_fb_fc:
            # Secant through (b, f(b)) and (c, f(c)).
            denom = fb - fc
            if denom != 0.0:
                s_candidate = b - fb * (b - c) / denom

        # --- Accept or reject the candidate ----------------------------
        # Brent's acceptance conditions, restated as guards. Any failure
        # here means the fast step is unsafe; we bisect instead.
        if (
            s_candidate is None
            or not math.isfinite(s_candidate)
            or not (
                (b < s_candidate < a) or (a < s_candidate < b)
            )
            or (
                mflag
                and abs(s_candidate - b)
                >= abs(b - c) / 2.0
            )
            or (
                (not mflag)
                and abs(s_candidate - b)
                >= abs(c - d) / 2.0
            )
        ):
            use_bisection = True

        if use_bisection:
            s_candidate = (a + b) / 2.0
            mflag = True
        else:
            mflag = False

        # Evaluate f at the chosen point. f may not be a Python float
        # (numpy scalars, etc.); coerce for the isfinite / sign checks.
        fs_raw = f(s_candidate)
        try:
            fs = float(fs_raw)
        except (TypeError, ValueError):
            fs = float("nan")

        if not math.isfinite(fs):
            # Non-finite function value. Bisect instead and retry this pass
            # logic on the next iteration; the bracket itself is unchanged.
            s_candidate = (a + b) / 2.0
            mflag = True
            fs_raw = f(s_candidate)
            try:
                fs = float(fs_raw)
            except (TypeError, ValueError):
                fs = float("nan")
            if not math.isfinite(fs):
                # Even bisection produced a non-finite value; the function
                # is not usable here. Stop with what we have.
                return BrentResult(
                    root=b, f_root=fb, iterations=iter_count,
                    converged=False, bracket=(a, b),
                    reason="max_iterations",
                )

        # --- Update the bracket ----------------------------------------
        # The invariant: after this block, f(a) and f(b) still have opposite
        # sign, and b is the most recent iterate.
        d = c
        c = b
        fc = fb

        if _sign(fs) == _sign(fa):
            a = s_candidate
            fa = fs
            b = b  # no-op, kept for clarity of the bookkeeping
            fb = fb
        else:
            a = a  # no-op
            fa = fa
            b = s_candidate
            fb = fs

        # Tighten the bracket so that a and b are adjacent (smaller width).
        # Brent keeps the latest sample at b and the contrapoint at c; here
        # we keep a as the strict opposite-sign endpoint.
        # (No further work needed; the assignments above maintain the invariant.)

    # Exhausted the iteration budget.
    return BrentResult(
        root=b, f_root=fb, iterations=iter_count, converged=False,
        bracket=(a, b), reason="max_iterations",
    )
