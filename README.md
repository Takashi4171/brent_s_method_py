# brent_s_method_py

Finds a root of a scalar function on a bracketing interval using Brent's method (bisection, secant, and inverse quadratic interpolation, with bisection as the unconditional fall-back).

## Usage

```python
from brent_s_method_py import brent, BrentResult

r: BrentResult = brent(lambda x: x * x - 2.0, 0.0, 5.0)
if r.converged:
    print(r.root, r.f_root, r.iterations, r.bracket)
else:
    print("no root:", r.reason)
```

Exported names: `brent`, `BrentResult`.

`brent(f, a, b, xtol=1e-12, max_iter=100) -> BrentResult`

`BrentResult` is a frozen dataclass with `root`, `f_root`, `iterations`, `converged`, `bracket`, and `reason` (one of `"converged"`, `"bad_bracket"`, `"max_iterations"`).

## Why this exists

Standard-library `scipy.optimize.brentq` is excellent but pulls SciPy. When the only job is "root of a continuous function on a bracket, no derivatives," SciPy is a heavy dependency. This library is one file, standard library only, and returns a structured result rather than raising.

The trade-off: no vectorised fancy interface, no relative tolerance across scales, no plural-root search. Tolerance is absolute (`xtol`, default `1e-12`), applied as `|b - a| < xtol and 2*|f(b)| < xtol`. For functions with steep slope or roots very close to zero, pass `xtol` explicitly.

## Awkward edge you will hit

If your function returns `NaN` or `inf` inside the interval, the implementation falls back to bisection for that step and, if even bisection yields a non-finite value, stops with `reason="max_iterations"`. A continuous, finite function on a real bracket is assumed; nothing here will rescue a function that is genuinely broken on the interval.
