# brent_s_method_py
#
# Public API:
#   brent - find a root of a scalar function on a bracketing interval.
#   BrentResult - the value returned by brent (see its attributes below).
#
# The implementation lives in brent_s_method_py.core.

from .core import BrentResult, brent

__all__ = ["brent", "BrentResult"]
