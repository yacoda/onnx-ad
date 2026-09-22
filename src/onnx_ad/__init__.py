"""Source-code-transforming automatic differentiation on ONNX graphs.

`forward` and `reverse` rewrite an ONNX model into another ONNX model that computes its
Jacobian-vector or vector-Jacobian products, by applying a rule per operation over the
protobuf. No framework is in the loop: any ONNX model has derivatives, from any producer,
and since a derivative model is itself an ONNX model, `forward(reverse(model))` gives
forward-over-adjoint -- the exact-Hessian building block -- with no special casing.

Several seed directions ride in one evaluation, and the conventions are CasADi's own: the
derivative prefixes and seed dimensions its `diff_prefix` rule would pick (`fwd_`/`nfwd`,
then `fwd2_`/`nfwd2`), the seed layout its 2-D reading of an ONNX tensor expects, and the
`<kind>_<filename>` sibling names its ONNX backend discovers. `family` writes the whole set.
"""
from ._build import UnsupportedOperator
from .family import family, sibling
from .forward import forward
from .reverse import reverse
from .unroll import UnrollTooLarge, unroll

__all__ = ["forward", "reverse", "family", "sibling", "unroll", "UnsupportedOperator",
           "UnrollTooLarge", "__version__"]
__version__ = "0.1.0"
