"""Write the sibling family CasADi's ONNX backend discovers from a primal model.

CasADi looks for a derivative model beside the primal one, named `<kind>_<filename>`:
`f.onnx` has siblings `adj_f.onnx` and `fwd_f.onnx`, and the adjoint in turn has
`fwd_adj_f.onnx`. That last file -- the forward derivative of the adjoint -- is what an
exact-Hessian solver needs, and it is just the two passes composed.

The family a solver wants is the primal, the adjoint, and the forward derivative of the
adjoint: reverse mode gives gradients at the cost of one evaluation, and differentiating it
forward gives Hessian-vector products. A `fwd_` sibling of the primal is optional and only
pays off when there are fewer inputs than outputs.
"""
import os

import onnx

from .forward import forward
from .reverse import reverse

#: The derivative kinds CasADi looks for, innermost first.
KINDS = ("adj", "fwd", "jac")


def sibling(path, kind):
    """CasADi's sibling rule: `<kind>_<filename>`, in the primal's own directory."""
    folder, filename = os.path.split(path)
    return os.path.join(folder, "%s_%s" % (kind, filename))


def family(model, path, forward_sibling=False, second_order=True, inputs=None, outputs=None):
    """Write `model` and its derivative siblings; returns the paths written.

    `path` is where the primal goes (`.../f.onnx`); the siblings are named from it. With
    `second_order`, the adjoint's own forward sibling `fwd_adj_<name>` is written too, which
    is what CasADi needs to build exact Hessians. `forward_sibling` additionally writes
    `fwd_<name>`, worth it only when the model has fewer inputs than outputs.

    The model is written as it is given -- make sure its initializers are inline rather than
    in an external data file, since CasADi hands the model to ONNX Runtime as bytes.
    """
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    written = [path]
    onnx.save(model, path)

    adjoint = reverse(model, inputs=inputs, outputs=outputs)
    onnx.save(adjoint, sibling(path, "adj"))
    written.append(sibling(path, "adj"))

    if forward_sibling:
        onnx.save(forward(model, inputs=inputs, outputs=outputs), sibling(path, "fwd"))
        written.append(sibling(path, "fwd"))

    if second_order:
        # Seed every input of the adjoint -- CasADi rejects an incomplete forward signature,
        # and the derivative with respect to the adjoint seed is part of it.
        produced = [v.name for v in adjoint.graph.output
                    if v.name not in {o.name for o in model.graph.output}]
        second = forward(adjoint, outputs=produced)
        onnx.save(second, sibling(sibling(path, "adj"), "fwd"))
        written.append(sibling(sibling(path, "adj"), "fwd"))
    return written
