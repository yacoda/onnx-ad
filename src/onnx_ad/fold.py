"""Fold the shape arithmetic of a graph into constants.

Function bodies, and many exported graphs, compute axes, pads and target shapes at run
time -- `Shape`, then `Gather`, `Sub`, `Div`, `Concat` -- even where every shape involved is
static. A rule that needs such an operand to be a constant (a Pad's widths, a reduction's
axes) would then refuse. Evaluating those nodes once, where all their operands are known,
turns them back into constants.

Only small results are folded, so a large constant computation on the weights is left for
the runtime. Nothing random is folded; a node with a subgraph is, once everything it captures
is known.
"""
import numpy as np
from onnx import helper, numpy_helper

from ._build import Shapes
from ._graph import all_constants, node_reads

RANDOM = {"RandomNormal", "RandomUniform", "RandomNormalLike", "RandomUniformLike",
          "Bernoulli", "Multinomial"}


def fold_constants(model, max_elements=4096):
    """A copy of `model` with small, fully determined nodes replaced by Constants.

    Iterated to a fixed point: folding a Pad's widths is what lets inference shape its
    output, which is what lets the Shape of that output fold in turn.
    """
    for _ in range(16):
        folded = _fold_once(model, max_elements)
        if folded is model:
            return model
        model = folded
    return model


def _fold_once(model, max_elements):
    try:
        from onnx.reference import ReferenceEvaluator
    except ImportError:  # onnx without a reference evaluator: nothing to fold with
        return model
    graph = model.graph
    shapes = Shapes(model)
    values = dict(all_constants(graph))
    opset = [o for o in model.opset_import]
    folded, changed = [], False
    for node in graph.node:
        result = None
        if node.op_type in ("Shape", "Size") and node.input[0] not in values:
            result = _shape_query(node, shapes.static(node.input[0]))
        elif (node.op_type not in RANDOM and node.op_type != "Constant"
              and all(name in values for name in node_reads(node))):
            # a node with a subgraph folds too when everything it captures is known
            result = _evaluate(ReferenceEvaluator, node, values, opset)
        if result is not None and all(v.size <= max_elements for v in result):
            for name, value in zip(node.output, result):
                values[name] = value
                folded.append(helper.make_node(
                    "Constant", [], [name], value=numpy_helper.from_array(value, name)))
            changed = True
        else:
            folded.append(node)
    if not changed:
        return model
    out = type(model)()
    out.CopyFrom(model)
    out.graph.ClearField("node")
    out.graph.node.extend(folded)
    return out


def _shape_query(node, shape):
    if shape is None:
        return None
    if node.op_type == "Size":
        return [np.array(int(np.prod(shape, dtype=np.int64)), dtype=np.int64)]
    start = next((a.i for a in node.attribute if a.name == "start"), 0)
    end = next((a.i for a in node.attribute if a.name == "end"), None)
    return [np.array(list(shape)[start:end], dtype=np.int64)]


def _evaluate(evaluator, node, values, opset):
    """Run one node through the reference evaluator; None if it cannot."""
    inputs = sorted(node_reads(node))  # a subgraph's captures are inputs here too
    graph = helper.make_graph(
        [node], "fold",
        [helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(values[name].dtype),
                                       values[name].shape) for name in dict.fromkeys(inputs)],
        [helper.make_empty_tensor_value_info(name) for name in node.output if name])
    try:
        model = helper.make_model(graph, opset_imports=opset)
        results = evaluator(model).run(None, {name: values[name] for name in inputs})
    except Exception:
        return None
    return [np.asarray(r) for r in results]
