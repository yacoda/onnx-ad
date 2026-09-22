"""Expand operations the ONNX specification itself defines as a function of simpler ones.

Many recent operators -- `Attention`, `RMSNormalization`, `RotaryEmbedding`,
`GroupNormalization`, `Swish`, the loss functions -- come with a function body in the
spec: a graph of primitive operations that *is* their definition. An operation without a
rule of its own is expanded into that body before differentiation, so the rule table only
has to cover primitives, and the primal semantics are the specification's rather than a
reimplementation of them.

The body is taken at the model's own opset -- the newest body version no newer than it -- and
for a context-dependent function it is generated for the node's actual input types.
"""
from onnx import AttributeProto, FunctionProto, TypeProto, defs, helper

from ._build import FLOAT_TYPES, Builder, Shapes
from ._graph import all_constants, all_names, node_reads, subgraphs
from .fold import fold_constants, localize_constants
from .rules import FORWARD
from .unroll import _substitute, inline_constant_ifs


def expand_functions(model):
    """A copy of `model` with every rule-less, spec-defined function operation expanded.

    Expansion and constant folding alternate: a body's shape arithmetic often folds away,
    and a node whose operands are then all known is simply evaluated rather than expanded --
    `Range` on constants would otherwise become a `Loop`, its own spec body. Expansion also
    needs the rounds for a context-dependent function inside a body (SoftmaxCrossEntropyLoss
    calls NegativeLogLikelihoodLoss), which can only be generated once inference has typed
    the body. Nodes in custom domains, and operations with a rule, are left as they are.

    A body often branches on a condition that folds to a constant -- AffineGrid on whether it
    is 2-D -- so a constant `If` is inlined as part of the same fixed point, and the taken
    branch's own shape arithmetic then folds too: shape inference cannot see through a
    `Concat` of constants into an `Expand` target, and everything after it would lose its
    shape.
    """
    opset = max([o.version for o in model.opset_import if o.domain in ("", "ai.onnx")] or [18])
    for _ in range(24):
        model = fold_constants(model)
        inlined = inline_constant_ifs(model)
        if inlined is not model:
            model = inlined  # the branch's shape arithmetic folds in the next round
            continue
        state = _Expander(model, opset)
        nodes = state.nodes(list(model.graph.node))
        if not state.changed:
            return localize_constants(model)
        result = type(model)()
        result.CopyFrom(model)
        result.graph.ClearField("node")
        result.graph.node.extend(nodes)
        model = result
    return localize_constants(fold_constants(model))


class _Expander:
    def __init__(self, model, opset):
        self.opset = opset
        self.shapes = Shapes(model)
        self.known = set(all_constants(model.graph))
        self.b = Builder(all_names(model.graph))
        self.changed = False

    def nodes(self, nodes):
        out = []
        for node in nodes:
            evaluable = all(not name or name in self.known for name in node_reads(node))
            body = None if node.op_type in FORWARD or evaluable else self.function(node)
            if body is not None:
                self.changed = True
                out.extend(self.nodes(self.inline(node, *body)))
            elif subgraphs(node):
                out.append(self.descend(node))
            else:
                out.append(node)
        return out

    def descend(self, node):
        copy = helper.make_node(node.op_type, list(node.input), list(node.output),
                                name=node.name, domain=node.domain)
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                inner = type(attribute.g)()
                inner.CopyFrom(attribute.g)
                body = self.nodes(list(inner.node))
                inner.ClearField("node")
                inner.node.extend(body)
                copy.attribute.append(helper.make_attribute(attribute.name, inner))
            else:
                copy.attribute.append(attribute)
        return copy

    # ------------------------------------------------------------------ the body -----
    def function(self, node):
        """(FunctionProto, schema) for a node the spec defines as a function, else None."""
        if node.domain not in ("", "ai.onnx"):
            return None
        try:
            schema = defs.get_schema(node.op_type, max_inclusive_version=self.opset)
        except Exception:
            return None
        if schema.has_function:
            versions = [v for v in schema.function_opset_versions if v <= self.opset]
            if versions:
                body = _parsed(schema.get_function_with_opset_version(max(versions)))
                return body, schema, True
        if schema.has_context_dependent_function:
            versions = [v for v in schema.context_dependent_function_opset_versions
                        if v <= self.opset]
            types = [self.type_of(name) for name in node.input]
            if versions and all(t is not None for t in types):
                try:
                    body = schema.get_context_dependent_function_with_opset_version(
                        max(versions), node.SerializeToString(),
                        [t.SerializeToString() for t in types])
                except Exception:
                    return None
                return _parsed(body), schema, False
        return None

    def type_of(self, name):
        """The TypeProto a context-dependent function is generated for."""
        if not name:
            return TypeProto()  # an omitted optional input
        shape = self.shapes.shape(name)
        if shape is None:
            return None
        return helper.make_tensor_type_proto(self.shapes.dtype(name), list(shape))

    def inline(self, node, function, schema, generic):
        """The body's nodes, bound to the node's operands, internals under fresh names.

        A context-independent body is written once for every type it accepts, yet some
        spell their float constants as float32 -- MeanVarianceNormalization's exponent and
        epsilon -- which is invalid for a double input. In such a body each float Constant
        is cast like the first float operand, as the body evidently intends.
        """
        like = next((name for name in node.input
                     if name and self.shapes.dtype(name) in FLOAT_TYPES), None) \
            if generic else None
        mapping = {}
        for formal, actual in zip(function.input, node.input):
            mapping[formal] = actual
        for formal in list(function.input)[len(node.input):]:
            mapping[formal] = ""  # trailing optional inputs the caller left out
        produced = {formal: actual for formal, actual in zip(function.output, node.output)}
        emitted = []
        for inner in function.node:
            outputs = []
            for name in inner.output:
                if name in produced:
                    fresh = produced[name]
                elif name:
                    fresh = self.b.name(name + "_f")
                else:
                    fresh = ""
                mapping[name] = fresh
                outputs.append(fresh)
            copy = helper.make_node(inner.op_type, [mapping.get(n, n) for n in inner.input],
                                    outputs, name=inner.name, domain=inner.domain)
            for attribute in inner.attribute:
                resolved = _resolve(attribute, node, function, schema)
                if resolved is None:
                    continue
                if resolved.type == AttributeProto.GRAPH:
                    resolved = helper.make_attribute(resolved.name,
                                                     _substitute(resolved.g, mapping))
                copy.attribute.append(resolved)
            if like is not None and inner.op_type == "Constant" and _float_constant(inner):
                raw = self.b.name(copy.output[0] + "_raw")
                typed = copy.output[0]
                copy.output[0] = raw
                emitted.append(copy)
                emitted.append(helper.make_node("CastLike", [raw, like], [typed]))
                continue
            emitted.append(copy)
        # an output that is simply one of the inputs needs a node of its own
        for formal, actual in produced.items():
            if formal in function.input:
                emitted.append(helper.make_node("Identity", [mapping[formal]], [actual]))
        return emitted


def _float_constant(node):
    """Whether a Constant node holds a floating-point value."""
    for attribute in node.attribute:
        if attribute.name in ("value_float", "value_floats"):
            return True
        if attribute.name == "value" and attribute.t.data_type in FLOAT_TYPES:
            return True
    return False


def _parsed(body):
    if isinstance(body, bytes):
        proto = FunctionProto()
        proto.ParseFromString(body)
        return proto
    return body


def _resolve(attribute, caller, function, schema):
    """A body attribute, with a reference to the caller's attribute substituted -- or the
    function's or schema's default when the caller left it out, or None when there is none."""
    if not attribute.ref_attr_name:
        return attribute
    source = next((a for a in caller.attribute if a.name == attribute.ref_attr_name), None)
    if source is None:
        source = next((a for a in getattr(function, "attribute_proto", [])
                       if a.name == attribute.ref_attr_name), None)
    if source is None and attribute.ref_attr_name in schema.attributes:
        default = schema.attributes[attribute.ref_attr_name].default_value
        if default.type != AttributeProto.UNDEFINED:
            source = default
    if source is None:
        return None
    copy = AttributeProto()
    copy.CopyFrom(source)
    copy.name = attribute.name
    return copy
