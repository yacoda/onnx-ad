"""Expand RNN, GRU and LSTM into a Scan over their per-step equations.

The spec defines the three recurrent operations by their equations but gives no function
body, so this module writes one: the input projections `X W^T + biases` for every time step
at once (one MatMul, outside the loop), then a `Scan` whose body is the recurrence itself --
the only part that is genuinely sequential. Differentiating that Scan is then ordinary
control flow, in both modes and to any order.

Supported: all three directions, both layouts, optional biases and initial states, LSTM
peepholes, GRU's `linear_before_reset`, and the activation functions the spec lists.
Refused, with a reason: `clip`, LSTM's `input_forget`, and `sequence_lens` that differ
across the batch.
"""
from onnx import TensorProto, helper

from ._build import Builder, Shapes, UnsupportedOperator, attribute
from ._graph import all_constants, all_names

RECURRENT = {"RNN": ("Tanh",), "GRU": ("Sigmoid", "Tanh"), "LSTM": ("Sigmoid", "Tanh", "Tanh")}
GATES = {"RNN": 1, "GRU": 3, "LSTM": 4}


def expand_recurrent(model):
    """A copy of `model` with every RNN, GRU and LSTM node replaced by a Scan."""
    if not any(node.op_type in RECURRENT for node in model.graph.node):
        return model
    opset = max([o.version for o in model.opset_import if o.domain in ("", "ai.onnx")] or [18])
    expander = _Recurrent(model, opset)
    nodes = []
    for node in model.graph.node:
        nodes.extend(expander.expand(node) if node.op_type in RECURRENT else [node])
    result = type(model)()
    result.CopyFrom(model)
    result.graph.ClearField("node")
    result.graph.node.extend(nodes)
    result.graph.initializer.extend(expander.b.initializers)
    return result


class _Recurrent:
    def __init__(self, model, opset):
        self.opset = opset
        self.shapes = Shapes(model)
        self.values = all_constants(model.graph)
        self.b = Builder(all_names(model.graph))

    # ---------------------------------------------------------------- spelling ---------
    def op(self, builder, op_type, inputs, **attrs):
        return builder.op(op_type, inputs, stem=op_type.lower() + "_r", **attrs)

    def unsqueeze(self, builder, value, axes):
        if self.opset >= 13:
            return self.op(builder, "Unsqueeze", [value, builder.ints(axes)])
        return self.op(builder, "Unsqueeze", [value], axes=list(axes))

    def slice(self, value, start, end, axis):
        b = self.b
        if self.opset >= 10:
            return self.op(b, "Slice", [value, b.ints([start]), b.ints([end]), b.ints([axis])])
        return self.op(b, "Slice", [value], starts=[start], ends=[end], axes=[axis])

    def pick(self, value, index):
        """value[index] along axis 0, the axis dropped."""
        return self.op(self.b, "Gather", [value, self.b.constant(index, TensorProto.INT64, ())],
                       axis=0)

    # ---------------------------------------------------------------- activations ------
    @staticmethod
    def slots(names, alphas, betas):
        """(name, alpha, beta) per activation slot. The spec's alpha and beta lists are
        consumed in slot order, by those activations that take the parameter."""
        alphas, betas = list(alphas), list(betas)
        takes = {"leakyrelu": (0.01, None), "thresholdedrelu": (1.0, None),
                 "elu": (1.0, None), "hardsigmoid": (0.2, 0.5), "affine": (1.0, 0.0),
                 "scaledtanh": (1.0, 1.0)}
        slots = []
        for name in names:
            alpha, beta = takes.get(name.lower(), (None, None))
            if alpha is not None and alphas:
                alpha = alphas.pop(0)
            if beta is not None and betas:
                beta = betas.pop(0)
            slots.append((name.lower(), alpha, beta))
        return slots

    def activation(self, builder, slot, value, dtype):
        """One of the spec's activation functions."""
        name, alpha, beta = slot
        direct = {"relu": "Relu", "tanh": "Tanh", "sigmoid": "Sigmoid", "softsign": "Softsign",
                  "softplus": "Softplus"}
        if name in direct:
            return self.op(builder, direct[name], [value])
        if name == "leakyrelu":
            return self.op(builder, "LeakyRelu", [value], alpha=alpha)
        if name == "thresholdedrelu":
            return self.op(builder, "ThresholdedRelu", [value], alpha=alpha)
        if name == "elu":
            return self.op(builder, "Elu", [value], alpha=alpha)
        if name == "hardsigmoid":
            return self.op(builder, "HardSigmoid", [value], alpha=alpha, beta=beta)
        if name == "affine":
            scaled = self.op(builder, "Mul", [value, builder.constant(alpha, dtype)])
            return self.op(builder, "Add", [scaled, builder.constant(beta, dtype)])
        if name == "scaledtanh":
            inner = self.op(builder, "Tanh", [self.op(builder, "Mul", [
                value, builder.constant(beta, dtype)])])
            return self.op(builder, "Mul", [inner, builder.constant(alpha, dtype)])
        raise UnsupportedOperator("recurrent activation '%s' is not supported" % name)

    # ---------------------------------------------------------------- the expansion ----
    def expand(self, node):
        op = node.op_type
        if attribute(node, "clip") is not None:
            raise UnsupportedOperator("%s with clip is not differentiated" % op)
        if op == "LSTM" and attribute(node, "input_forget", 0):
            raise UnsupportedOperator("LSTM with input_forget is not differentiated")
        gates = GATES[op]
        hidden = attribute(node, "hidden_size")
        if hidden is None:  # R is [directions, gates*hidden, hidden]
            shape = self.shapes.shape(node.input[2])
            if shape is None or shape[2] is None:
                raise UnsupportedOperator("%s needs hidden_size or a declared R shape" % op)
            hidden = shape[2]
        direction = attribute(node, "direction", "forward")
        directions = 2 if direction == "bidirectional" else 1
        layout = attribute(node, "layout", 0)
        dtype = self.shapes.dtype(node.input[0])
        activations = attribute(node, "activations") or list(RECURRENT[op])*directions
        if len(activations) == len(RECURRENT[op]) and directions == 2:
            activations = activations*2
        slots = self.slots(activations, attribute(node, "activation_alpha") or [],
                           attribute(node, "activation_beta") or [])
        operand = lambda i: node.input[i] if len(node.input) > i and node.input[i] else None

        x = node.input[0]
        if layout == 1:
            x = self.op(self.b, "Transpose", [x], perm=[1, 0, 2])  # [seq, batch, input]
        self.check_lengths(node, operand(4), x)
        batch_hidden = self.op(self.b, "Concat", [
            self.slice(self.op(self.b, "Shape", [x]), 1, 2, 0), self.b.ints([hidden])], axis=0)

        ys, hs, cs = [], [], []
        for d in range(directions):
            backwards = direction == "reverse" or d == 1
            chosen = slots[d*len(RECURRENT[op]):(d + 1)*len(RECURRENT[op])]
            w, r = self.pick(node.input[1], d), self.pick(node.input[2], d)
            projected = self.op(self.b, "MatMul", [x, self.op(self.b, "Transpose", [w],
                                                               perm=[1, 0])])
            bias = self.pick(operand(3), d) if operand(3) else None
            inputs = []   # per-gate projections, [seq, batch, hidden] each
            recurrent = []  # per-gate R^T, [hidden, hidden]
            reset_bias = None
            for k in range(gates):
                gate = self.slice(projected, k*hidden, (k + 1)*hidden, 2)
                if bias is not None:
                    wb = self.slice(bias, k*hidden, (k + 1)*hidden, 0)
                    rb = self.slice(bias, (gates + k)*hidden, (gates + k + 1)*hidden, 0)
                    if op == "GRU" and k == 2 and attribute(node, "linear_before_reset", 0):
                        reset_bias = rb  # applied inside, before the reset gate
                        gate = self.op(self.b, "Add", [gate, wb])
                    else:
                        gate = self.op(self.b, "Add", [gate, self.op(self.b, "Add", [wb, rb])])
                inputs.append(gate)
                recurrent.append(self.op(self.b, "Transpose", [
                    self.slice(r, k*hidden, (k + 1)*hidden, 0)], perm=[1, 0]))
            h0 = self.pick(self.directions_first(operand(5), layout), d) if operand(5) \
                else self.zeros(batch_hidden, dtype)
            state = [h0]
            peepholes = None
            if op == "LSTM":
                state.append(self.pick(self.directions_first(operand(6), layout), d)
                             if operand(6) else self.zeros(batch_hidden, dtype))
                if operand(7):
                    p = self.pick(operand(7), d)
                    peepholes = [self.slice(p, k*hidden, (k + 1)*hidden, 0) for k in range(3)]
            body = self.body(op, chosen, recurrent, peepholes, reset_bias, hidden, dtype)
            outs = [self.b.name("state_r") for _ in state] + [self.b.name("sequence_r")]
            order = [1 if backwards else 0]
            self.b.nodes.append(helper.make_node(
                "Scan", state + inputs, outs, body=body, num_scan_inputs=gates,
                scan_input_directions=order*gates, scan_output_directions=order))
            ys.append(self.unsqueeze(self.b, outs[-1], [1]))
            hs.append(self.unsqueeze(self.b, outs[0], [0]))
            if op == "LSTM":
                cs.append(self.unsqueeze(self.b, outs[1], [0]))

        results = [self.concat(ys, 1), self.concat(hs, 0)] + \
            ([self.concat(cs, 0)] if op == "LSTM" else [])
        if layout == 1:
            results[0] = self.op(self.b, "Transpose", [results[0]], perm=[2, 0, 1, 3])
            results[1:] = [self.op(self.b, "Transpose", [v], perm=[1, 0, 2]) for v in results[1:]]
        for value, name in zip(results, node.output):
            if name:
                self.b.nodes.append(helper.make_node("Identity", [value], [name]))
        emitted, self.b.nodes = self.b.nodes, []
        return emitted

    def directions_first(self, state, layout):
        """An initial state as [directions, batch, hidden]: layout 1 gives it batch first."""
        if layout == 1:
            return self.op(self.b, "Transpose", [state], perm=[1, 0, 2])
        return state

    def concat(self, parts, axis):
        return parts[0] if len(parts) == 1 else self.op(self.b, "Concat", parts, axis=axis)

    def zeros(self, shape, dtype):
        return self.op(self.b, "ConstantOfShape", [shape],
                       value=helper.make_tensor("zero", dtype, [1], [0]))

    def check_lengths(self, node, lengths, x):
        """Variable-length sequences would need per-step masking; only equal lengths, or
        lengths equal to the sequence length, are accepted."""
        if lengths is None:
            return
        value = self.values.get(lengths)
        shape = self.shapes.shape(x)
        steps = shape[0] if shape is not None else None
        if value is None or steps is None or not (value == steps).all():
            raise UnsupportedOperator(
                "%s with sequence_lens shorter than the sequence is not differentiated"
                % node.op_type)

    # ---------------------------------------------------------------- one step ---------
    def body(self, op, activations, recurrent, peepholes, reset_bias, hidden, dtype):
        """The recurrence as a Scan body: (state..., projected gate inputs...) ->
        (state..., output). Its constants are local, as every subgraph's must be."""
        b = self.b.child()
        vi = lambda name: helper.make_tensor_value_info(name, dtype, [None, hidden])
        h = b.name("h_r")
        gate_names = [b.name("x%d_r" % k) for k in range(GATES[op])]
        state_in = [h]
        if op == "LSTM":
            c = b.name("c_r")
            state_in.append(c)

        def act(i, value):
            return self.activation(b, activations[i], value, dtype)

        def pre(k, carried):
            """The input projection plus the recurrent one for gate k."""
            return self.op(b, "Add", [gate_names[k], self.op(b, "MatMul", [carried, recurrent[k]])])

        if op == "RNN":
            new = [act(0, pre(0, h))]
        elif op == "GRU":
            z = act(0, pre(0, h))
            reset = act(0, pre(1, h))
            if reset_bias is None:
                candidate = act(1, pre(2, self.op(b, "Mul", [reset, h])))
            else:
                inner = self.op(b, "Add", [self.op(b, "MatMul", [h, recurrent[2]]), reset_bias])
                candidate = act(1, self.op(b, "Add", [gate_names[2],
                                                      self.op(b, "Mul", [reset, inner])]))
            keep = self.op(b, "Sub", [b.constant(1.0, dtype), z])
            new = [self.op(b, "Add", [self.op(b, "Mul", [keep, candidate]),
                                      self.op(b, "Mul", [z, h])])]
        else:  # LSTM, gates in the spec's i, o, f, c order
            def peep(k, cell):
                return None if peepholes is None else self.op(b, "Mul", [peepholes[k], cell])

            def with_peephole(k, index, cell):
                value = pre(k, h)
                extra = peep(index, cell)
                return value if extra is None else self.op(b, "Add", [value, extra])

            i_gate = act(0, with_peephole(0, 0, c))
            f_gate = act(0, with_peephole(2, 2, c))
            candidate = act(1, pre(3, h))
            cell = self.op(b, "Add", [self.op(b, "Mul", [f_gate, c]),
                                      self.op(b, "Mul", [i_gate, candidate])])
            o_gate = act(0, with_peephole(1, 1, cell))
            new = [self.op(b, "Mul", [o_gate, act(2, cell)]), cell]
        outputs = [b.name("h_next_r")] + ([b.name("c_next_r")] if op == "LSTM" else [])
        for value, name in zip(new, outputs):
            b.nodes.append(helper.make_node("Identity", [value], [name]))
        sequence = b.name("y_r")
        b.nodes.append(helper.make_node("Identity", [outputs[0]], [sequence]))
        graph = helper.make_graph(
            b.nodes, "recurrence", [vi(n) for n in state_in + gate_names],
            [vi(n) for n in outputs + [sequence]], b.initializers)
        return graph
