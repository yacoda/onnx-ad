"""Command line: write the derivative model, or the whole sibling family, of an ONNX model."""
import argparse

import onnx

from . import __version__, family, forward, reverse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="onnx-ad", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
            ("forward", "emit a model computing fwd_y = J . fwd_x"),
            ("reverse", "emit a model computing adj_x = J^T . adj_y"),
            ("family", "write the primal and the sibling derivative models CasADi discovers")]:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("input")
        command.add_argument("output")
        command.add_argument("--inputs", nargs="+",
                             help="graph inputs to differentiate [every floating-point one]")
        command.add_argument("--outputs", nargs="+",
                             help="graph outputs to differentiate [every floating-point one]")
        if name == "family":
            command.add_argument("--forward-sibling", action="store_true",
                                 help="also write fwd_<name>, worth it when inputs < outputs")
            command.add_argument("--no-second-order", dest="second_order",
                                 action="store_false",
                                 help="skip fwd_adj_<name>, the exact-Hessian sibling")
        else:
            command.add_argument("--prefix", help="derivative-tensor prefix [CasADi's rule]")
            command.add_argument("--dim", help="symbolic seed dimension [CasADi's rule]")
            command.add_argument("--layout", choices=["casadi", "onnx"], default="casadi",
                                 help="seed layout: CasADi's packed matrix, or a trailing "
                                      "axis on the primal's own shape [casadi]")
    args = parser.parse_args(argv)
    model = onnx.load(args.input)
    if args.command == "family":
        for path in family(model, args.output, forward_sibling=args.forward_sibling,
                           second_order=args.second_order, inputs=args.inputs,
                           outputs=args.outputs):
            print(path)
        return
    pass_ = forward if args.command == "forward" else reverse
    result = pass_(model, inputs=args.inputs, outputs=args.outputs, prefix=args.prefix,
                   dim=args.dim, layout=args.layout)
    onnx.checker.check_model(result)
    onnx.save(result, args.output)


if __name__ == "__main__":
    main()
