"""Command line entry point.

The same engine the web UI uses, without a browser -- so an analysis can be
scripted, put in a pipeline, or checked in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import bundle as bundle_mod
from .core.data import DataError, template_rows
from .core.registry_likelihoods import LIKELIHOODS
from .core.registry_priors import PRIORS
from .core.resolve import SpecError
from .core.spec import ModelSpec, SamplerConfig


def _read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xltx"):
        import pandas as pd

        frame = pd.read_excel(path)
    elif path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    else:
        import pandas as pd

        frame = pd.read_csv(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    return frame.where(frame.notna(), None).to_dict(orient="records")


def cmd_list(_: argparse.Namespace) -> int:
    print("Likelihoods:")
    for lik in LIKELIHOODS.values():
        params = ", ".join(f"{p.name} ({p.role})" for p in lik.params)
        print(f"  {lik.id:22s} {lik.label}")
        print(f"  {'':22s} parameters: {params}")
        print(f"  {'':22s} columns: {', '.join(lik.required_columns)}")
    print("\nPrior families:")
    for prior in PRIORS.values():
        hypers = ", ".join(h.name for h in prior.hypers) or "(none)"
        proper = "" if prior.is_proper else "  [needs both bounds]"
        print(f"  {prior.id:18s} {prior.label:34s} {hypers}{proper}")
    return 0


def cmd_template(args: argparse.Namespace) -> int:
    header, rows = template_rows(args.likelihood)
    print(",".join(header))
    for row in rows:
        print(",".join(str(row.get(c, "")) for c in header))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from .core.render import render

    spec = ModelSpec.model_validate_json(Path(args.spec).read_text())
    print(render(spec).source, end="")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .core.results import (
        as_dict,
        draws_frame,
        reliability_curve,
        reliability_over_time,
        summarize,
    )
    from .core.sample import Progress, run

    spec = ModelSpec.model_validate_json(Path(args.spec).read_text())
    rows = _read_rows(Path(args.data)) if args.data else []
    # A sampler file (as `pirao example` writes) sets the base; any flag
    # given on the command line overrides it.
    sampler = (
        SamplerConfig.model_validate_json(Path(args.sampler).read_text())
        if args.sampler
        else SamplerConfig()
    )
    overrides = {
        "chains": args.chains,
        "iter_warmup": args.warmup,
        "iter_sampling": args.draws,
        "seed": args.seed,
    }
    sampler = sampler.model_copy(
        update={k: v for k, v in overrides.items() if v is not None}
    )

    def report(progress: Progress) -> None:
        print(f"[{progress.stage}] {progress.message}", file=sys.stderr)

    result = run(spec, rows, sampler, progress=report)
    summary = summarize(result)

    for warning in result.warnings:
        print(f"note: {warning}", file=sys.stderr)
    for diagnostic in summary.diagnostics:
        if diagnostic.severity != "ok":
            print(f"{diagnostic.severity}: {diagnostic.message}", file=sys.stderr)

    print(summary.table.to_string())
    curve = reliability_curve(result)
    if curve is not None:
        print("\nReliability at the mission:")
        print(curve.to_string(index=False))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        draws_frame(result).to_csv(out / "draws.csv", index=False)

        # The saved bundle carries what the GUI shows, so a run fitted from the
        # shell and one fitted from the browser are the same artifact.
        payload = as_dict(summary)
        if curve is not None:
            payload["reliability_curve"] = curve.to_dict(orient="records")
        over_time = reliability_over_time(result)
        if over_time is not None:
            payload["reliability_over_time"] = over_time

        bundle_mod.save(
            root=out,
            spec=spec,
            rows=rows,
            sampler=result.sampler,
            stan_source=result.rendered.source,
            summary=payload,
            warnings=result.warnings,
        )
        print(f"\nWritten to {out}", file=sys.stderr)
    return 0


def cmd_examples(_: argparse.Namespace) -> int:
    from .core.examples import list_examples

    for example in list_examples():
        kind = "" if example["data_kind"] == "public" else f"  [{example['data_kind']}]"
        print(f"  {example['id']:28s} {example['title']}{kind}")
    print("\nWrite one out with: pirao example <id> --out <dir>")
    return 0


def cmd_example(args: argparse.Namespace) -> int:
    import csv

    from .core.examples import ExampleNotFound, get_example

    try:
        example = get_example(args.id)
    except ExampleNotFound:
        print(
            f"error: no example called {args.id!r}; see `pirao examples`",
            file=sys.stderr,
        )
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(example["spec"], indent=2) + "\n")
    (out / "sampler.json").write_text(json.dumps(example["sampler"], indent=2) + "\n")
    columns = list(dict.fromkeys(c for row in example["rows"] for c in row))
    with open(out / "data.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(example["rows"])
    print(example["title"])
    print(example["note"])
    print(f"Source: {example['source']}\n")
    print(
        f"pirao run {out / 'spec.json'} --data {out / 'data.csv'} "
        f"--sampler {out / 'sampler.json'} --out {out / 'run'}"
    )
    return 0


def cmd_install_stan(_: argparse.Namespace) -> int:
    from .install import install_stan

    return install_stan()


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui import serve

    return serve(host=args.host, port=args.port, browser=not args.no_browser)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pirao",
        description="Fit Bayesian reliability models with a prior you choose.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "list", help="show the available likelihoods and priors"
    ).set_defaults(func=cmd_list)

    template = sub.add_parser("template", help="print a blank data template")
    template.add_argument("likelihood", choices=sorted(LIKELIHOODS))
    template.set_defaults(func=cmd_template)

    render_cmd = sub.add_parser("render", help="print the Stan program for a spec")
    render_cmd.add_argument("spec", help="path to a JSON model specification")
    render_cmd.set_defaults(func=cmd_render)

    run_cmd = sub.add_parser("run", help="fit a model")
    run_cmd.add_argument("spec", help="path to a JSON model specification")
    run_cmd.add_argument("--data", help="CSV, XLSX or JSON observation table")
    run_cmd.add_argument("--out", help="directory for draws and the run bundle")
    run_cmd.add_argument("--sampler", help="JSON sampler settings to start from")
    run_cmd.add_argument("--chains", type=int, help="default 4")
    run_cmd.add_argument("--warmup", type=int, help="default 1000")
    run_cmd.add_argument("--draws", type=int, help="default 1000")
    run_cmd.add_argument("--seed", type=int)
    run_cmd.set_defaults(func=cmd_run)

    sub.add_parser("examples", help="list the ready-made examples").set_defaults(
        func=cmd_examples
    )

    example_cmd = sub.add_parser(
        "example", help="write an example's spec.json and data.csv"
    )
    example_cmd.add_argument("id", help="example id, from `pirao examples`")
    example_cmd.add_argument("--out", default=".", help="directory to write into")
    example_cmd.set_defaults(func=cmd_example)

    sub.add_parser(
        "install-stan", help="install the pinned CmdStan (once; a few minutes)"
    ).set_defaults(func=cmd_install_stan)

    gui_cmd = sub.add_parser("gui", help="open the graphical interface")
    gui_cmd.add_argument("--host", default="127.0.0.1")
    gui_cmd.add_argument(
        "--port",
        type=int,
        default=None,
        help="default 8765, or a free port if that is taken",
    )
    gui_cmd.add_argument(
        "--no-browser", action="store_true", help="do not open a browser window"
    )
    gui_cmd.set_defaults(func=cmd_gui)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SpecError as exc:
        for message in exc.messages:
            print(f"error: {message}", file=sys.stderr)
        return 2
    except DataError as exc:
        for error in exc.errors:
            print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
