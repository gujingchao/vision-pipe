"""vision-pipe CLI entry."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from vision_pipe_cli import __version__
from vision_pipe_cli.engine import infer
from vision_pipe_cli.io_util import list_images, write_json, write_overlay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vision-pipe", description="Batch infer and bench")
    parser.add_argument("--version", action="version", version=f"vision-pipe {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    infer_p = sub.add_parser("infer", help="run detection on a file or directory")
    infer_p.add_argument("--input", required=True)
    infer_p.add_argument("--output", required=True)
    infer_p.add_argument("--model", default=None)
    infer_p.add_argument("--conf", type=float, default=0.25)
    infer_p.add_argument("--save-json", action="store_true")
    infer_p.add_argument("--save-overlay", action="store_true")

    bench_p = sub.add_parser("bench", help="latency / FPS benchmark")
    bench_p.add_argument("--input", required=True)
    bench_p.add_argument("--model", default=None)
    bench_p.add_argument("--conf", type=float, default=0.25)
    bench_p.add_argument("--repeat", type=int, default=20)
    bench_p.add_argument("--warmup", type=int, default=3)
    bench_p.add_argument("--output", default=None, help="optional bench JSON path")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "infer":
            print(_run_infer(args))
        else:
            print(_run_bench(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_infer(args: argparse.Namespace) -> str:
    images = list_images(Path(args.input))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in images:
        result = infer(path, model=args.model, conf=args.conf)
        if args.save_json or not args.save_overlay:
            write_json(result, out / f"{path.stem}.json")
            written += 1
        if args.save_overlay:
            write_overlay(path, result, out / f"{path.stem}_overlay{path.suffix}")
            written += 1
    return f"inferred {len(images)} image(s), wrote {written} file(s) -> {out}"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((q / 100) * (len(ordered) - 1)))))
    return ordered[idx]


def _run_bench(args: argparse.Namespace) -> str:
    images = list_images(Path(args.input))
    if args.repeat < 1:
        raise ValueError("repeat must be >= 1")
    for path in images:
        for _ in range(max(0, args.warmup)):
            infer(path, model=args.model, conf=args.conf)
    samples: list[float] = []
    for _ in range(args.repeat):
        for path in images:
            result = infer(path, model=args.model, conf=args.conf)
            samples.append(float(result.timings_ms.get("total", 0)))
    p50 = _percentile(samples, 50)
    p95 = _percentile(samples, 95)
    mean = sum(samples) / len(samples)
    fps = 1000.0 / mean if mean else 0.0
    payload = {
        "images": len(images),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "samples": len(samples),
        "mean_ms": round(mean, 3),
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "fps": round(fps, 3),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + "\n", encoding="utf-8")
    return text


if __name__ == "__main__":
    raise SystemExit(main())
