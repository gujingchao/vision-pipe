from __future__ import annotations

import json
from pathlib import Path

from vision_pipe_cli.cli import main
from vision_pipe_cli.engine import dummy_infer, infer
from vision_pipe_cli.io_util import list_images


def _ppm(path: Path, w: int = 32, h: int = 24) -> Path:
    header = f"P6\n{w} {h}\n255\n".encode()
    path.write_bytes(header + bytes([200, 80, 80]) * (w * h))
    return path


def test_dummy_infer(tmp_path: Path):
    img = _ppm(tmp_path / "a.ppm")
    result = dummy_infer(img)
    assert result.width == 32
    assert result.height == 24
    assert result.detections[0].label == "object"
    assert infer(img).detections


def test_infer_cli_writes_json(tmp_path: Path):
    src = tmp_path / "images"
    src.mkdir()
    _ppm(src / "one.ppm")
    out = tmp_path / "out"
    assert main(["infer", "--input", str(src), "--output", str(out), "--save-json"]) == 0
    data = json.loads((out / "one.json").read_text())
    assert data["width"] == 32
    assert data["detections"][0]["xyxy"]


def test_bench_cli(tmp_path: Path, capsys):
    src = tmp_path / "images"
    src.mkdir()
    _ppm(src / "one.ppm")
    dest = tmp_path / "bench.json"
    cmd = ["bench", "--input", str(src), "--repeat", "5", "--warmup", "1", "--output", str(dest)]
    assert main(cmd) == 0
    payload = json.loads(dest.read_text())
    assert payload["samples"] == 5
    assert payload["fps"] >= 0
    assert list_images(src)[0].name == "one.ppm"
    capsys.readouterr()
