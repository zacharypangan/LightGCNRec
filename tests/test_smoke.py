import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def test_package_imports_without_training_dependencies():
    import lightgcnrec

    assert lightgcnrec.__version__


def test_config_dry_run_reports_missing_yaml_dependency_or_success():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lightgcnrec.experiments.train",
            "--config",
            "configs/smoke_synthetic.yaml",
            "--dry-run",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    if importlib.util.find_spec("yaml") is None:
        assert "Install PyYAML" in result.stderr
    else:
        assert result.returncode == 0
        assert "Validated config" in result.stdout


def test_evaluate_entrypoint_with_json_files(tmp_path):
    recs = tmp_path / "recs.json"
    truth = tmp_path / "truth.json"
    recs.write_text(json.dumps({"0": [1, 2], "1": [3, 4]}), encoding="utf-8")
    truth.write_text(json.dumps({"0": [1], "1": [5]}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lightgcnrec.experiments.evaluate",
            "--recommendations",
            str(recs),
            "--ground-truth",
            str(truth),
            "--k",
            "2",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"recall@2": 0.5' in result.stdout


def test_notebooks_are_valid_and_clean():
    for path in Path("notebooks").rglob("*.ipynb"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "cells" in payload
        for cell in payload["cells"]:
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
