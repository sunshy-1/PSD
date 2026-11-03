"""Portable launcher for the restricted Linux code-test subprocess."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def execute(program, scratch_dir):
    scratch = Path(scratch_dir); scratch.mkdir(parents=True, exist_ok=True)
    system_root = Path(Path(sys.executable).anchor)
    env = {"PATH": os.pathsep.join(str(p) for p in (system_root / "usr" / "bin", system_root / "bin")),
           "HOME": str(system_root / "tmp"), "CUDA_VISIBLE_DEVICES": "",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory(dir=scratch) as work:
        try:
            with (Path(work) / "result.txt").open("w+") as stream:
                result = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).with_name("sandbox_child.py"))],
                    input=json.dumps({"program": program}), text=True, stdout=stream, stderr=subprocess.STDOUT,
                    cwd=work, env=env, timeout=6)
                stream.seek(0)
                text = stream.read(10000)
            if result.returncode:
                return dict(passed=False, error="process_exit_" + str(result.returncode), detail=text[-1000:])
            verdict = json.loads(text)
            if not verdict.get("seccomp") or verdict.get("landlock_abi", 0) < 3:
                raise RuntimeError("Code execution isolation was not initialized")
            return verdict
        except subprocess.TimeoutExpired:
            return dict(passed=False, error="wall_timeout_6_seconds")


def score_code(prepared, output_dir):
    output = Path(output_dir); cache = output / "grading_cache"; cache.mkdir(parents=True, exist_ok=True)
    def grade(candidate):
        source = candidate + "\n" + prepared["test_code"]
        digest = hashlib.sha256(source.encode()).hexdigest(); path = cache / (digest + ".json")
        if path.exists():
            value = json.loads(path.read_text())
        else:
            value = execute(source, output / "execution_scratch")
            path.write_text(json.dumps(value))
        return dict(program_hash=digest, **value)
    normalized = grade(prepared["candidate_code"])
    raw = grade(prepared["raw_candidate_code"])
    return dict(correct=normalized["passed"], raw_correct=raw["passed"], accuracy_metric="normalized_pass@1",
                test_error=normalized.get("error"), execution=normalized, raw_execution=raw)


def validate_executor(output_dir):
    good = execute("assert 2 + 2 == 4", Path(output_dir) / "execution_scratch")
    bad = execute("assert 2 + 2 == 5", Path(output_dir) / "execution_scratch")
    if not good["passed"] or bad["passed"]:
        raise RuntimeError("Code-test executor self-check failed: " + repr((good, bad)))
    return dict(correct_program_passes=True, incorrect_program_fails=True, **good)
