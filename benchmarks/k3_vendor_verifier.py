#!/usr/bin/env python3
"""Pinned Moonshot K3 Vendor Verifier orchestration for the local exact runtime.

This script never substitutes home-grown benchmark prompts/scorers for Moonshot's public
verifier. It checks out the pinned upstream commit, invokes its own pytest/Inspect/BEAM
entrypoints against localhost, extracts Inspect's official accuracy metric, and gates the
result against Moonshot's submitted K3 numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE / "k3_vendor_verifier_contract.json"


def load_contract(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != 1:
        raise SystemExit(f"unsupported vendor verifier contract schema: {doc.get('schema')}")
    return doc


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def git_output(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def prepare_checkout(contract: dict[str, Any], dest: Path) -> None:
    upstream = contract["upstream"]
    repo, sha = upstream["repository"], upstream["commit"]
    if not (dest / ".git").is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        run(["git", "clone", "--filter=blob:none", "--no-checkout", repo, str(dest)])
    run(["git", "fetch", "--force", "origin", sha], cwd=dest)
    run(["git", "checkout", "--detach", sha], cwd=dest)
    actual = git_output(dest, "rev-parse", "HEAD")
    if actual != sha:
        raise SystemExit(f"Vendor Verifier checkout mismatch: {actual} != {sha}")
    # Prompt-token fixtures and BEAM data use Git LFS. Fail closed if LFS is unavailable.
    try:
        run(["git", "lfs", "pull"], cwd=dest)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit("git-lfs is required for the pinned Vendor Verifier fixtures") from exc
    print(f"PINNED KIMI VENDOR VERIFIER PASS {actual}")


def endpoint_env(base_url: str, api_key: str, model: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "KIMI_BASE_URL": base_url.rstrip("/"),
            "KIMI_API_KEY": api_key,
            "MODEL_NAME": model,
            "THINK_MODE": "kimi",
        }
    )
    return env


def common_pytest_args(base_url: str, api_key: str, model: str) -> list[str]:
    return [
        "--base-url", base_url.rstrip("/"),
        "--api-key", api_key,
        "--smoke-model", model,
        "--think-mode", "kimi",
        "-ra", "-v",
    ]


def run_preflight(checkout: Path, *, base_url: str, api_key: str, model: str, jobs: int) -> None:
    env = endpoint_env(base_url, api_key, model)
    common = common_pytest_args(base_url, api_key, model)
    suites = ["tests/params", "tests/k3_features", "tests/prompt_tokens"]
    for suite in suites:
        run([sys.executable, "-m", "pytest", suite, *common], cwd=checkout, env=env)
    run(
        [
            sys.executable, "-m", "pytest", "tests/tool_call_json_schema",
            *common,
            "--thinking",
            "--max-tokens", "2048",
            "-n", str(max(1, jobs)),
            "--reruns", "3",
            "--reruns-delay", "2",
            "--tool-json-report", "tool-call-schema-report.json",
        ],
        cwd=checkout,
        env=env,
    )
    print("K3 VENDOR PREFLIGHT PASS")


def extract_inspect_accuracy(log_dir: Path) -> tuple[float, str]:
    from inspect_ai.log import list_eval_logs, read_eval_log

    candidates = list_eval_logs(str(log_dir), descending=True)
    if not candidates:
        raise SystemExit(f"no Inspect logs in {log_dir}")
    diagnostics: list[str] = []
    for info in candidates:
        path = getattr(info, "name", None) or getattr(info, "file", None) or str(info)
        log = read_eval_log(path, header_only=True)
        diagnostics.append(f"{path}:{log.status}")
        if log.status != "success" or log.results is None:
            continue
        found: list[float] = []
        for score in log.results.scores:
            for metric in score.metrics.values():
                if metric.name == "accuracy":
                    found.append(float(metric.value))
        if len(found) == 1:
            return found[0], str(path)
        if found:
            raise SystemExit(f"ambiguous accuracy metrics in {path}: {found}")
    raise SystemExit("no successful Inspect accuracy result; " + "; ".join(diagnostics))


def run_inspect(
    contract: dict[str, Any],
    checkout: Path,
    bench: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    log_dir: Path,
    max_connections: int,
) -> dict[str, Any]:
    canonical = {"ocrbench": "OCRBench", "mmmu": "MMMU Pro Vision"}[bench]
    params = contract["official_eval_parameters"][canonical]
    log_dir.mkdir(parents=True, exist_ok=True)
    for old in log_dir.glob("*"):
        if old.is_file() or old.is_symlink():
            old.unlink()
        elif old.is_dir():
            shutil.rmtree(old)
    env = endpoint_env(base_url, api_key, model)
    env["INSPECT_LOG_DIR"] = str(log_dir.resolve())
    run(
        [
            sys.executable, "eval.py", bench,
            "--model", f"kimi/{model}",
            "--max-tokens", str(params["max_tokens"]),
            "--thinking",
            "--think-mode", "kimi",
            "--thinking-effort", "max",
            "--stream",
            "--max-connections", str(max(1, max_connections)),
            "--epochs", str(params["epochs"]),
            "--temperature", "1.0",
            "--top-p", "0.95",
        ],
        cwd=checkout,
        env=env,
    )
    score, log_path = extract_inspect_accuracy(log_dir)
    target = float(contract["official_targets"][canonical])
    result = {
        "benchmark": canonical,
        "score": score,
        "moonshot_target": target,
        "pass": score >= target,
        "inspect_log": log_path,
        "verifier_commit": contract["upstream"]["commit"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if score < target:
        raise SystemExit(f"{canonical}: {score} < Moonshot target {target}")
    return result


def run_beam_generate(
    contract: dict[str, Any],
    checkout: Path,
    *,
    base_url: str,
    api_key: str,
    model: str,
    tokenizer: Path,
    output: Path,
    concurrency: int,
    limit: int,
) -> None:
    cmd = [
        sys.executable, "beam/beam_generate.py",
        "--model", model,
        "--base-url", base_url.rstrip("/"),
        "--api-key", api_key,
        "--temperature", "1.0",
        "--top-p", "0.95",
        "--max-tokens", "32768",
        "--max-context-tokens", "1048576",
        "--thinking-json", '{"thinking":{"type":"enabled","keep":"all","effort":"max"}}',
        "--tokenizer", str(tokenizer.resolve()),
        "--concurrency", str(max(1, concurrency)),
        "--output", str(output.resolve()),
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    run(cmd, cwd=checkout, env=endpoint_env(base_url, api_key, model))


def run_beam_judge(
    contract: dict[str, Any],
    checkout: Path,
    *,
    answers: Path,
    output: Path,
    judge_base_url: str,
    judge_api_key: str,
    judge_model: str,
    judge_reasoning_effort: str,
    concurrency: int,
) -> float:
    cmd = [
        sys.executable, "beam/beam_judge.py",
        "--answers", str(answers.resolve()),
        "--judge-model", judge_model,
        "--judge-base-url", judge_base_url.rstrip("/"),
        "--judge-api-key", judge_api_key,
        "--judge-max-tokens", "16384",
        "--concurrency", str(max(1, concurrency)),
        "--output", str(output.resolve()),
    ]
    if judge_reasoning_effort:
        cmd += ["--judge-reasoning-effort", judge_reasoning_effort]
    run(cmd, cwd=checkout)
    scores: list[float] = []
    with output.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scores.append(float(json.loads(line)["score"]))
    if not scores:
        raise SystemExit("BEAM judge produced no scores")
    value = statistics.fmean(scores)
    target = float(contract["official_targets"]["BEAM (1M)"])
    print(json.dumps({"benchmark": "BEAM (1M)", "score": value, "moonshot_target": target}, indent=2))
    if value < target:
        raise SystemExit(f"BEAM (1M): {value} < Moonshot target {target}")
    return value


def gate_deepswe(contract: dict[str, Any], score: float) -> None:
    target = float(contract["official_targets"]["DeepSWE"])
    print(json.dumps({"benchmark": "DeepSWE", "score": score, "moonshot_target": target}, indent=2))
    if score < target:
        raise SystemExit(f"DeepSWE: {score} < Moonshot target {target}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("checkout", type=Path)

    p = sub.add_parser("preflight")
    p.add_argument("checkout", type=Path)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default="local-only")
    p.add_argument("--model", default="kimi-k3-local")
    p.add_argument("--jobs", type=int, default=1)

    p = sub.add_parser("inspect")
    p.add_argument("checkout", type=Path)
    p.add_argument("bench", choices=["ocrbench", "mmmu"])
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default="local-only")
    p.add_argument("--model", default="kimi-k3-local")
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--max-connections", type=int, default=1)
    p.add_argument("--result", type=Path)

    p = sub.add_parser("beam-generate")
    p.add_argument("checkout", type=Path)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default="local-only")
    p.add_argument("--model", default="kimi-k3-local")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)

    p = sub.add_parser("beam-judge")
    p.add_argument("checkout", type=Path)
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--judge-base-url", required=True)
    p.add_argument("--judge-api-key", required=True)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-reasoning-effort", default="")
    p.add_argument("--concurrency", type=int, default=1)

    p = sub.add_parser("deepswe-gate")
    p.add_argument("score", type=float)

    args = ap.parse_args()
    contract = load_contract(args.contract)
    if args.cmd == "prepare":
        prepare_checkout(contract, args.checkout)
    elif args.cmd == "preflight":
        run_preflight(args.checkout, base_url=args.base_url, api_key=args.api_key, model=args.model, jobs=args.jobs)
    elif args.cmd == "inspect":
        result = run_inspect(contract, args.checkout, args.bench, base_url=args.base_url, api_key=args.api_key, model=args.model, log_dir=args.log_dir, max_connections=args.max_connections)
        if args.result:
            args.result.parent.mkdir(parents=True, exist_ok=True)
            args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.cmd == "beam-generate":
        run_beam_generate(contract, args.checkout, base_url=args.base_url, api_key=args.api_key, model=args.model, tokenizer=args.tokenizer, output=args.output, concurrency=args.concurrency, limit=args.limit)
    elif args.cmd == "beam-judge":
        run_beam_judge(contract, args.checkout, answers=args.answers, output=args.output, judge_base_url=args.judge_base_url, judge_api_key=args.judge_api_key, judge_model=args.judge_model, judge_reasoning_effort=args.judge_reasoning_effort, concurrency=args.concurrency)
    elif args.cmd == "deepswe-gate":
        gate_deepswe(contract, args.score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
