"""basal-station command line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import __version__
from .config import load_settings


def _fmt(data) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = load_settings(args.config)
    uvicorn.run(
        "basal_station.api.app:create_app",
        factory=True,
        host=settings.server.host,
        port=settings.server.port,
        log_level=args.log_level,
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .backends import EngineManager
    from .hardware import detect
    from .knowledge import KnowledgeService

    settings = load_settings(args.config)
    hw = detect()
    print(f"basal-station {__version__}")
    print(f"  host    : {hw.os_label} {hw.arch}" + (f" ({hw.chip})" if hw.chip else ""))
    print(f"  cpu/ram : {hw.cpu_count} cores / {hw.ram_bytes / 1024**3:.0f} GB")
    for gpu in hw.gpus:
        print(f"  gpu     : [{gpu.vendor}] {gpu.name} — {gpu.vram_bytes / 1024**3:.1f} GB")
    kb = KnowledgeService(settings.knowledge)

    async def probe() -> dict:
        await kb.start()
        try:
            return await kb.status()
        finally:
            await kb.close()

    kstatus = asyncio.run(probe())
    print(f"  knowledge-palace: {'connected' if kstatus.get('connected') else 'NOT connected'}"
          f" ({' '.join(settings.knowledge.command)})")
    mgr = EngineManager(settings)
    st = mgr.status()
    print(f"  engine  : {'running' if st['running'] else 'stopped'} ({settings.engine.command})")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from .api.localrepo import model_payload
    from .catalog import list_models

    for m in list_models():
        p = model_payload(m)
        fmts = "  ".join(
            f"{f['format']}{'*' if f['local'] else ''}" for f in p["formats"]
        )
        media = " [media]" if m.media else ""
        print(f"{m.key:<18} {p['label']}{media}")
        print(f"{'':<18} formats: {fmts}   (* = cached)")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from .api import downloads
    from .catalog import get_model

    model = get_model(args.model)
    if model is None:
        print(f"error: unknown model '{args.model}' (see `basal-station models`)", file=sys.stderr)
        return 2
    if args.format not in model.repos:
        print(f"error: model '{args.model}' has no {args.format} repo", file=sys.stderr)
        return 2
    key = downloads.job_key(model.key, args.format, args.quant)
    print(f"downloading {model.repo(args.format)} ({args.format}"
          + (f", {args.quant}" if args.quant else "") + ")…")
    asyncio.run(downloads.run_download(downloads.submit(model, args.format, args.quant), model, args.format, args.quant))
    job = downloads.get_job(key) or {}
    if job.get("status") == "done":
        print(f"done → {job.get('path')}")
        return 0
    print(f"FAILED: {job.get('error')}", file=sys.stderr)
    return 1


def cmd_decide(args: argparse.Namespace) -> int:
    """One-shot decision. Exit 0 = decided, 3 = deferred / flagged by policy.

    Built for shell pipelines and agent harnesses:
        echo '{"state": ..., "questions': ...}' | basal-station decide -
    """
    from .policy import apply_policy

    if args.file == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(args.file).read_text()
    payload = json.loads(raw)

    settings = load_settings(args.config)

    async def run() -> tuple[dict, int]:
        from .backends import EngineManager

        mgr = EngineManager(settings)
        result = await mgr.decide(payload)
        policy_spec = payload.get("policy") or args.policy_json
        if policy_spec:
            result["policy"] = apply_policy(result.get("answers", {}), policy_spec)
        code = 0
        if (result.get("policy") or {}).get("flagged") or any(a.get("deferred") for a in result.get("answers", {}).values()):
            code = 3
        return result, code

    result, code = asyncio.run(run())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return code


def cmd_recipes(args: argparse.Namespace) -> int:
    from .recipes import list_recipes, recipe_payload

    for r in list_recipes():
        if args.role and r.role != args.role:
            continue
        print(f"{r.key:<22} [{r.role}] {r.title}")
        print(f"{'':<22} {r.description.splitlines()[0]}")
    if args.json:
        print(_fmt([recipe_payload(r) for r in list_recipes()]))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .eval import evaluate

    questions = json.loads(Path(args.questions).read_text())
    examples = []
    with open(args.file) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                examples.append(json.loads(line))
    if not examples:
        print("no examples found", file=sys.stderr)
        return 2

    settings = load_settings(args.config)

    async def run() -> dict:
        from .backends import EngineManager

        mgr = EngineManager(settings)
        results, labels = [], []
        for ex in examples:
            try:
                res = await mgr.decide({"state": ex["state"], "questions": questions})
            except Exception as exc:
                print(f"warning: {str(exc)[:160]}", file=sys.stderr)
                res = {"answers": {}}
            results.append(res)
            labels.append(ex.get("labels", {}))
        return evaluate(questions, results, labels, args.threshold)

    report = asyncio.run(run())
    print(_fmt(report))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .knowledge import KnowledgeService

    settings = load_settings(args.config)

    async def run() -> int:
        kb = KnowledgeService(settings.knowledge)
        await kb.start()
        try:
            out = await kb.ask(args.question)
            print(out if isinstance(out, str) else _fmt(out))
            return 0
        finally:
            await kb.close()

    return asyncio.run(run())


def cmd_search(args: argparse.Namespace) -> int:
    from .knowledge import KnowledgeService

    settings = load_settings(args.config)

    async def run() -> int:
        kb = KnowledgeService(settings.knowledge)
        await kb.start()
        try:
            out = await kb.search(args.query, limit=args.limit)
            print(out if isinstance(out, str) else _fmt(out))
            return 0
        finally:
            await kb.close()

    return asyncio.run(run())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="basal-station",
        description="Local control plane for basal typed-decision models.",
    )
    parser.add_argument("--version", action="version", version=f"basal-station {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def with_config(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--config", default=None, help="path to basal-station.yaml")
        return p

    p = with_config(sub.add_parser("serve", help="run the web UI + API"))
    p.add_argument("--log-level", default="info")

    with_config(sub.add_parser("status", help="show hardware, knowledge base and engine status"))
    sub.add_parser("models", help="list the model catalog")

    p = with_config(sub.add_parser("download", help="download model weights"))
    p.add_argument("model", help="catalog key, e.g. omni-1.0-max")
    p.add_argument("--format", default="gguf", choices=["gguf", "mlx", "fp8", "safetensors"])
    p.add_argument("--quant", default=None, help="GGUF quant, e.g. Q4_K_M")

    p = with_config(sub.add_parser(
        "decide",
        help="POST a /v1/systemone JSON (file or '-' for stdin) to a running engine; "
             "exit 3 when the policy flags/defers",
    ))
    p.add_argument("file", help="JSON file, or '-' for stdin")
    p.add_argument("--policy", default=None, dest="policy_json",
                   help="policy spec as inline JSON, e.g. '{\"fallback\":\"review\"}'")

    with_config(sub.add_parser("recipes", help="list the recipe cookbook"))
    p.add_argument("--role", default=None, help="filter: triage|router|gate|judge|ground")
    p.add_argument("--json", action="store_true", help="also dump full recipe specs")

    p = with_config(sub.add_parser("eval", help="score a labeled JSONL set ({state, labels})"))
    p.add_argument("file", help="JSONL file with {state, labels} per line")
    p.add_argument("--questions", required=True, help="questions spec JSON file")
    p.add_argument("--threshold", type=float, default=0.5)

    p = with_config(sub.add_parser("ask", help="ask the knowledge base"))
    p.add_argument("question")
    p.add_argument("--limit", type=int, default=None)

    p = with_config(sub.add_parser("search", help="search the knowledge base"))
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=5)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = {
        "serve": cmd_serve,
        "status": cmd_status,
        "models": cmd_models,
        "download": cmd_download,
        "decide": cmd_decide,
        "recipes": cmd_recipes,
        "eval": cmd_eval,
        "ask": cmd_ask,
        "search": cmd_search,
    }[args.cmd]
    try:
        return handler(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
