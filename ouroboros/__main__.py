"""CLI: python -m ouroboros [serve | run [--once] | eval MODEL...]"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="ouroboros")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="open the web UI (default)")
    s.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    r = sub.add_parser("run", help="process jobs/pending headless, until empty")
    r.add_argument("--once", action="store_true", help="process a single job")
    sub.add_parser("briefs", help="write short LoRA descriptions for the LLM that picks LoRAs (see lora_briefs.py)",
                   add_help=False)
    e = sub.add_parser("eval", help="compare judge models on quality and speed (eval/judge_set.json)")
    e.add_argument("models", nargs="+", help="model names (Ollama tags, or hosted ids like deepseek-ai/...)")
    e.add_argument("--repeats", type=int, default=2, help="passes per image (2 measures consistency)")
    e.add_argument("--temp", type=float, default=None, help="override the model's temperature")
    e.add_argument("--backend", default=None, choices=["deepinfra", "ollama", "openai"],
                   help="where the models run (default: guessed from the name, else the configured backend)")
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "briefs":  # its own options (--force, --only, ...)
        from .lora_briefs import main as briefs_main
        return briefs_main(sys.argv[2:])
    args = ap.parse_args()

    if args.cmd == "eval":
        from .judge_eval import main as eval_main
        eval_main(args.models, args.repeats, args.temp, args.backend)
    elif args.cmd == "run":
        from .runner import Runner
        Runner().run_blocking(once=args.once)
    else:
        from .server import serve
        serve(open_browser=not getattr(args, "no_browser", False))


if __name__ == "__main__":
    main()
