from __future__ import annotations

import argparse
import json
import sys

from semantic_index.build.runner import (
    build_full,
    build_incremental,
    build_plaintiff,
    init_db,
    resume_build,
    validate_build,
)
from semantic_index.config.settings import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semantic-index")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db")
    subparsers.add_parser("build-full")

    plaintiff = subparsers.add_parser("build-plaintiff")
    plaintiff.add_argument("--cgid", required=True)

    subparsers.add_parser("build-incremental")

    resume = subparsers.add_parser("resume-build")
    resume.add_argument("--build-id", default="latest")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--build-id", default="latest")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if args.command == "init-db":
        result = init_db(settings)
    elif args.command == "build-full":
        result = build_full(settings)
    elif args.command == "build-plaintiff":
        result = build_plaintiff(settings, cgid=args.cgid)
    elif args.command == "build-incremental":
        result = build_incremental(settings)
    elif args.command == "resume-build":
        result = resume_build(
            settings, build_id=None if args.build_id == "latest" else int(args.build_id)
        )
    elif args.command == "validate":
        build_id = None if args.build_id == "latest" else int(args.build_id)
        result = validate_build(settings, build_id=build_id)
    else:
        parser.error(f"unknown command: {args.command}")
        return 2

    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
