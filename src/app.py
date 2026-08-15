"""CodeMix command line entry point.

This file was empty. It now dispatches to the three things the project does,
so there is one obvious way in:

    python src/app.py audit     data/Processed/code1.json
    python src/app.py translate --text "మన goals ద్వారా we can do wonders"
    python src/app.py build     raw.conll -o data/Processed/lince.json --format conll
"""

from __future__ import annotations

import sys

COMMANDS = {
    "audit":     ("corpus",          "measure whether a corpus is real language"),
    "translate": ("evaluation",      "translate code-mixed text and embed it"),
    "build":     ("data_collection", "convert a real dataset into corpus JSON"),
}


def usage() -> str:
    lines = ["usage: python src/app.py <command> [args...]", "", "commands:"]
    lines += [f"  {name:<10} {desc}" for name, (_, desc) in COMMANDS.items()]
    lines += ["", "Run a command with --help for its own options."]
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0

    command = argv[0]
    if command not in COMMANDS:
        print(f"unknown command {command!r}\n", file=sys.stderr)
        print(usage(), file=sys.stderr)
        return 2

    module_name = COMMANDS[command][0]
    module = __import__(module_name)
    return module.main(argv[1:]) or 0


if __name__ == "__main__":
    raise SystemExit(main())
