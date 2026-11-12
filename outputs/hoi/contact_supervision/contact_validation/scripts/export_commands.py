"""Export exact shell commands from this task's own Codex tool-call transcript."""

import argparse
import json
import re
from pathlib import Path


def main(session, output):
    commands = []
    pattern = re.compile(r'tools\.exec_command\(\{cmd:\s*("(?:\\.|[^"\\])*")')
    for line in session.read_text().splitlines():
        record = json.loads(line)
        call = record.get("payload", {})
        if call.get("type") != "custom_tool_call" or call.get("name") != "exec":
            continue
        code = call["input"]
        for match in pattern.finditer(code):
            commands.append(
                {
                    "number": len(commands) + 1,
                    "timestamp": record.get("timestamp"),
                    "call_id": call.get("call_id"),
                    "cwd": "/home/han/pyx/SMHOI",
                    "command": json.loads(match.group(1)),
                }
            )
    output.write_text(
        json.dumps({"source_session": str(session), "commands": commands}, indent=2) + "\n"
    )
    print("Exact shell commands exported:", len(commands))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.session, args.output)
