"""Export this authorized task's exact tool inputs and literal shell commands."""

import argparse
import ast
import json
import re
from pathlib import Path


def main(session, output):
    calls = []
    commands = []
    started = False
    pattern = re.compile(
        r'tools\.exec_command\(\{\s*cmd:\s*("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')'
    )
    for line in session.open():
        r = json.loads(line)
        c = r.get("payload", {})
        if c.get("role") == "user":
            t = " ".join(v.get("text", "") for v in c.get("content", []))
            if t.startswith("# Task: P6 Closure"):
                started = True
        if not started or c.get("type") != "custom_tool_call" or c.get("name") != "exec":
            continue
        code = c["input"]
        calls.append({"timestamp": r.get("timestamp"), "call_id": c.get("call_id"), "input": code})
        for match in pattern.finditer(code):
            literal = match.group(1)
            command = json.loads(literal) if literal.startswith('"') else ast.literal_eval(literal)
            commands.append(
                {
                    "number": len(commands) + 1,
                    "timestamp": r.get("timestamp"),
                    "call_id": c.get("call_id"),
                    "cwd": "/home/han/pyx/SMHOI",
                    "command": command,
                }
            )
    result = {
        "source_session": str(session),
        "scope": "Only calls after this task's P6 Closure user prompt",
        "commands": commands,
        "tool_inputs": calls,
        "exec_command_call_count": sum(c["input"].count("tools.exec_command(") for c in calls),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "commands",
        len(commands),
        "exec_command occurrences",
        result["exec_command_call_count"],
        "tool inputs",
        len(calls),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    main(a.session, a.output)
