#!/usr/bin/env python3
"""
FREE coding agent: same idea as coding_agent.py, but uses a free model.

Two free options (both speak the OpenAI-compatible API, so one script handles both):
  1. gemini  - Google Gemini API free tier (needs a free key from aistudio.google.com)
  2. ollama  - a model running on your own PC via Ollama (no key, no internet needed)

Setup:
    pip install openai pytest

Gemini:
    $env:GEMINI_API_KEY="your_free_key"          # PowerShell
    python coding_agent_free.py "Fix the failing tests" --provider gemini --workspace ./practice_project

Ollama (install from ollama.com first, then: ollama pull qwen2.5-coder:7b):
    python coding_agent_free.py "Fix the failing tests" --provider ollama --workspace ./practice_project

Safety: file tools are limited to the workspace; edits and commands ask for approval.
This is NOT a real sandbox. Use a throwaway folder.
"""

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import openai
from openai import OpenAI

# Errors worth waiting out and retrying (server busy, rate limit, network blip)
RETRYABLE = (openai.InternalServerError, openai.RateLimitError, openai.APIConnectionError)
RETRY_WAITS = [10, 20, 40, 60, 90]  # seconds to wait before each retry

MAX_STEPS = 25
MAX_OUTPUT_CHARS = 8000
MAX_FILE_CHARS = 30000   # read_file can return more than command output
COMMAND_TIMEOUT = 60
BACKUP_ROOT = Path(__file__).resolve().parent / "agent_backups"

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "default_model": "gemini-flash-latest",
        "default_delay": 5.0,   # seconds between steps, to stay under free-tier requests/minute
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,        # Ollama ignores the key
        "default_model": "qwen2.5-coder:7b",
        "default_delay": 0.0,
    },
}

def explain_rejection(e) -> str:
    """Build a readable message from an API rejection (400/401/403)."""
    detail = ""
    try:
        detail = e.response.text.strip()[:600]
    except Exception:
        pass
    if not detail:
        detail = str(e)
    return (
        f"\nStopped: the server rejected the request ({type(e).__name__}).\n"
        f"Server said: {detail}\n\n"
        "Most common cause: the API key is wrong, expired or was deleted.\n"
        "  1. Create a new key at aistudio.google.com (API keys).\n"
        "  2. In this window run:  $env:GEMINI_API_KEY=\"your_new_key\"\n"
        "  3. Test it with:        python coding_agent_free.py --list-models --provider gemini\n"
        "If the key is fine, the model name may not accept tool use: try --model gemini-3.1-flash-lite"
    )


SYSTEM_PROMPT = """You are a careful coding agent working inside a project directory.

Workflow:
1. Explore first: list files and search before changing anything.
2. For bugs: reproduce the problem (run the tests) before fixing it.
3. Make small, targeted edits with edit_file rather than rewriting whole files.
4. After every change, run the tests to verify it worked.
5. If something fails, read the error, form a hypothesis, and try again.
6. Fix the code, never edit the tests just to make them pass.
7. Before finishing, call show_changes to review your own edits and check they match the task.
8. When finished, give a short summary of what you changed and why.

Never guess at file contents. Read the file first, and copy text for edit_file EXACTLY as it
appears after the '| ' in read_file output, including indentation. For long files, read them in
sections with start_line and end_line. Use the search tool instead of grep. To add text to the END of an existing file, use append_file. Stay inside the
project directory."""

if sys.platform == "win32":
    SYSTEM_PROMPT += "\n\nThe computer runs Windows. Unix commands like grep, cat and ls do not exist."

# Tool schemas in OpenAI function-calling format
def _tool(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}

S = {"type": "string"}
TOOLS = [
    _tool("list_dir", "List files and folders in a directory (relative to the workspace).", {"path": S}, ["path"]),
    _tool("read_file", "Read a text file with line numbers. For long files use start_line and end_line to read a section.",
          {"path": S, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]),
    _tool("search", "Search all files in the workspace for a plain-text string. Returns path:line: text.", {"pattern": S}, ["pattern"]),
    _tool("edit_file", "Replace an exact string in a file. old_str must appear exactly once.",
          {"path": S, "old_str": S, "new_str": S}, ["path", "old_str", "new_str"]),
    _tool("write_file", "Create a new file or fully overwrite an existing one.",
          {"path": S, "content": S}, ["path", "content"]),
    _tool("append_file", "Add text to the end of an existing file without changing anything else.",
          {"path": S, "content": S}, ["path", "content"]),
    _tool("show_changes", "Show a diff of the files you have changed in this run so far, to review your own work. Optionally give a path to see just one file.", {"path": S}, []),
    _tool("run_command", "Run a shell command in the workspace (e.g. 'pytest -x').", {"command": S}, ["command"]),
]


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------
class RunLog:
    """Writes everything the agent does to a text file in agent_logs/ (next to this script)."""

    def __init__(self, enabled: bool, model: str, workspace, task: str):
        self.f = None
        self.path = None
        if enabled:
            folder = Path(__file__).resolve().parent / "agent_logs"
            folder.mkdir(exist_ok=True)
            self.path = folder / datetime.now().strftime("run-%Y%m%d-%H%M%S.txt")
            self.f = open(self.path, "w", encoding="utf-8")
            self.write(f"Task: {task}\nModel: {model}\nWorkspace: {workspace}\n"
                       f"Started: {datetime.now().isoformat(timespec='seconds')}\n")

    def write(self, text: str) -> None:
        if self.f:
            self.f.write(text + "\n")
            self.f.flush()

    def close(self) -> None:
        if self.f:
            self.write(f"\nEnded: {datetime.now().isoformat(timespec='seconds')}")
            self.f.close()
            self.f = None


class BackupStore:
    """Saves a copy of each file before the agent changes it, so changes can be reviewed and undone.
    Only edits made with edit_file / write_file / append_file are tracked (not shell commands)."""

    def __init__(self, workspace, enabled: bool = True):
        self.enabled = enabled
        self.workspace = Path(workspace).resolve()
        self.entries = {}   # relative path -> did the file exist before the agent touched it?
        self.dir = None
        if enabled:
            self.dir = BACKUP_ROOT / datetime.now().strftime("run-%Y%m%d-%H%M%S")

    def _rel(self, p: Path) -> str:
        return str(p.resolve().relative_to(self.workspace)).replace("\\", "/")

    def _save_manifest(self) -> None:
        manifest = {"workspace": str(self.workspace), "entries": self.entries, "undone": False}
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def before_change(self, p: Path) -> None:
        """Call right before a file is changed. Only the first call per file is kept."""
        if not self.enabled:
            return
        rel = self._rel(p)
        if rel in self.entries:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        existed = p.is_file()
        if existed:
            dest = self.dir / "files" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        self.entries[rel] = existed
        self._save_manifest()

    def _old_text(self, rel: str, existed: bool) -> str:
        if not existed:
            return ""
        return (self.dir / "files" / rel).read_text(encoding="utf-8", errors="replace")

    def _new_text(self, rel: str) -> str:
        fp = self.workspace / rel
        return fp.read_text(encoding="utf-8", errors="replace") if fp.is_file() else ""

    def changed_files(self):
        """[(relative path, is_new_file)] for files whose content really differs from the backup."""
        out = []
        for rel, existed in self.entries.items():
            if self._old_text(rel, existed) != self._new_text(rel):
                out.append((rel, not existed))
        return out

    def diff(self, only: str = None) -> str:
        parts = []
        for rel, existed in self.entries.items():
            if only and rel != only:
                continue
            lines = list(difflib.unified_diff(
                self._old_text(rel, existed).splitlines(), self._new_text(rel).splitlines(),
                fromfile=f"before/{rel}", tofile=f"after/{rel}", lineterm=""))
            if lines:
                parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def summary(self) -> str:
        changed = self.changed_files()
        if not changed:
            return ""
        rows = "\n".join(f"  {rel}{'  (new file)' if new else ''}" for rel, new in changed)
        return (f"\nFiles changed in this run:\n{rows}\n"
                f"Undo all of it with:  python coding_agent_free.py --undo")


def undo_last_run() -> None:
    """Restore the files changed by the most recent run that has not been undone yet."""
    manifests = sorted(BACKUP_ROOT.glob("run-*/manifest.json")) if BACKUP_ROOT.is_dir() else []
    for mf in reversed(manifests):
        data = json.loads(mf.read_text(encoding="utf-8"))
        if data.get("undone"):
            continue
        run_dir, workspace = mf.parent, Path(data["workspace"])
        print(f"Latest run with changes: {run_dir.name}\nProject folder: {workspace}\n")
        for rel, existed in data["entries"].items():
            print(f"  {rel}  ->  " + ("put the original back" if existed else "delete (the agent created it)"))
        print("\nWARNING: this also throws away any edits you made to these files by hand since that run.")
        if input("\nUndo this run? [y/N] ").strip().lower() != "y":
            print("Nothing changed.")
            return
        for rel, existed in data["entries"].items():
            target = workspace / rel
            if existed:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(run_dir / "files" / rel, target)
            elif target.is_file():
                target.unlink()
        data["undone"] = True
        mf.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print("Done. The files are back to how they were before that run.")
        return
    print("Nothing to undo.")


class Agent:
    def __init__(self, workspace: Path, client, model: str, delay: float,
                 auto_approve_edits: bool, auto_approve_commands: bool, log=None,
                 backup=None, max_steps: int = MAX_STEPS):
        self.workspace = workspace.resolve()
        self.client = client
        self.model = model
        self.delay = delay
        self.auto_edits = auto_approve_edits
        self.auto_cmds = auto_approve_commands
        self.log = log or RunLog(False, "", "", "")
        self.backup = backup or BackupStore(workspace, enabled=False)
        self.max_steps = max_steps

    # ---- helpers ----
    def _safe_path(self, rel: str) -> Path:
        """Resolve a path and make sure it stays inside the workspace."""
        p = (self.workspace / rel).resolve()
        if p != self.workspace and self.workspace not in p.parents:
            raise ValueError(f"Path '{rel}' is outside the workspace.")
        return p

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        half = MAX_OUTPUT_CHARS // 2
        return text[:half] + "\n...[output truncated]...\n" + text[-half:]

    @staticmethod
    def _confirm(prompt: str) -> bool:
        return input(f"\n{prompt} [y/N] ").strip().lower() == "y"

    # ---- tools ----
    def list_dir(self, path: str) -> str:
        p = self._safe_path(path)
        if not p.is_dir():
            return f"Error: '{path}' is not a directory."
        skip = {".git", "node_modules", "__pycache__", ".venv", "venv"}
        entries = sorted(e for e in p.iterdir() if e.name not in skip)
        return "\n".join(f"{e.name}/" if e.is_dir() else e.name for e in entries) or "(empty)"

    def read_file(self, path: str, start_line: int = None, end_line: int = None) -> str:
        p = self._safe_path(path)
        if not p.is_file():
            return f"Error: '{path}' is not a file."
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            return "Error: file is not valid UTF-8 text."
        total = len(lines)
        first = max(1, int(start_line)) if start_line else 1
        last = min(total, int(end_line)) if end_line else total
        out, used = [], 0
        for n in range(first, last + 1):
            row = f"{n:4d}| {lines[n - 1]}"
            if used + len(row) > MAX_FILE_CHARS:
                out.append(f"[output cut off at line {n - 1} of {total}. "
                           f"Call read_file with start_line={n} to continue.]")
                break
            out.append(row)
            used += len(row) + 1
        header = f"(file has {total} lines; showing {first}-{last})\n"
        return header + "\n".join(out)

    def search(self, pattern: str) -> str:
        skip = {".git", "node_modules", "__pycache__", ".venv", "venv"}
        matches = []
        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if d not in skip]
            for name in files:
                fp = Path(root) / name
                try:
                    if fp.stat().st_size > 1_000_000:
                        continue
                    for n, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
                        if pattern in line:
                            matches.append(f"{fp.relative_to(self.workspace)}:{n}: {line.strip()[:200]}")
                            if len(matches) >= 100:
                                return "\n".join(matches) + "\n[stopped at 100 matches]"
                except (UnicodeDecodeError, OSError):
                    continue
        return "\n".join(matches) or "No matches."

    @staticmethod
    def _not_found_hint(text: str, old_str: str) -> str:
        """Help the model fix an edit that didn't match: show the real lines."""
        msg = "Error: old_str not found."
        first = next((l.strip() for l in old_str.splitlines() if l.strip()), "")
        if first:
            # try the whole line first, then just its last two words (e.g. "Get Started")
            for needle in (first, " ".join(first.split()[-2:])):
                hits = [(i, l) for i, l in enumerate(text.splitlines(), 1) if needle in l][:3]
                if hits:
                    break
            if hits:
                msg += " A line containing the start of your text exists at: " + "; ".join(
                    f"line {i}: {l!r}" for i, l in hits
                ) + ". Copy the exact whitespace and text."
                return msg
        return msg + " Re-read the file with read_file and copy the text exactly."

    def edit_file(self, path: str, old_str: str, new_str: str) -> str:
        p = self._safe_path(path)
        if not p.is_file():
            return f"Error: '{path}' does not exist."
        text = p.read_text(encoding="utf-8")
        count = text.count(old_str)
        if count == 0:
            return self._not_found_hint(text, old_str)
        if count > 1:
            return f"Error: old_str appears {count} times. Include more surrounding context."
        if not self.auto_edits and not self._confirm(f"Edit {path}?\n--- old\n{old_str}\n+++ new\n{new_str}\n"):
            return "Edit rejected by user."
        self.backup.before_change(p)
        p.write_text(text.replace(old_str, new_str, 1), encoding="utf-8")
        return f"Edited {path}."

    def append_file(self, path: str, content: str) -> str:
        p = self._safe_path(path)
        if not p.is_file():
            return f"Error: '{path}' does not exist. Use write_file to create a new file."
        if not self.auto_edits and not self._confirm(
                f"Append to {path}?\n--- text to add at the end\n{content}\n"):
            return "Append rejected by user."
        self.backup.before_change(p)
        existing = p.read_text(encoding="utf-8")
        joiner = "" if (not existing or existing.endswith("\n")) else "\n"
        tail = "" if content.endswith("\n") else "\n"
        p.write_text(existing + joiner + content + tail, encoding="utf-8")
        return f"Appended to {path}."

    def write_file(self, path: str, content: str) -> str:
        p = self._safe_path(path)
        action = "Overwrite" if p.exists() else "Create"
        if not self.auto_edits and not self._confirm(f"{action} {path} ({len(content)} chars)?"):
            return "Write rejected by user."
        self.backup.before_change(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {path}."

    def show_changes(self, path: str = None) -> str:
        if not self.backup.enabled:
            return "Backups are turned off, so changes cannot be shown."
        only = self.backup._rel(self._safe_path(path)) if path else None
        return self._truncate(self.backup.diff(only)) or "No files have been changed yet."

    def run_command(self, command: str) -> str:
        if not self.auto_cmds and not self._confirm(f"Run command: {command}"):
            return "Command rejected by user."
        try:
            result = subprocess.run(
                command, shell=True, cwd=self.workspace,
                capture_output=True, text=True, timeout=COMMAND_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {COMMAND_TIMEOUT}s."
        out = f"exit code: {result.returncode}\n"
        if result.stdout:
            out += f"--- stdout ---\n{result.stdout}\n"
        if result.stderr:
            out += f"--- stderr ---\n{result.stderr}\n"
        return self._truncate(out)

    def dispatch(self, name: str, args: dict) -> str:
        handlers = {
            "list_dir": self.list_dir,
            "read_file": self.read_file,
            "search": self.search,
            "edit_file": self.edit_file,
            "write_file": self.write_file,
            "append_file": self.append_file,
            "show_changes": self.show_changes,
            "run_command": self.run_command,
        }
        if name not in handlers:
            return f"Error: unknown tool '{name}'."
        try:
            return handlers[name](**args)
        except Exception as e:  # return errors to the model so it can recover
            return f"Error: {type(e).__name__}: {e}"

    def _chat(self, messages):
        """Call the model, waiting and retrying if the server is busy."""
        for attempt in range(len(RETRY_WAITS) + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=TOOLS,
                )
            except RETRYABLE as e:
                if attempt == len(RETRY_WAITS):
                    raise
                wait = RETRY_WAITS[attempt]
                print(f"\n[retry] Server busy ({type(e).__name__}). "
                      f"Waiting {wait}s... (try {attempt + 1}/{len(RETRY_WAITS)})")
                time.sleep(wait)

    # ---- the agent loop ----
    def run(self, task: str) -> None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        for step in range(1, self.max_steps + 1):
            if step > 1 and self.delay:
                time.sleep(self.delay)  # be gentle with free-tier rate limits

            response = self._chat(messages)
            msg = response.choices[0].message

            if msg.content and msg.content.strip():
                print(f"\n[agent] {msg.content}")
                self.log.write(f"\n[agent] {msg.content}")

            # Rebuild the assistant message by hand (keeps it clean for any provider)
            assistant = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                calls = []
                for tc in msg.tool_calls:
                    entry = {"id": tc.id, "type": "function",
                             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    extra = getattr(tc, "model_extra", None) or {}
                    if "extra_content" in extra:  # Gemini 3 "thought signatures" must be sent back
                        entry["extra_content"] = extra["extra_content"]
                    calls.append(entry)
                assistant["tool_calls"] = calls
            messages.append(assistant)

            if not msg.tool_calls:
                print(f"\nDone in {step} step(s).")
                self.log.write(f"\nDone in {step} step(s).")
                return

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    print(f"\n[tool] {name}({args})")
                    self.log.write(f"\n[tool] {name}({args})")
                    output = self.dispatch(name, args)
                except json.JSONDecodeError:
                    output = "Error: tool arguments were not valid JSON. Try again."
                print(f"[result] {output[:300]}{'...' if len(output) > 300 else ''}")
                self.log.write(f"[result] {output[:2000]}{'...(cut)' if len(output) > 2000 else ''}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

        print(f"\nStopped: reached the {self.max_steps}-step limit.")
        self.log.write(f"\nStopped: reached the {self.max_steps}-step limit.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Free coding agent (Gemini free tier or local Ollama)")
    parser.add_argument("task", nargs="?", help="What you want the agent to do")
    parser.add_argument("--list-models", action="store_true", help="Show the model names your key can use, then exit")
    parser.add_argument("--provider", choices=PROVIDERS.keys(), default="gemini")
    parser.add_argument("--model", help="Override the default model name")
    parser.add_argument("--workspace", default=".", help="Project directory")
    parser.add_argument("--delay", type=float, help="Seconds to wait between steps")
    parser.add_argument("--no-log", action="store_true", help="Do not write a log file for this run")
    parser.add_argument("--no-backup", action="store_true", help="Do not back up files before changing them")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help=f"Step limit (default {MAX_STEPS})")
    parser.add_argument("--undo", action="store_true", help="Undo the file changes of the last run, then exit")
    parser.add_argument("--auto-approve-edits", action="store_true")
    parser.add_argument("--auto-approve-commands", action="store_true")
    args = parser.parse_args()
    if args.undo:
        undo_last_run()
        return

    cfg = PROVIDERS[args.provider]
    api_key = "ollama"
    if cfg["key_env"]:
        api_key = os.environ.get(cfg["key_env"], "")
        if not api_key:
            sys.exit(f"Set the {cfg['key_env']} environment variable first.")

    workspace = Path(args.workspace)
    if not workspace.is_dir():
        sys.exit(f"Workspace '{workspace}' is not a directory.")

    client = OpenAI(base_url=cfg["base_url"], api_key=api_key, max_retries=2)
    if args.list_models:
        try:
            names = sorted(m.id.replace("models/", "") for m in client.models.list())
        except (openai.BadRequestError, openai.AuthenticationError, openai.PermissionDeniedError) as e:
            sys.exit(explain_rejection(e))
        print("Models available to your key:")
        for n in names:
            print("  " + n)
        print("\nUse one with:  --model NAME")
        return
    if not args.task:
        parser.error("Please give a task, e.g. \"Run the tests and fix all failures\"")
    delay = cfg["default_delay"] if args.delay is None else args.delay
    model = args.model or cfg["default_model"]
    print(f"Provider: {args.provider} | Model: {model}")

    log = RunLog(not args.no_log, model, workspace, args.task)
    if log.path:
        print(f"Log file: {log.path}")
    backup = BackupStore(workspace, enabled=not args.no_backup)
    agent = Agent(workspace, client, model, delay, args.auto_approve_edits, args.auto_approve_commands,
                  log, backup, args.max_steps)
    try:
        agent.run(args.task)
    except RETRYABLE as e:
        sys.exit(
            f"\nStopped: the model server kept failing ({type(e).__name__}).\n"
            "Try again in a few minutes, or use a different model, e.g. --model gemini-flash-lite-latest"
        )
    except (openai.BadRequestError, openai.AuthenticationError, openai.PermissionDeniedError) as e:
        sys.exit(explain_rejection(e))
    except openai.NotFoundError:
        sys.exit(
            f"\nStopped: the model '{model}' was not found for your key.\n"
            f"Run:  python coding_agent_free.py --list-models --provider {args.provider}\n"
            "and pick a name from the list with --model NAME"
        )
    except KeyboardInterrupt:
        log.write("\nStopped by the user (Ctrl+C).")
        sys.exit("\nStopped by you (Ctrl+C).")
    except Exception as e:
        log.write(f"\nCRASH: {e!r}")
        raise
    finally:
        report = backup.summary()
        if report:
            print(report)
            log.write(report)
        log.close()


if __name__ == "__main__":
    main()
