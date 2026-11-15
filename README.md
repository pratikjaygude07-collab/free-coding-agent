# Free Coding Agent

A small command-line coding agent in **one Python file**. You give it a task
("fix the failing tests", "add a method that ..."), and it explores your project,
edits files, runs commands, and checks its own work. It uses a **free** model:

- **Gemini free tier** (needs a free key from [Google AI Studio](https://aistudio.google.com)), or
- **Ollama** (a model running on your own computer). *Experimental: not tested live.*

It was built step by step on Windows with the Gemini free tier, and tested on small
Python and Node.js projects.

## Features

- Agent loop with tools: `list_dir`, `read_file` (line ranges for long files), `search`,
  `edit_file`, `write_file`, `append_file`, `show_changes`, `run_command`
- **Asks for your approval (y/N)** before every edit and every command
- File tools are limited to the project folder you pass with `--workspace`
- **Backups + undo:** originals are saved before each change; `--undo` restores the last run
- **Run logs:** every run is saved as a text file in `agent_logs/`
- Retries automatically when the free API is busy; clear messages for bad keys or model names
- The agent reviews its own diff (`show_changes`) before finishing

## Quick start (Windows PowerShell)

```powershell
python -m venv venv
venv\Scripts\activate
pip install openai
$env:GEMINI_API_KEY="your_free_key"
python coding_agent_free.py --list-models --provider gemini
```

Pick a model from the list (a "flash" or "flash-lite" model), then:

```powershell
python coding_agent_free.py "Run the tests and fix every failure. Never edit the tests." `
  --provider gemini --model MODEL_NAME --workspace path\to\your\project
```

On macOS/Linux use `source venv/bin/activate` and `export GEMINI_API_KEY=...`.

**Work on a copy of your project first.**

## Options

| Option | Meaning |
|---|---|
| `--provider gemini\|ollama` | Which free backend to use (default `gemini`) |
| `--model NAME` | Model name (use `--list-models` to see what your key can use) |
| `--workspace PATH` | The project folder the agent may read and change |
| `--list-models` | Show the models your key can use, then exit |
| `--max-steps N` | Step limit (default 25) |
| `--undo` | Undo the file changes of the last run, then exit |
| `--no-log` / `--no-backup` | Turn off the run log / the backups |
| `--auto-approve-edits` / `--auto-approve-commands` | Skip the y/N questions (risky) |

## Safety: please read

- **Not a sandbox.** Approved commands run on your computer with your permissions.
  Your y/N approval is the only guard. Read every diff and command.
- **Undo only covers file edits made through the agent's own tools,** not changes made by shell commands.
- **Free tiers may use your data.** Google's free Gemini tier can use submitted content to improve its
  products. Don't use it on private or confidential code.
- **Never commit your API key.** It is read from the `GEMINI_API_KEY` environment variable.
- Logs and backups contain your project's code. They are listed in `.gitignore`.
- Small free models make mistakes. Treat their reports as leads and verify them.

## Limitations

- No memory between runs: give each task full context.
- Free-tier rate limits and busy servers can slow it down.
- Tested only on Windows with Gemini; the Ollama path is untested.
- No sandboxing, no parallel agents, no web access.

## How it works

The script sends your task and the tool list to the model. If the model asks to use a tool,
the script runs it (after your approval if it changes anything), sends the result back,
and repeats until the model answers without a tool call or the step limit is reached.

## License

Add a license of your choice (for example MIT) before sharing widely.
