# Workspace Commands

`quanttradeai init` creates a QuantTradeAI workspace from a supported project template. Use it when starting a new project directory or when adding QuantTradeAI workspace files to the current directory.

## When To Use

Use this command before editing project YAML by hand. It gives humans and coding agents the same baseline files:

- a canonical project config at `config/project.yaml`
- a disposable uv project pinned to the local QuantTradeAI checkout
- minimal agent-facing context files
- workspace metadata under `.quanttradeai/`

For installation and first-run setup, see [`docs/getting-started.md`](../getting-started.md).

## Syntax

```bash
quanttradeai init [PROJECT_DIR]
```

```bash
quanttradeai init
quanttradeai init my-strategy-lab
quanttradeai init --template research
quanttradeai init my-agent-lab --template rule-agent
quanttradeai init --force
```

## Options

| Option | Default | Required | Description |
|---|---:|:---:|---|
| `[PROJECT_DIR]` | current directory | No | Workspace directory to initialize. If omitted, QuantTradeAI initializes the current working directory. |
| `--template TEXT` | `strategy-lab` | No | Project template to write into `config/project.yaml`. |
| `--force` | `false` | No | Overwrite generated files that are protected by the initializer. |

## Supported Templates

The current CLI supports these template names:

| Template | Use it for |
|---|---|
| `strategy-lab` | A combined research and agent workspace with a rule agent and a simple sweep. |
| `research` | A research-only starting point with research enabled and no agents. |
| `rule-agent` | A rule-based paper agent starting point. |
| `model-agent` | A model-backed paper agent starting point. |
| `llm-agent` | An LLM-backed paper agent starting point with a prompt file. |
| `hybrid` | A hybrid agent starting point that combines model signals and an LLM. |

## Reads

`init` reads package templates embedded in `quanttradeai/templates/workspace_context/`.

It does not read an existing project config to infer project name, symbols, or agent names.

## Writes

Every template writes:

```text
config/project.yaml
pyproject.toml
.python-version
.env.example
.gitignore
AGENTS.md
CLAUDE.md
.quanttradeai/workspace.yaml
```

Some templates also write template-specific assets:

- `llm-agent`: `prompts/breakout.md`
- `hybrid`: `prompts/hybrid_swing.md` and a placeholder promoted model directory
- `model-agent`: a placeholder promoted model directory

## Expected Outcome

On success, the command prints the initialized workspace path and the project template path:

```text
Initialized QuantTradeAI workspace at <workspace>
Wrote <template> template to <workspace>/config/project.yaml
```

After that, edit `config/project.yaml`, then run:

```bash
uv sync
uv run quanttradeai validate -c config/project.yaml
```

## Common Mistakes

| Mistake | What happens |
|---|---|
| Initializing over an existing generated project config or `pyproject.toml` without `--force` | The command refuses to overwrite protected generated files. |
| Assuming the folder name changes `project.name` | The template controls `project.name`; the CLI does not rewrite it from `PROJECT_DIR`. |
| Choosing a template name outside the supported list | The command fails with the valid template names. |

## Related Docs

- [`docs/config/`](../config/)
- [`docs/artifacts.md`](../artifacts.md)
