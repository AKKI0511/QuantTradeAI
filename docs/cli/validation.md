# Validation

`quanttradeai validate` checks the canonical project YAML and emits resolved config artifacts. It is the fastest way to confirm that a project file can be consumed by research, agent, promotion, and deployment workflows.

## When To Use

Run validation:

- after creating a workspace
- after editing `config/project.yaml`
- before launching experiments or deployment bundle generation
- when a coding agent needs a normalized view of the project config

Validation does not run experiments, train models, execute agents, or promote anything.

## Syntax

```bash
quanttradeai validate [OPTIONS]
```

```bash
quanttradeai validate
quanttradeai validate -c config/project.yaml
quanttradeai validate --config path/to/project.yaml
```

## Options

| Option | Default | Required | Description |
|---|---:|:---:|---|
| `-c, --config TEXT` | `config/project.yaml` | No | Path to the project config YAML. |

## What Validation Checks

The command calls `validate_project_config()` and checks the same config shape used by runtime commands.

It verifies:

- required top-level project sections: `project`, `profiles`, `data`, `features`, `research`, `agents`, and `deployment`
- schema constraints for data windows, feature definitions, research settings, agents, sweeps, deployment, risk, and position manager settings
- research promotion targets, including nonblank names, symbols present in `data.symbols`, project-relative paths under `models/`, and uniqueness of target names and paths
- agent requirements by kind:
  - `rule` agents require a valid `rule` block
  - `model` agents require a model path that exists
  - `llm` and `hybrid` agents require prompt files that exist
- agent context references, including configured features, model signals, news, and notes
- paper/live prerequisites such as streaming configuration when an agent is configured for `paper` or `live`
- live prerequisites such as risk and position-manager sections when live runtime config is required
- sweep definitions and each expanded sweep variant
- data/provider environment warnings where applicable

## Reads

```text
config/project.yaml
```

Validation can also read files referenced by the project config, such as:

- prompt files for LLM or hybrid agents
- model artifacts for model agents and hybrid model signal sources
- notes files when an agent enables notes context

## Writes / Artifacts

By default, validation writes a timestamped directory under:

```text
reports/config_validation/<timestamp>/
```

Artifacts include:

| Artifact | Purpose |
|---|---|
| `resolved_project_config.yaml` | Schema-normalized project config with defaults and accepted fields preserved. |
| `summary.json` | Compact project summary used by humans and coding agents. |

The CLI also prints a readable summary followed by the full JSON validation result.

## Expected Output

The human-readable section starts with:

```text
Resolved project config summary:
- project: <name> (<profile>)
- data: symbols=<count>, timeframe=<timeframe>, range=<start>..<end>
- features: definitions=<count>, research_enabled=<true|false>, agents=<count>, sweeps=<count>
```

If paper replay is configured, the summary includes the paper source and replay window.

Warnings are printed to stderr and included in the final JSON payload. A failed validation exits with code `1`.

## Examples

Validate the default project file:

```bash
quanttradeai validate
```

Validate a generated sweep variant or copied config:

```bash
quanttradeai validate -c runs/agent/batches/example/variants/001_variant/project.yaml
```

## Common Mistakes

| Mistake | Why it matters |
|---|---|
| Validating the wrong config path | The CLI defaults to `config/project.yaml`; pass `-c` when working from another file. |
| Continuing after validation failure | Runtime commands call validation too, but fixing the config first gives clearer failures. |
| Assuming validation runs experiments | Validation writes resolved config artifacts only; it does not train, backtest, or execute agents. |
| Ignoring warnings | Warnings are included because they may affect paper/live execution or reproducibility. |

## Related Docs

- [`docs/config/`](../config/)
- [`docs/artifacts.md`](../artifacts.md)
