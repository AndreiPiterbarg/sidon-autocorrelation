# gpupod

Self-contained Python package for orchestrating remote GPU pods
(Prime Intellect / similar). Importable as `gpupod`; entry point
`python -m gpupod` via `__main__.py`.

## Modules

- `cli.py` — command-line interface (start/stop/status/sync).
- `config.py` — pod / project config loader.
- `session.py` — long-running session state (pod ID, ssh, lifecycle).
- `pod_manager.py` — provider API wrapper (create/destroy/list pods).
- `remote.py` — remote command exec, file ops over ssh.
- `sync.py` — bidirectional code/data sync (rsync wrapper).
- `budget.py` — per-session $ budget tracker, hard stop on overrun.
- `.session.json` — runtime state of the most recent session.

Do not flatten this folder — internal imports are relative.
