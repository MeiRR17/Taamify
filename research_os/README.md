# Machine-readable Research OS catalog

This directory is **configuration / planning tools only**. It contains no raw audio, models or credentials and has no access to the local active Taamify service.

- `config/pipelines.json`: canonical P00–P12 list and hardware assumption; research and product dependencies.
- `config/augmentations.json`: canonical A00–A08 registry.
- `plan.py`: standard-library CLI to list pipeline gates, compute planned fit-epochs and optionally read real measured `minutes_per_epoch` from a JSON benchmark. Example:

```bash
python research_os/plan.py
python research_os/plan.py --measurements /path/to/local/benchmark.json
```

Benchmark JSON schema (no actual measurement has been taken here):
```json
{
  "P00": {"minutes_per_epoch": 2.5, "setup_hours": 1.0},
  "P01": {"minutes_per_epoch": 0.1, "setup_hours": 0.5}
}
```

`plan.py` intentionally does **not** schedule training or perform any network I/O. The agent should read `../docs/research-os/AGENT_HANDOFF.md` before implementing live integration. See `../docs/research-os/BRANCH_SYNC.md` to safely apply these files to the current local checkout.
