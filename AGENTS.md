# Taamify — repository agent entrypoint

Scope: this repository is the AI cantillation-reading tutor for the Jerusalem-Sephardi tradition. Existing `CLAUDE.md` and `PLAN_STAGE_A.md` are required engineering context; do not replace or contradict them.

## When the user says "תמשיך" / "continue" about Taamify

Follow **`docs/research-os/CONTINUE_AGENT_PROMPT_HE.md`** and **`docs/research-os/MASTER_COMPLETION_PLAN_HE.md`**. Inspect **`docs/research-os/CONTINUATION_STATE.md`** for current evidence and blockers. These are agent work instructions, not a claim that research has been completed.

First reconcile local and remote changes before acting. Other agents may have private local research not yet on GitHub. Review GitHub Issue #4 and draft PR #5, then the local worktree. Do not overwrite or redo already verified work. Take the next smallest safe issue-sized task that passes its dependencies, execute it, test it, update the ledger and report next steps with actual artifacts.

**A simple "תמשיך" is not permission** to merge/deploy, modify live thresholds or checkpoints, upload any voices or minors' information, force push, spend paid compute, run large training, recruit people or publish an academic paper. Request fresh explicit approval for those actions.

Scientific guardrails: Stage A 85.7% is a one-reader result, 2.1% false accusation is above the 2% product target, and 92% error catch concerns proxy errors, not genuine student errors. No speaker-invariant generalization or product readiness claim without distinct consented readers and real human mistakes. Preserve model/checkpoint/calibration/manifest SHA and split integrity. Keep private audio and records out of Git.

If the master-plan documents are absent from the checked-out branch, inspect branch `docs/agent-continue-masterplan-20261029` via `git show` after `git fetch`; do not merge any branch merely to read instructions.

Primary links: https://github.com/MeiRR17/Taamify/issues/4 and https://github.com/MeiRR17/Taamify/pull/5.
