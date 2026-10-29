# S1 — Repo skeleton

- **Tool:** Claude Code (CLI), model Claude Opus 5.5
- **Date:** 2026-10-29

## Original prompt
> ok lets start day 1  *(executing prompt S1 from plan.md: create the repo skeleton, common/utils.py, docs stubs, requirements, README with AI tools register; create venv)*

## AI response (excerpt)
Created a separate git repo in `ot_lab8/` (the home folder is itself a git repo, so the assignment needs its own),
10 track folders with README stubs, `common/utils.py` (set_seed, timer, save_results, save_plot), prompts/A–J,
docs stubs, `requirements.txt`, `pytest.ini`, `.gitignore`, README.

## Problems / revisions
- First attempt used a zsh variable holding the folder list; zsh does not split unquoted variables, so it created **one
  folder whose name was the whole list**. Fixed by removing it and re-running the loop in bash. (Failure log #1.)

## Verification note
`ls` shows the 10 track folders and matching `results/` folders; README table lists all tracks.
