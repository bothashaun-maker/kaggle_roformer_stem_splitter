# Stem Studio engineering

- This repository owns the stem-splitting workflow. Publish checked source and
  generated notebooks here. Do not modify working Music/Movies workflows.
- Preserve original WAV bytes, exact output rate/frame count, float WAV exports,
  a normal visible ComfyUI canvas with Cloudflare. Keep one startup cell.
- Model weights are a one-time, verified private Kaggle Dataset. No silent
  replacement weight downloads in rendering and no silent CPU inference fallback.
- Dataset creation explicitly requests private visibility. Never update an
  existing Dataset. Model files are saved before submission and reused on retry.
- Build with `python tools/build_notebooks.py`. Core checks:
  `python -m unittest discover -s tests -v`.
- Do not publish user audio, projects, generated stems, tokens or local research
  weights. Distinguish local CPU/browser checks from real T4/Kaggle validation.
- No subagents unless the user asks for delegation.
