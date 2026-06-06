---
name: Bug report
about: Report a problem with UnReflect Batch (the wrapper)
title: "[bug] "
labels: bug
---

**Describe the bug**
A clear description of what went wrong.

**To reproduce**
- Command or GUI action:
- Input (format, resolution, number of images):
- Flags used (e.g. `--mask-composite`, `--device`):

**Expected vs actual**

**Environment**
- OS:
- GPU + driver (or CPU-only):
- Python version:
- torch version (`python -c "import torch;print(torch.__version__)"`, e.g. `2.9.1+cu128`):
- unreflectanything version:

**Logs**
Attach the console output and, if present, `<output>/logs/errors.csv` and
`<output>/logs/run_summary.json`.

**Pre-flight checklist**
- [ ] (GPU users) torch came from the CUDA index and `torch.cuda.is_available()` is True
- [ ] `transformers` is the pinned commit from `requirements.txt` (no `Missing key(s) ... dinov3`)
- [ ] I ran `unreflectanything download --weights` (or `--download-weights`)
