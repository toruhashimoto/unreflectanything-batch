# Examples

![before / after / diff](demo_before_after.jpg)

*Left → right: **original** (a synthetic scene with bright specular glare) · **UnReflect**
(`--mask-composite`) · **diff heatmap**. Only the blown-highlight regions are altered;
the rest of the image is preserved, and the heatmap localizes exactly what changed.*

This demo uses a **synthetic** scene on purpose, so nothing private is published —
results on real photos look better.

## Reproduce

From the repo root, with the project venv and the model weights downloaded
(`unreflectanything download --weights`):

```bash
python examples/make_demo.py
```

It synthesizes `assets/demo_input.png`, runs this tool with
`--mask-composite --make-preview --heatmap`, and writes `demo_before_after.jpg`.

## Try it on your own photos

```powershell
# CLI (batch a folder; originals are never modified)
python main.py --input "D:\photo_input" --output "D:\photo_unreflect" --recursive --make-preview --mask-composite

# GUI
run_app.bat
```

## Does cleaning actually help your 3DGS / photogrammetry?

Use the A/B harnesses (see the main [README](../README.md#9b-ab-evaluation-pipelines-optional)):

```powershell
# Structure-from-Motion A/B
python tools\ab_colmap.py --work ab_work --set original "D:\in" --set cleaned "D:\out"

# 3D Gaussian Splatting A/B (COLMAP -> LichtFeld Studio -> same-view figures)
python tools\ab_3dgs.py --work ab3dgs --set original "D:\in" --set cleaned "D:\out" --shared-poses original
```
