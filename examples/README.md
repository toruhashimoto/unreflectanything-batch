# Examples

![before / after / diff](demo_before_after.jpg)

*Left → right: **original** (a synthetic scene with bright specular glare) · **cleaned**
(`clean` mode, `--mask-composite`) · **diff heatmap**.*

> **This shows the experimental `clean` mode.** The **recommended** ReflectMask workflow
> keeps your originals untouched and emits a tight **RealityScan alignment mask** instead —
> `python main.py reflectmask -i <in> -o <out>` (or `diagnostic` for a 6-panel inspection
> sheet of original / cleaned / heatmap / candidate / final mask / overlay). See the main
> [README](../README.md).

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
# ReflectMask — RealityScan alignment masks (recommended; originals never modified)
python main.py reflectmask --input "D:\photo_input" --output "D:\rs_reflectmask" --recursive

# Cleaned-image export (experimental)
python main.py clean --input "D:\photo_input" --output "D:\photo_unreflect" --recursive --mask-composite

# GUI
run_app.bat
```

## Does masking / cleaning actually help your RealityScan / 3DGS?

Build all four comparison variants (original / reflectmask / luma / cleaned) in one go,
then import each into RealityScan and compare:

```powershell
python tools\make_ab_sets.py -i "D:\photo_input" -o "D:\ab_work" --recursive
```

Or use the COLMAP / 3DGS harnesses (see the main [README](../README.md#9b-ab-evaluation-pipelines-optional)):

```powershell
# Structure-from-Motion A/B
python tools\ab_colmap.py --work ab_work --set original "D:\in" --set cleaned "D:\out"

# 3D Gaussian Splatting A/B (COLMAP -> LichtFeld Studio -> same-view figures)
python tools\ab_3dgs.py --work ab3dgs --set original "D:\in" --set cleaned "D:\out" --shared-poses original
```
