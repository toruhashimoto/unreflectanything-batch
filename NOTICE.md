# Third-party notices

This project (`unreflectanything-batch`) is licensed under the MIT License (see
[`LICENSE`](LICENSE)). The MIT license applies to **this project's own source code only**.

This project **invokes, but does not bundle or redistribute**, the third-party tools,
libraries, and models listed below. Each carries its own license, and users are
responsible for complying with them — and with the terms of any model weights they
download.

| Component | Role | License |
|---|---|---|
| [UnReflectAnything](https://github.com/alberto-rota/UnReflectAnything) | Reflection-removal model (Python API/CLI) | MIT |
| └ DINOv3 backbone (used by UnReflectAnything) | Frozen vision encoder | **Meta DINOv3 License** — not open source; "Built with DINOv3" attribution + use restrictions |
| [COLMAP](https://github.com/colmap/colmap) | Structure-from-Motion (A/B harnesses) | BSD |
| [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) | 3D Gaussian Splatting trainer, invoked as an external process by `tools/ab_3dgs.py` | **GPL-3.0** |
| [PyTorch](https://pytorch.org) | Inference runtime | BSD-style |
| [Pillow](https://python-pillow.org) | Image I/O | HPND/MIT-CMU |
| [piexif](https://github.com/hMatoba/Piexif) | EXIF transplant | MIT |
| [Streamlit](https://streamlit.io) | Optional GUI | Apache-2.0 |

Notes:
- **LichtFeld Studio is GPL-3.0.** This project does not link to or bundle it; it is
  invoked as a separate external process (subprocess) only. Running it does not affect
  the license of this project's source.
- **Model weights** for UnReflectAnything are downloaded by the user at runtime
  (`unreflectanything download --weights`) and are governed by the upstream model's and
  DINOv3's terms — review them before any redistribution or commercial use.
