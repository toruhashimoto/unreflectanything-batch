"""UnReflect Batch — wrapper app around the `unreflectanything` package for
3D Gaussian Splatting / photogrammetry image pre-processing.

The package is intentionally split so that the pure-Python utility modules
(`image_io`, `metrics`, `preview`, `logger`) import without torch /
unreflectanything, keeping unit tests fast and dependency-free. Heavy imports
(torch, unreflectanything) are done lazily inside `unreflect_batch`.
"""

__version__ = "0.1.1"
