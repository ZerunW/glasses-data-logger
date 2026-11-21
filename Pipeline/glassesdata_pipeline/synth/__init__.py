"""Synthetic recordings with ground truth (PIPELINE.md section 8).

``generate(out_dir, seconds=None, seed=None, size=None, draw_hands=None, config=None, overwrite=False)``
renders a textured room / table / boxes scene by analytic ray casting (numpy), moves a right hand
that grasps and moves a box, and writes a recording folder in the receiver's format plus ``gt/``.
Modules: ``scene`` (ray casting), ``textures``, ``motion`` (the choreography), ``imu``, ``writer``.
"""

from .generate import generate, parse_size

__all__ = ["generate", "parse_size"]
