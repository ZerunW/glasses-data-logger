"""Offline processing pipeline for first-person recordings from Ray-Ban Meta glasses.

The data formats and step contracts are defined in ``Docs/PIPELINE.md``. Entry points:

* ``python -m glassesdata_pipeline run REC`` - run the processing steps on a recording folder
* ``python -m glassesdata_pipeline synth OUT`` - write a synthetic recording with ground truth
* ``python -m glassesdata_pipeline eval REC`` - compare ``derived/`` with ``gt/``

Library entry points: :class:`glassesdata_pipeline.episode.Episode`,
:func:`glassesdata_pipeline.config.load_config`, :mod:`glassesdata_pipeline.geometry`,
:mod:`glassesdata_pipeline.skeleton`, :mod:`glassesdata_pipeline.runner`.

Heavy packages (torch, mujoco, open3d, rerun, ...) are never imported at module import time.

Importing the package points the caches and temporary files of every library into the
repository's gitignored ``local/`` folder (:func:`glassesdata_pipeline.paths.use_local_dirs`;
variables that are already set win), so nothing is written outside the project folder.
"""

from .paths import use_local_dirs as _use_local_dirs

__version__ = "0.1.0"

_use_local_dirs()
