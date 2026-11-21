"""Pipeline steps: one module per step, discovered by :mod:`glassesdata_pipeline.runner` in the
fixed order ingest -> hands2d -> depth -> hand3d -> slam -> imu -> world -> arms -> objects ->
scene -> retarget -> physics -> viz. A missing module is reported as "not implemented yet".

A step module declares::

    NAME = "hand3d"                        # equals the module name
    VERSION = 1                            # bump when the output changes for the same inputs
    REQUIRES = ["hands2d", "depth"]        # steps whose outputs this one reads
    INPUTS = ["derived/hands2d.npz", "derived/depth/index.npz", "derived/camera.json"]
    #   paths relative to the RECORDING folder (raw files like "video.mp4", derived ones as
    #   "derived/..."); globs allowed; missing files are recorded as null. May also be a function
    #   cfg -> list when the inputs depend on the configuration (e.g. the backend).
    OUTPUTS = ["hand3d.npz"]               # paths relative to derived/ (PIPELINE.md section 5)
    CONFIG_SECTIONS = ["hand3d"]           # config sections whose values are the step's params

    def run(ep: Episode, cfg: Config) -> dict:
        ...                                # cfg is the full merged config: cfg["hand3d"]["..."]
        ep.save_npz("hand3d", ep.step_meta(backend="..."), joints_cam=..., valid=...)
        return {"frames": T}               # short summary for the log

Rules: no heavy imports at module level (import torch etc. inside ``run`` or a backend and raise
``errors.MissingDependency`` when absent); write outputs only through ``ep.save_npz`` /
``ep.save_json`` (or into ``ep.derived``) and always include ``ep.step_meta()`` so the runner can
cache the result.
"""
