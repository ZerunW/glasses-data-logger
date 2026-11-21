"""Model wrappers used by the steps (mediapipe_hands, wilor, hamer, depth_anything, sam2, gt, ...).

Each backend imports its heavy dependencies lazily and raises ``errors.MissingDependency`` when
they are missing. ``gt`` serves synthetic recordings from ``gt/`` with configurable noise.
"""
