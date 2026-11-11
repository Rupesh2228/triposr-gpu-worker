from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .config import Settings
from .gpu import require_cuda


class TripoSRModel:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device: str | None = None
        self.model: Any = None
        self._torch: Any = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        self.device = require_cuda()
        self._torch = self._import_torch()
        from tsr.system import TSR

        self.model = TSR.from_pretrained(
            self.settings.model_path,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self.model.renderer.set_chunk_size(self.settings.chunk_size)
        self.model.to(self.device)

    def generate(self, image: Any, output_path: Path) -> None:
        if self.model is None or self.device is None or self._torch is None:
            raise RuntimeError("TripoSR model is not loaded")
        with self._torch.inference_mode():
            scene_codes = self.model([image], device=self.device)
        meshes = self.model.extract_mesh(
            scene_codes,
            True,
            resolution=self.settings.mc_resolution,
        )
        if not meshes:
            raise RuntimeError("TripoSR returned no mesh")
        meshes[0].export(str(output_path))
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("TripoSR did not produce a valid GLB")

    def _import_torch(self) -> Any:
        import torch

        sys.path.insert(0, str(self.settings.triposr_repo_dir))
        return torch

