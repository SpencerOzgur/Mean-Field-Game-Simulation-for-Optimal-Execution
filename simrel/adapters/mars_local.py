"""
Run Microsoft MarS without its Ray/CUDA serving stack.

MarS officially ships as a Ray Serve deployment that requires Linux, Docker
and a GPU. The released order models are small (2.5M to 10M parameters), so
this module loads one in-process on CPU and exposes the same
`get_prediction(state)` call the MarS BackgroundAgent expects. Nothing in the
MarS repository is modified.

Requires a MarS checkout on sys.path (see setup_mars()).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

DEFAULT_REPO = "Don-Don/mars-order-10m"
SEQ_LEN, TOKEN_DIM = 1024, 15


def setup_mars(mars_root: str, data_dir: str) -> None:
    """Put a MarS checkout on sys.path and point its config at the converter directory."""
    os.environ["DIRECTORY__INPUT_ROOT_DIR"] = str(data_dir)
    if mars_root not in sys.path:
        sys.path.insert(0, str(mars_root))
    # MarS imports `ray` in its HTTP client module, which we never call; a stub avoids the dependency.
    if "ray" not in sys.modules:
        import types

        ray = types.ModuleType("ray")
        ray.remote = lambda f: f
        ray.is_initialized = lambda: True
        sys.modules["ray"] = ray


def download_assets(data_dir: str, repo_id: str = DEFAULT_REPO) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id="Don-Don/mars-order-assets", repo_type="dataset", local_dir=data_dir, allow_patterns=["converters/*"])
    snapshot_download(repo_id=repo_id)


class LocalModelClient:
    """In-process replacement for market_simulation.rollout.model_client.ModelClient."""

    def __init__(self, repo_id: str = DEFAULT_REPO, temperature: float = 1.0, seed: int = 0) -> None:
        import torch
        from market_simulation.models.order_model import OrderModel

        torch.set_num_threads(1)
        self.torch = torch
        self.model = OrderModel.from_pretrained(repo_id).eval()
        self.temperature = temperature
        self.generator = torch.Generator().manual_seed(seed)
        self.n_calls = 0

    def __deepcopy__(self, memo: dict) -> "LocalModelClient":
        """Copies share the (read-only) network but get their own sampling stream, so a branched episode is reproducible."""
        new = object.__new__(LocalModelClient)
        new.torch, new.model, new.temperature, new.n_calls = self.torch, self.model, self.temperature, self.n_calls
        new.generator = self.torch.Generator()
        new.generator.set_state(self.generator.get_state())
        return new

    def get_prediction(self, state: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(np.asarray(state, dtype=np.int32)).reshape((1, SEQ_LEN, TOKEN_DIM))
        with torch.no_grad():
            logits = self.model(x)[:, -1, :] / self.temperature
            idx = torch.multinomial(torch.softmax(logits, dim=1), 1, replacement=True, generator=self.generator)
        self.n_calls += 1
        return idx.int().reshape(-1).numpy()
