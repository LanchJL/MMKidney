from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import h5py
import numpy as np
import torch


@dataclass
class H5ReadResult:
    features: torch.Tensor
    coords: Optional[torch.Tensor]
    n_instances: int
    feat_dim: int


class H5BagReader:
    def __init__(
        self,
        feature_key_candidates: Optional[Iterable[str]] = None,
        coord_key_candidates: Optional[Iterable[str]] = None,
        cache_size: int = 64,
    ):
        self.feature_key_candidates = list(feature_key_candidates or ["features", "feats", "embeddings", "x", "patches"])
        self.coord_key_candidates = list(coord_key_candidates or ["coords", "coordinates", "xy"])
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, H5ReadResult] = OrderedDict()

    def _pick_feature_key(self, f: h5py.File) -> str:
        keys = list(f.keys())
        for k in self.feature_key_candidates:
            if k in f and getattr(f[k], "ndim", None) == 2:
                return k

        # fallback: any 2D dataset
        two_d = [k for k in keys if getattr(f[k], "ndim", None) == 2]
        if not two_d:
            raise KeyError(f"No 2D feature dataset in {f.filename}. Available keys: {keys}")

        # prefer feature-like names
        score = []
        for k in two_d:
            lk = k.lower()
            s = int("feat" in lk or "embed" in lk or "patch" in lk or lk == "x")
            score.append((s, k))
        score.sort(reverse=True)
        return score[0][1]

    def _pick_coord_key(self, f: h5py.File) -> Optional[str]:
        for k in self.coord_key_candidates:
            if k in f and getattr(f[k], "ndim", None) == 2:
                return k
        return None

    def _to_tensor_2d(self, arr: np.ndarray, name: str) -> torch.Tensor:
        t = torch.as_tensor(arr)
        if t.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape={tuple(t.shape)}")
        return t

    def _read_uncached(self, path: str) -> H5ReadResult:
        with h5py.File(path, "r") as f:
            fk = self._pick_feature_key(f)
            features = self._to_tensor_2d(f[fk][:], fk).float()

            ck = self._pick_coord_key(f)
            coords = None
            if ck is not None:
                c = self._to_tensor_2d(f[ck][:], ck)
                coords = c.long()

        return H5ReadResult(
            features=features,
            coords=coords,
            n_instances=int(features.shape[0]),
            feat_dim=int(features.shape[1]),
        )

    def read(self, path: str) -> Dict[str, object]:
        if self.cache_size > 0 and path in self._cache:
            item = self._cache.pop(path)
            self._cache[path] = item
            return {
                "features": item.features,
                "coords": item.coords,
                "n_instances": item.n_instances,
                "feat_dim": item.feat_dim,
            }

        item = self._read_uncached(path)
        if self.cache_size > 0:
            self._cache[path] = item
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return {
            "features": item.features,
            "coords": item.coords,
            "n_instances": item.n_instances,
            "feat_dim": item.feat_dim,
        }
