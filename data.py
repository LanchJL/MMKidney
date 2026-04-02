import os
from typing import Dict, List, Optional
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


def _read_h5_any(h5_path: str, candidate_keys=None) -> torch.Tensor:
    candidate_keys = candidate_keys or ["patches", "features", "feats", "x"]
    with h5py.File(h5_path, "r") as f:
        data = None
        for k in candidate_keys:
            if k in f:
                data = f[k][:]
                break
        if data is None:
            raise KeyError(
                f"HDF5 missing keys {candidate_keys} in {h5_path}. Available keys = {list(f.keys())}"
            )
    t = torch.tensor(data, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    elif t.ndim > 2:
        t = t.reshape(t.shape[0], -1)
    return t


def _read_h5_vector(h5_path: str) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        if "features" not in f:
            raise KeyError(f"HDF5 missing 'features' in {h5_path}. Available keys = {list(f.keys())}")
        data = f["features"][:]
    t = torch.tensor(data, dtype=torch.float32)
    if t.ndim == 2:
        t = t.mean(dim=0)
    return t


def _get_extra(rec: Dict) -> Optional[torch.Tensor]:
    extra = rec.get("extra", None)
    if extra is None:
        return None
    return torch.tensor(extra, dtype=torch.float32)


class SlideVectorDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        feat = _read_h5_vector(rec["h5"])
        labels = rec["labels"]
        return feat, labels, rec["id"], _get_extra(rec)


class SlidePatchSetDataset(Dataset):
    def __init__(self, samples: List[Dict], max_patches: Optional[int] = None):
        self.samples = samples
        self.max_patches = max_patches

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        patches = _read_h5_any(rec["h5"])
        if self.max_patches is not None and patches.shape[0] > self.max_patches:
            perm = torch.randperm(patches.shape[0])[: self.max_patches]
            patches = patches[perm]
        labels = rec["labels"]
        return patches, labels, rec["id"], _get_extra(rec)


class SlideMultiStainPatchSetDataset(Dataset):
    def __init__(self, samples: List[Dict], max_patches_per_stain: Optional[int] = None):
        self.samples = samples
        self.max_patches = max_patches_per_stain

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        if "h5s" not in rec or not isinstance(rec["h5s"], dict) or len(rec["h5s"]) == 0:
            raise ValueError(f"Sample {rec.get('id','?')} missing 'h5s' dict for multi-stain.")

        x_dict = {}
        for stain, p in rec["h5s"].items():
            if not os.path.exists(p):
                continue
            t = _read_h5_any(p)
            if self.max_patches is not None and t.shape[0] > self.max_patches:
                perm = torch.randperm(t.shape[0])[: self.max_patches]
                t = t[perm]
            x_dict[stain] = t

        if len(x_dict) == 0:
            raise ValueError(f"All stain files missing for sample {rec.get('id','?')}")

        labels = rec["labels"]
        return x_dict, labels, rec["id"], _get_extra(rec)


def _stack_labels(labels_list: List[Dict]) -> Dict[str, torch.Tensor]:
    out = {}
    for lv in labels_list[0].keys():
        out[lv] = torch.stack([torch.tensor(lab[lv], dtype=torch.float32) for lab in labels_list], 0)
    return out


def collate_vector(batch):
    feats, labels_list, ids, extras = zip(*batch)
    feats = torch.stack(feats, 0)
    labels = _stack_labels(list(labels_list))
    extra_t = None
    if all(e is not None for e in extras):
        extra_t = torch.stack(list(extras), 0)
    return feats, labels, list(ids), extra_t


def collate_varlen(batch):
    xs, labels_list, ids, extras = [], [], [], []
    for patches, labels, sid, extra in batch:
        xs.append(patches)
        labels_list.append(labels)
        ids.append(sid)
        extras.append(extra)
    labels = _stack_labels(labels_list)
    extra_t = None
    if all(e is not None for e in extras):
        extra_t = torch.stack(extras, 0)
    return xs, labels, ids, extra_t


def collate_varlen_multistain(batch):
    xs_ms, labels_list, ids, extras = [], [], [], []
    for x_dict, labels, sid, extra in batch:
        xs_ms.append(x_dict)
        labels_list.append(labels)
        ids.append(sid)
        extras.append(extra)
    labels = _stack_labels(labels_list)
    extra_t = None
    if all(e is not None for e in extras):
        extra_t = torch.stack(extras, 0)
    return xs_ms, labels, ids, extra_t


def make_dataloader(samples, mode="vector", batch_size=16, shuffle=True, num_workers=4, max_patches=None):
    if mode == "vector":
        dataset = SlideVectorDataset(samples)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_vector)
    if mode == "mil":
        dataset = SlidePatchSetDataset(samples, max_patches=max_patches)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_varlen)
    if mode == "mil_ms":
        dataset = SlideMultiStainPatchSetDataset(samples, max_patches_per_stain=max_patches)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_varlen_multistain)
    raise ValueError(f"Unknown mode: {mode}")
