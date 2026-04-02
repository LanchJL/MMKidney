#./data/datasets.py
import os, h5py, torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional

###########################
# 数据集划分与标签处理
###########################

def split_dataset_from_csv(
    csv_path: str,
    h5_dir: str,
    split_info: Dict[str, List[str]],
    kept_levels=("L1","L2","L3"),
    remove_normal=True
):
    """
    从 CSV 中读取样本，匹配 h5 特征文件，并返回划分好的样本列表。
    Args:
        csv_path: CSV 文件，包含 slide_id / label_L1 / label_L2 / label_L3
        h5_dir: h5 特征存放目录
        split_info: dict, { "train": [ids], "val": [...], "test": [...] }
        kept_levels: 哪些层级标签参与训练
        remove_normal: 是否去掉 Normal 类
    Return:
        dict: {"train": [records], "val": [...], "test": [...]}
    """
    df = pd.read_csv(csv_path)
    datasets = {"train": [], "val": [], "test": []}

    for split, ids in split_info.items():
        for sid in ids:
            row = df[df["slide_id"] == sid]
            if len(row) == 0: continue
            rec = {"id": sid, "labels": {}}

            # 检查是否存在对应 h5 文件
            h5_path = os.path.join(h5_dir, f"{sid}.h5")
            if not os.path.exists(h5_path):
                print(f"[WARN] Missing h5 for {sid}, skip")
                continue
            rec["h5"] = h5_path

            # Multi-hot 标签
            for lvl in kept_levels:
                lab = row[f"label_{lvl}"].values[0]
                if remove_normal and lab == "Normal":
                    continue
                rec["labels"][lvl] = lab
            datasets[split].append(rec)

    return datasets

###########################
# Slide-level 数据集
###########################

class SlideVectorDataset(Dataset):
    """
    直接读取 slide-level 向量的 Dataset
    每个 h5 文件应包含 ['features'] key
    """
    def __init__(self, samples: List[Dict], kept_levels=("L1","L2","L3")):
        self.samples = samples
        self.kept_levels = kept_levels

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        with h5py.File(rec["h5"], "r") as f:
            feat = torch.tensor(f["features"][:], dtype=torch.float32)
        labels = {lvl: rec["labels"].get(lvl, None) for lvl in self.kept_levels}
        return feat, labels, rec["id"]

###########################
# Patch-level MIL 数据集
###########################

class SlidePatchSetDataset(Dataset):
    """
    Patch set dataset for MIL
    支持以下任意一个 key:
      - 'patches'  (常见命名)
      - 'features' (很多人保存成 features[N, D])
      - 'feats' / 'x' (容错)
    如果读到的是 [D] 向量，会自动视为单 patch，扩一维成为 [1, D]
    """
    def __init__(self, samples: List[Dict], kept_levels=("L1","L2","L3")):
        self.samples = samples
        self.kept_levels = kept_levels
        self._candidate_keys = ["patches", "features", "feats", "x"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import numpy as np, h5py, torch
        rec = self.samples[idx]
        with h5py.File(rec["h5"], "r") as f:
            # 依次尝试多个 key
            data = None
            chosen = None
            for k in self._candidate_keys:
                if k in f:
                    data = f[k][:]
                    chosen = k
                    break
            if data is None:
                raise KeyError(
                    f"HDF5 missing any of keys {self._candidate_keys} in {rec['h5']}. "
                    f"Available keys = {list(f.keys())}"
                )

        patches = torch.tensor(data, dtype=torch.float32)
        # 统一成 [N, D]
        if patches.ndim == 1:
            patches = patches.unsqueeze(0)
        elif patches.ndim > 2:
            # 假如有人存成 [N, H, W, C] 之类的，这里最简单先平铺
            patches = patches.reshape(patches.shape[0], -1)

        labels = {lvl: rec["labels"].get(lvl, None) for lvl in self.kept_levels}
        return patches, labels, rec["id"]


###########################
# collate function for MIL
###########################

def collate_varlen(batch):
    """
    batch: [(patches, labels, id), ...]
    """
    xs, labels_list, ids = [], [], []
    for patches, labels, sid in batch:
        xs.append(patches)
        labels_list.append(labels)
        ids.append(sid)
    return xs, labels_list, ids



###########################
# DataLoader 构建工具
###########################

def make_dataloader(samples, mode="vector", batch_size=16, shuffle=True, num_workers=4):
    if mode == "vector":
        dataset = SlideVectorDataset(samples)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    elif mode == "mil":
        dataset = SlidePatchSetDataset(samples)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, collate_fn=collate_varlen)
    elif mode == "mil_ms":  # NEW
        dataset = SlideMultiStainPatchSetDataset(samples)  # 先 5k，能跑再调
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, collate_fn=collate_varlen_multistain)
    else:
        raise ValueError(f"Unknown mode {mode}")


###########################
# Multi Stain
###########################

class SlideMultiStainPatchSetDataset(Dataset):
    """
    Multi-stain Patch set dataset for MIL fusion.
    样本格式要求：
        rec = {
          "id": "...",
          "h5s": {"HE": ".../xxx_HE.h5", "LCA": "...", ...},
          "labels": {"L1": ..., "L2": ..., "L3": ...}
        }
    每个 h5 内部的特征 key 支持: ["patches", "features", "feats", "x"]（与单染色一致）
    返回：
        x_dict: {stain_name: torch.FloatTensor [N, D]}
        labels: {L1:..., L2:..., L3:...}
        id: str
    """
    def __init__(self, samples: List[Dict], kept_levels=("L1","L2","L3"), max_patches_per_stain: int = None):
        self.samples = samples
        self.kept_levels = kept_levels
        self._candidate_keys = ["patches", "features", "feats", "x"]

        self.max_patches = max_patches_per_stain
    def __len__(self):
        return len(self.samples)

    def _read_h5_to_tensor(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            data = None
            for k in self._candidate_keys:
                if k in f:
                    data = f[k][:]
                    break
            if data is None:
                raise KeyError(
                    f"HDF5 missing keys {self._candidate_keys} in {h5_path}. "
                    f"Available keys = {list(f.keys())}"
                )
        t = torch.tensor(data, dtype=torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(0)              # [D] -> [1, D]
        elif t.ndim > 2:
            t = t.reshape(t.shape[0], -1)   # [N, ...] -> [N, D]

        if self.max_patches is not None and t.shape[0] > self.max_patches:
            idx = torch.randperm(t.shape[0])[: self.max_patches]
            t = t[idx]
        return t

    def __getitem__(self, idx):
        rec = self.samples[idx]
        if "h5s" not in rec or not isinstance(rec["h5s"], dict) or len(rec["h5s"]) == 0:
            raise ValueError(f"Sample {rec.get('id','?')} missing 'h5s' dict for multi-stain.")

        x_dict = {}
        for stain, p in rec["h5s"].items():
            if not os.path.exists(p):
                # 允许缺失某些染色：跳过即可（后续融合会做掩码/自适应）
                # 也可以改成 raise，让数据严格
                continue
            x_dict[stain] = self._read_h5_to_tensor(p)

        if len(x_dict) == 0:
            raise ValueError(f"All stain files missing for sample {rec.get('id','?')}")

        labels = {lvl: rec["labels"].get(lvl, None) for lvl in self.kept_levels}
        return x_dict, labels, rec["id"]
def collate_varlen_multistain(batch):
    """
    batch: [(x_dict, labels, id), ...]
    x_dict: {stain_name: tensor [Ni, D]}，不同样本/染色的 Ni 可变
    直接保持“list of dict”结构，交给模型逐个处理
    """
    xs_ms, labels_list, ids = [], [], []
    for x_dict, labels, sid in batch:
        xs_ms.append(x_dict)
        labels_list.append(labels)
        ids.append(sid)
    return xs_ms, labels_list, ids