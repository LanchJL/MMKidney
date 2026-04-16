import argparse
import csv
from pathlib import Path
from typing import List


TEXT_FIELDS = [
    "pathology_summary",
    "gross_findings",
    "structured_report",
    "clinical_notes",
    "gt15_nephrology_dx",
    "microscopic_findings",
    "special_stain_description",
    "puncture_note",
    "general_note",
    "gt2",
]


def _read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _compose_text(row: dict, fields: List[str]) -> str:
    parts = []
    for k in fields:
        x = (row.get(k, "") or "").strip()
        if x:
            parts.append(f"[{k}] {x}")
    return "\n".join(parts)


def _mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    s = (last_hidden_state * mask).sum(dim=1)
    d = mask.sum(dim=1).clamp(min=1e-9)
    return s / d


def main():
    p = argparse.ArgumentParser("Offline text embedding builder for clinicopath pipeline")
    p.add_argument("--cohort-tabular", default="data/processed/cohort_tabular.csv")
    p.add_argument("--out-csv", default="data/processed/qwen_global_embeddings.csv")
    p.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--instruction", default="Represent the kidney pathology report for diagnosis and prognosis.")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--embed-dim", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:
        raise RuntimeError("build_text_embeddings requires torch and transformers installed.") from e

    rows = _read_rows(Path(args.cohort_tabular))
    fields = [k for k in TEXT_FIELDS if any((r.get(k, "") or "").strip() for r in rows)]
    texts = []
    sids = []
    for r in rows:
        sids.append((r.get("sample_id", "") or "").strip())
        body = _compose_text(r, fields)
        full = f"Instruct: {args.instruction}\nText: {body}"
        texts.append(full)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    dev = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model.to(dev).eval()

    out = []
    bs = max(1, int(args.batch_size))
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            bt = texts[i : i + bs]
            enc = tokenizer(
                bt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(args.max_length),
            )
            enc = {k: v.to(dev) for k, v in enc.items()}
            feat = model(**enc)
            if hasattr(feat, "last_hidden_state"):
                pooled = _mean_pool(feat.last_hidden_state, enc["attention_mask"])
            else:
                pooled = feat[0][:, 0]

            if int(args.embed_dim) > 0 and pooled.shape[1] != int(args.embed_dim):
                # dim adaptation by projection-free slicing/padding
                d = int(args.embed_dim)
                if pooled.shape[1] > d:
                    pooled = pooled[:, :d]
                else:
                    pad = torch.zeros((pooled.shape[0], d - pooled.shape[1]), device=pooled.device, dtype=pooled.dtype)
                    pooled = torch.cat([pooled, pad], dim=1)

            pooled = pooled.detach().cpu()
            for j in range(pooled.shape[0]):
                row = {"sample_id": sids[i + j]}
                for k, v in enumerate(pooled[j].tolist()):
                    row[f"emb_{k:04d}"] = float(v)
                out.append(row)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fields = list(out[0].keys()) if out else ["sample_id"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    print("[text-emb] rows:", len(out))
    print("[text-emb] out:", out_path)
    print("[text-emb] model:", args.model_name)


if __name__ == "__main__":
    main()
