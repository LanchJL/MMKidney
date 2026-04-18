import argparse
import os
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser("Build markdown checklist for manual L3 review")
    p.add_argument("--output-dir", default="prototype_pipeline_output")
    p.add_argument("--batch-run-dir", default="", help="Path like output-dir/batch_l3_runs/<timestamp>")
    p.add_argument("--out-md", default="")
    args = p.parse_args()

    root = Path(args.output_dir) / "batch_l3_runs"
    if args.batch_run_dir:
        run_dir = Path(args.batch_run_dir)
    else:
        cands = sorted(root.glob("*"))
        if not cands:
            raise FileNotFoundError(f"No batch run dirs found under {root}")
        run_dir = cands[-1]

    sum_csv = run_dir / "batch_l3_summary.csv"
    chk_csv = run_dir / "batch_l3_check_report.csv"
    if not sum_csv.exists():
        raise FileNotFoundError(sum_csv)

    sum_df = pd.read_csv(sum_csv)
    chk_df = pd.read_csv(chk_csv) if chk_csv.exists() else pd.DataFrame()

    out_md = Path(args.out_md) if args.out_md else run_dir / "manual_review_checklist.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# L3 Batch Manual Review Checklist")
    lines.append("")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- summary: `{sum_csv}`")
    if chk_csv.exists():
        lines.append(f"- check_report: `{chk_csv}`")
    lines.append("")
    lines.append("## Recommended Review Order")
    lines.append("")

    if len(chk_df):
        order_df = chk_df.copy()
        rank_map = {"KEEP": 0, "REVIEW": 1, "DROP": 2}
        order_df["rank"] = order_df["recommendation"].map(rank_map).fillna(9)
        order_df = order_df.sort_values(["rank", "top1_ratio", "target_l2"], ascending=[True, True, True])
        targets = order_df["target_l2"].astype(str).tolist()
        lines.append("| target_l2 | recommendation | n_l3_kept | top1_ratio | reason |")
        lines.append("|---|---:|---:|---:|---|")
        for _, r in order_df.iterrows():
            lines.append(
                f"| {r['target_l2']} | {r['recommendation']} | {int(r['n_l3_kept'])} | {float(r['top1_ratio']):.6f} | {r['reason']} |"
            )
    else:
        targets = sum_df["target_l2"].astype(str).tolist()
        lines.append("- No check report found; use batch summary order.")

    lines.append("")
    lines.append("## Files To Inspect (Per Target)")
    lines.append("")
    for t in targets:
        lines.append(f"### {t}")
        lines.append(f"- meta: `{Path('prototype_pipeline_output') / f'l3_refine_meta_target_{t}_HE_2048.json'}`")
        lines.append(f"- size: `{Path('prototype_pipeline_output') / f'final_cluster_sizes_HE_2048_l3_target_{t}.csv'}`")
        lines.append(f"- patch_final: `{Path('prototype_pipeline_output') / f'patch_final_clusters_HE_2048_l3_target_{t}.parquet'}`")
        lines.append(f"- slide_hist: `{Path('prototype_pipeline_output') / f'slide_final_cluster_hist_HE_2048_l3_target_{t}.csv'}`")
        lines.append(f"- run_copy_dir: `{run_dir / t}`")
        lines.append("")

    lines.append("## Suggested Manual Checks")
    lines.append("")
    lines.append("- 检查每个 target 的 L3 子簇是否均衡（避免1大多小）。")
    lines.append("- 抽检 overlay：每个 L3 子簇至少看 2-3 张切片。")
    lines.append("- 如果出现明显噪声子簇，记录并考虑回退该 target 到 L2。")
    lines.append("- 在确认后更新最终保留列表：KEEP / REVIEW / DROP。")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_md}")
    print("[done] manual review checklist generated")


if __name__ == "__main__":
    main()

