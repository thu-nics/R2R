"""
Hidden State L2 距离测试

两组对比：
1. divergent vs neutral: 仅 mismatch token（SLM ≠ LLM），按 divergent 标记分组
2. divergent vs non-divergent: 所有 token，non-divergent = identical + neutral

对每个 token 位置，利用 SLM 的 KV cache 分别 forward draft token 和 real token，
比较两者的 last-layer hidden state 的 L2 距离。
identical 位置（draft==real）直接赋 distance=0 以跳过冗余 forward。

Usage:
    python test_hidden_state_pair_distance.py \
        --csv_path output_sampling/prefill_output/prediction_comparison_with_divergent.csv \
        --model_path /share/public/public_models/Qwen3-0.6B \
        --output_dir output/neutral_token_test/metric2_hidden_state
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from test_embedding_pair_distance import (
    compute_group_stats,
    plot_boxplot,
    plot_density,
)

TOKEN_TYPES_TO_KEEP = {1, 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hidden state L2 distance test."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="output_sampling/prefill_output/prediction_comparison_with_divergent.csv",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/share/public/public_models/Qwen3-0.6B",
        help="SLM model path for forward passes.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/neutral_token_test/metric2_hidden_state",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Max number of data samples to process (-1 = all).",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=32768,
        help="Skip samples longer than this.",
    )
    return parser.parse_args()


# ─────────────────────── data preparation ────────────────────────


def _prepare_aligned_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["data_id", "row_id"]).reset_index(drop=True)
    grouped = df.groupby("data_id", sort=False)

    df["llm_real_token"] = grouped["real_token"].shift(-1)
    if "divergent" in df.columns:
        df["llm_divergent"] = grouped["divergent"].shift(-1)

    df = df[df["token_type"].isin(TOKEN_TYPES_TO_KEEP)].copy()
    df = df.dropna(subset=["llm_real_token"]).reset_index(drop=True)
    df["llm_real_token"] = df["llm_real_token"].astype(int)
    if "llm_divergent" in df.columns:
        df["llm_divergent"] = df["llm_divergent"].fillna(0).astype(int)
    return df


def prepare_test_data(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (test_df, full_df)."""
    print(f"Loading CSV from {csv_path}")
    full_df = pd.read_csv(csv_path)
    aligned = _prepare_aligned_rows(full_df.copy())

    aligned["draft_token"] = aligned["SLM_predictions"].astype(int)
    aligned["real_token_next"] = aligned["llm_real_token"].astype(int)

    divergent_col = "llm_divergent" if "llm_divergent" in aligned.columns else "divergent"
    aligned["label"] = aligned[divergent_col].fillna(0).astype(int)

    is_identical = aligned["draft_token"] == aligned["real_token_next"]
    is_divergent = aligned["label"] == 1
    aligned["category"] = "neutral"
    aligned.loc[is_identical, "category"] = "identical"
    aligned.loc[is_divergent, "category"] = "divergent"

    n_identical = int(is_identical.sum())
    n_neutral = int(((~is_identical) & (~is_divergent)).sum())
    n_divergent = int(is_divergent.sum())
    print(f"Total positions: {len(aligned)}")
    print(f"  identical={n_identical}, neutral={n_neutral}, divergent={n_divergent}")
    return aligned, full_df


# ──────────────────── KV cache helpers ───────────────────────────


def _slice_past_kv(past_key_values, end_pos: int):
    """Return past_key_values truncated to positions [0, end_pos] (inclusive)."""
    if past_key_values is None:
        return None

    if hasattr(past_key_values, "key_cache"):
        from transformers import DynamicCache
        sliced = DynamicCache()
        for i in range(len(past_key_values.key_cache)):
            sliced.update(
                past_key_values.key_cache[i][:, :, : end_pos + 1, :],
                past_key_values.value_cache[i][:, :, : end_pos + 1, :],
                i,
            )
        return sliced

    return tuple(
        (k[:, :, : end_pos + 1, :], v[:, :, : end_pos + 1, :])
        for k, v in past_key_values
    )


# ─────────────────── distance computation ────────────────────────


def compute_hidden_state_distances(
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    model: torch.nn.Module,
    max_samples: int = -1,
    max_input_length: int = 32768,
) -> pd.DataFrame:
    device = next(model.parameters()).device
    test_df = test_df.reset_index(drop=True)

    l2_array = np.full(len(test_df), np.nan)

    identical_mask = test_df["draft_token"].values == test_df["real_token_next"].values
    l2_array[identical_mask] = 0.0
    print(f"Identical positions (distance=0 trivially): {int(identical_mask.sum())}")

    full_sorted = full_df.sort_values(["data_id", "token_id"]).reset_index(drop=True)
    token_seqs: dict[int, torch.Tensor] = {}
    for data_id, group in full_sorted.groupby("data_id"):
        token_seqs[int(data_id)] = torch.tensor(
            group["real_token"].values, dtype=torch.long
        )

    mismatch_indices = test_df.index[~identical_mask]
    mismatch_sub = test_df.loc[mismatch_indices]
    mismatch_groups = mismatch_sub.groupby("data_id").groups
    data_ids = sorted(mismatch_groups.keys())
    if max_samples > 0:
        data_ids = data_ids[:max_samples]

    processed = 0
    skipped = 0
    for data_id in tqdm(data_ids, desc="Processing samples (mismatch only)"):
        if data_id not in token_seqs:
            continue

        input_ids = token_seqs[data_id].unsqueeze(0).to(device)
        seq_len = input_ids.shape[1]

        if seq_len > max_input_length:
            skipped += 1
            continue

        with torch.no_grad():
            outputs = model(
                input_ids, use_cache=True, output_hidden_states=True
            )

        all_hidden = outputs.hidden_states[-1]
        full_kv = outputs.past_key_values

        indices = mismatch_groups[data_id]
        for idx in indices:
            row = test_df.loc[idx]
            t = int(row["token_id"])
            draft_token_id = int(row["draft_token"])

            if t + 1 >= seq_len:
                continue

            h_real = all_hidden[0, t + 1].cpu().float()

            sliced_kv = _slice_past_kv(full_kv, t)
            draft_input = torch.tensor(
                [[draft_token_id]], dtype=torch.long, device=device
            )
            with torch.no_grad():
                draft_out = model(
                    draft_input,
                    past_key_values=sliced_kv,
                    use_cache=True,
                    output_hidden_states=True,
                )
            h_draft = draft_out.hidden_states[-1][0, 0].cpu().float()

            l2_array[idx] = torch.norm(h_real - h_draft).item()

            del sliced_kv, draft_out

        del outputs, all_hidden, full_kv
        torch.cuda.empty_cache()
        processed += 1

    print(f"Processed {processed} samples, skipped {skipped} (too long)")

    test_df = test_df.copy()
    valid_mask = ~np.isnan(l2_array)
    test_df["l2_distance"] = l2_array
    test_df = test_df[valid_mask].reset_index(drop=True)
    print(f"Valid positions with distances: {len(test_df)}")
    return test_df


# ─────────────────────── main ────────────────────────────────────


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    test_df, full_df = prepare_test_data(args.csv_path)
    if len(test_df) == 0:
        print("No test positions found. Exiting.")
        return

    print(f"Loading model {args.model_path} ...")
    model_config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=model_config,
        device_map="auto",
        torch_dtype=torch.float16,
    ).eval()
    print("Model loaded.")

    test_df = compute_hidden_state_distances(
        test_df,
        full_df,
        model,
        max_samples=args.max_samples,
        max_input_length=args.max_input_length,
    )

    if len(test_df) == 0:
        print("No valid distances computed. Exiting.")
        return

    divergent_l2 = test_df.loc[test_df["category"] == "divergent", "l2_distance"].values
    neutral_l2 = test_df.loc[test_df["category"] == "neutral", "l2_distance"].values
    non_divergent_l2 = test_df.loc[test_df["category"] != "divergent", "l2_distance"].values

    all_stats = {}

    # 对比 1: divergent vs neutral
    if len(divergent_l2) > 0 and len(neutral_l2) > 0:
        stats = compute_group_stats(divergent_l2, neutral_l2, "divergent", "neutral")
        all_stats["divergent_vs_neutral"] = stats
        plot_density(
            neutral_l2, divergent_l2, "neutral", "divergent",
            title="Hidden State L2 Distance: Divergent vs Neutral",
            filename=os.path.join(args.output_dir, "density_divergent_vs_neutral.png"),
        )
        plot_boxplot(
            neutral_l2, divergent_l2, "neutral", "divergent",
            title="Hidden State L2 Distance: Divergent vs Neutral",
            filename=os.path.join(args.output_dir, "boxplot_divergent_vs_neutral.png"),
        )

    # 对比 2: divergent vs non-divergent (identical + neutral)
    if len(divergent_l2) > 0 and len(non_divergent_l2) > 0:
        stats = compute_group_stats(
            divergent_l2, non_divergent_l2, "divergent", "non_divergent"
        )
        all_stats["divergent_vs_non_divergent"] = stats
        plot_density(
            non_divergent_l2, divergent_l2, "non-divergent", "divergent",
            title="Hidden State L2 Distance: Divergent vs Non-Divergent",
            filename=os.path.join(args.output_dir, "density_divergent_vs_non_divergent.png"),
        )
        plot_boxplot(
            non_divergent_l2, divergent_l2, "non-divergent", "divergent",
            title="Hidden State L2 Distance: Divergent vs Non-Divergent",
            filename=os.path.join(args.output_dir, "boxplot_divergent_vs_non_divergent.png"),
        )

    stats_path = os.path.join(args.output_dir, "statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\nSaved statistics to {stats_path}")

    for comp_name, comp_stats in all_stats.items():
        print(f"\n=== {comp_name} ===")
        for key in comp_stats:
            if isinstance(comp_stats[key], dict):
                g = comp_stats[key]
                print(f"  {key}: count={g['count']}  mean={g['mean']:.4f}  "
                      f"std={g['std']:.4f}  median={g['median']:.4f}")
        print(f"  Mann-Whitney p={comp_stats['mann_whitney_p']:.2e}  "
              f"Cohen's d={comp_stats['cohens_d']:.4f}")


if __name__ == "__main__":
    main()
