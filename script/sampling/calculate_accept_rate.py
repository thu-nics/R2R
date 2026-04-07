import argparse
import ast
import json
from pathlib import Path

import pandas as pd


TOKEN_TYPES_TO_KEEP = {1, 2}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate speculative-sampling-style accept rate from prediction comparison CSV."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="output_sampling/prefill_output/prediction_comparison_with_divergent.csv",
        help="Path to prediction comparison CSV.",
    )
    parser.add_argument(
        "--detail_output",
        type=str,
        default=None,
        help="Optional path to save per-token accept probability details.",
    )
    parser.add_argument(
        "--summary_output",
        type=str,
        default=None,
        help="Optional path to save accept-rate summary JSON.",
    )
    return parser.parse_args()


def _parse_list_cell(value) -> list:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    parsed = ast.literal_eval(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list cell, but got: {value!r}")
    return parsed


def _build_prob_lookup(token_ids, probs) -> dict[int, float]:
    aligned_count = min(len(token_ids), len(probs))
    lookup = {}
    for idx in range(aligned_count):
        token_id = int(token_ids[idx])
        lookup[token_id] = lookup.get(token_id, 0.0) + float(probs[idx])
    return lookup


def _has_effective_divergent_column(df: pd.DataFrame) -> bool:
    if "divergent" not in df.columns:
        return False
    divergent_series = pd.to_numeric(df["divergent"], errors="coerce").fillna(0)
    return divergent_series.ne(0).any()


def _resolve_target_probability(
    q_lookup: dict[int, float],
    p_lookup: dict[int, float],
    draft_token_id: int,
    real_token_id: int,
    use_merged_token: bool,
) -> tuple[float, float, list[int]]:
    if not use_merged_token:
        return (
            float(q_lookup.get(draft_token_id, 0.0)),
            float(p_lookup.get(draft_token_id, 0.0)),
            [draft_token_id],
        )
    merged_token_ids = list(dict.fromkeys([draft_token_id, real_token_id]))
    q_prob = sum(float(q_lookup.get(token_id, 0.0)) for token_id in merged_token_ids)
    p_prob = sum(float(p_lookup.get(token_id, 0.0)) for token_id in merged_token_ids)
    return q_prob, p_prob, merged_token_ids


def _prepare_aligned_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["data_id", "row_id"]).reset_index(drop=True)
    grouped = df.groupby("data_id", sort=False)

    df["llm_row_id"] = grouped["row_id"].shift(-1)
    df["llm_token_id"] = grouped["token_id"].shift(-1)
    df["llm_real_token"] = grouped["real_token"].shift(-1)
    df["llm_prediction_samples_shifted"] = grouped["LLM_prediction_samples"].shift(-1)
    df["llm_prediction_sample_probs_shifted"] = grouped[
        "LLM_prediction_sample_probs"
    ].shift(-1)
    if "divergent" in df.columns:
        df["llm_divergent"] = grouped["divergent"].shift(-1)

    df = df[df["token_type"].isin(TOKEN_TYPES_TO_KEEP)].copy()
    df = df.dropna(
        subset=[
            "llm_row_id",
            "llm_token_id",
            "llm_real_token",
            "llm_prediction_samples_shifted",
            "llm_prediction_sample_probs_shifted",
        ]
    ).reset_index(drop=True)
    return df


def _compute_accept_details(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    detail_rows = []
    dropped_missing_slm_prob = 0
    dropped_empty_candidates = 0
    merged_token_rows = 0
    effective_divergent = _has_effective_divergent_column(df)

    for row in df.itertuples(index=False):
        slm_token_ids = _parse_list_cell(row.SLM_prediction_samples)
        slm_probs = _parse_list_cell(row.SLM_prediction_sample_probs)
        llm_token_ids = _parse_list_cell(row.llm_prediction_samples_shifted)
        llm_probs = _parse_list_cell(row.llm_prediction_sample_probs_shifted)

        if not slm_token_ids or not slm_probs or not llm_token_ids or not llm_probs:
            dropped_empty_candidates += 1
            continue

        draft_token_id = int(row.SLM_predictions)
        real_token_id = int(row.llm_real_token)
        q_lookup = _build_prob_lookup(slm_token_ids, slm_probs)
        p_lookup = _build_prob_lookup(llm_token_ids, llm_probs)

        row_divergent = int(getattr(row, "llm_divergent", getattr(row, "divergent", 0)) or 0)
        use_merged_token = bool(effective_divergent and row_divergent != 0)
        q_prob, p_prob, merged_token_ids = _resolve_target_probability(
            q_lookup=q_lookup,
            p_lookup=p_lookup,
            draft_token_id=draft_token_id,
            real_token_id=real_token_id,
            use_merged_token=use_merged_token,
        )

        if q_prob <= 0.0:
            dropped_missing_slm_prob += 1
            continue

        accept_prob = min(1.0, float(p_prob) / max(float(q_prob), EPS))
        if use_merged_token:
            merged_token_rows += 1

        detail_rows.append(
            {
                "data_id": int(row.data_id),
                "token_type": int(row.token_type),
                "slm_row_id": int(row.row_id),
                "llm_row_id": int(row.llm_row_id),
                "slm_token_id": int(row.token_id),
                "llm_token_id": int(row.llm_token_id),
                "real_token_id": real_token_id,
                "draft_token_id": draft_token_id,
                "divergent": row_divergent,
                "used_merged_token": int(use_merged_token),
                "merged_token_ids": json.dumps(merged_token_ids, ensure_ascii=False),
                "q_prob": float(q_prob),
                "p_prob": float(p_prob),
                "accept_prob": float(accept_prob),
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    drop_stats = {
        "dropped_missing_slm_prob": int(dropped_missing_slm_prob),
        "dropped_empty_candidates": int(dropped_empty_candidates),
        "merged_token_rows": int(merged_token_rows),
        "effective_divergent_column": bool(effective_divergent),
    }
    return detail_df, drop_stats


def _build_summary(detail_df: pd.DataFrame, aligned_rows: int, drop_stats: dict) -> dict:
    def summarize(frame: pd.DataFrame) -> dict:
        drafted_tokens = int(len(frame))
        expected_accepted_tokens = float(frame["accept_prob"].sum()) if drafted_tokens else 0.0
        expected_rejected_tokens = float(drafted_tokens - expected_accepted_tokens)
        accept_rate = (
            expected_accepted_tokens / drafted_tokens if drafted_tokens > 0 else 0.0
        )
        return {
            "drafted_tokens": drafted_tokens,
            "expected_accepted_tokens": expected_accepted_tokens,
            "expected_rejected_tokens": expected_rejected_tokens,
            "accept_rate": accept_rate,
        }

    summary = {
        "aligned_rows_before_probability_drop": int(aligned_rows),
        "kept_rows": int(len(detail_df)),
        **drop_stats,
        "overall": summarize(detail_df),
        "token_type_1": summarize(detail_df[detail_df["token_type"] == 1]),
        "token_type_2": summarize(detail_df[detail_df["token_type"] == 2]),
    }
    return summary


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    output_dir = csv_path.parent

    detail_output = (
        Path(args.detail_output)
        if args.detail_output is not None
        else output_dir / "accept_rate_details.csv"
    )
    summary_output = (
        Path(args.summary_output)
        if args.summary_output is not None
        else output_dir / "accept_rate_summary.json"
    )

    df = pd.read_csv(csv_path)
    aligned_df = _prepare_aligned_rows(df)
    detail_df, drop_stats = _compute_accept_details(aligned_df)
    summary = _build_summary(detail_df, len(aligned_df), drop_stats)

    detail_df.to_csv(detail_output, index=False)
    with open(summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    overall = summary["overall"]
    print(f"Input CSV: {csv_path}")
    print(f"Aligned rows before probability drop: {summary['aligned_rows_before_probability_drop']}")
    print(f"Kept rows: {summary['kept_rows']}")
    print(f"Dropped empty candidate rows: {summary['dropped_empty_candidates']}")
    print(f"Dropped missing SLM probability rows: {summary['dropped_missing_slm_prob']}")
    print(f"Accept rate: {overall['accept_rate']:.6f}")
    print(f"Expected accepted tokens: {overall['expected_accepted_tokens']:.6f}")
    print(f"Expected rejected tokens: {overall['expected_rejected_tokens']:.6f}")
    print(f"Detail CSV saved to: {detail_output}")
    print(f"Summary JSON saved to: {summary_output}")


if __name__ == "__main__":
    main()
