"""Shared noise-assignment helpers for experiments and benchmarks."""

from __future__ import annotations

from collections import Counter

import pandas as pd


def noise_assignment_mode(cfg: dict) -> str:
    return str(cfg.get("noise_assignment", "cartesian")).strip().lower()


def expected_noise_rows(n_examples: int, cfg: dict) -> int:
    mode = noise_assignment_mode(cfg)
    noise_types = cfg["noise_types"]
    variants = int(cfg.get("noise_variants_per_question", 1))

    if mode == "round_robin":
        if variants != 1:
            raise ValueError("round_robin noise assignment requires noise_variants_per_question == 1")
        return n_examples

    return n_examples * len(noise_types) * variants


def build_noise_plan(df: pd.DataFrame, cfg: dict) -> list[tuple[pd.Series, str]]:
    mode = noise_assignment_mode(cfg)
    noise_types = list(cfg["noise_types"])
    variants = int(cfg.get("noise_variants_per_question", 1))

    if not noise_types:
        raise ValueError("Config must define at least one noise_type")

    plan: list[tuple[pd.Series, str]] = []
    if mode == "round_robin":
        if variants != 1:
            raise ValueError("round_robin noise assignment requires noise_variants_per_question == 1")
        for idx, (_, row) in enumerate(df.iterrows()):
            plan.append((row, noise_types[idx % len(noise_types)]))
        return plan

    for _, row in df.iterrows():
        for noise_type in noise_types:
            for _ in range(variants):
                plan.append((row, noise_type))
    return plan


def summarize_noise_plan(plan: list[tuple[pd.Series, str]]) -> dict[str, int]:
    return dict(Counter(noise_type for _, noise_type in plan))
