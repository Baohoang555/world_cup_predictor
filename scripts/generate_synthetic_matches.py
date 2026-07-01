"""
generate_synthetic_matches.py

Bradley-Terry model cần dữ liệu LỊCH SỬ ĐỐI ĐẦU (match-level), điều mà dataset
Kaggle "Player Performance" KHÔNG cung cấp. Script này sinh dữ liệu lịch sử
đối đầu giả lập để bạn test pipeline end-to-end.

>>> KHUYẾN NGHỊ THAY THẾ BẰNG DỮ LIỆU THẬT <<<
Bạn nên tìm thêm 1 trong các bộ dữ liệu sau trên Kaggle và đặt vào
data/raw/international_matches.csv với 3 cột: team_a, team_b, result
(result = 1 nếu team_a thắng, 0 nếu team_b thắng, bỏ qua các trận hòa hoặc
tách thành 2 dòng 0.5/0.5 tuỳ chiến lược):
  - "International football results from 1872 to 2024" (Kaggle, martj42)
  - "FIFA World Ranking" (Kaggle)

Cách dùng:
    python scripts/generate_synthetic_matches.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, resolve_path
from scripts.generate_synthetic_data import TEAM_STRENGTH, TEAMS


def generate(n_matches: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_matches):
        a, b = rng.choice(TEAMS, size=2, replace=False)
        diff = (TEAM_STRENGTH[a] - TEAM_STRENGTH[b]) / 10
        p_a = 1 / (1 + np.exp(-diff))
        result = 1 if rng.random() < p_a else 0
        rows.append({"team_a": a, "team_b": b, "result": result})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = load_config()
    out_path = resolve_path(cfg["data"]["historical_matches_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate()
    df.to_csv(out_path, index=False)
    print(f"Đã sinh {len(df)} trận đấu lịch sử giả lập -> {out_path}")
    print(df.head())
