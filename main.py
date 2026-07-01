"""
main.py — Điểm khởi chạy toàn bộ hệ thống dự đoán vô địch World Cup 2026

Chạy:
    python main.py

Pipeline thực hiện tuần tự:
1. Đọc & gộp dữ liệu cầu thủ -> dữ liệu đội tuyển        (src/data_processing.py)
2. Tính điểm sức mạnh theo tuyến + Power Index            (src/feature_engineering.py)
3. Huấn luyện Bradley-Terry trên lịch sử đối đầu          (src/models/bradley_terry.py)
4. Huấn luyện LightGBM Hybrid Match Predictor             (src/models/lgbm_model.py)
5. Mô phỏng Monte Carlo toàn bộ giải đấu (vòng bảng +
   knock-out) hàng nghìn lần                              (src/simulation.py)
6. Xuất bảng tỷ lệ % vô địch ra outputs/championship_odds.csv
"""

import random

import numpy as np
import pandas as pd

from src.data_processing import PlayerDataProcessor
from src.feature_engineering import FeatureEngineer
from src.models.bradley_terry import BradleyTerryModel
from src.models.lgbm_model import LGBMMatchPredictor, build_match_features, FEATURE_COLUMNS
from src.simulation import HybridMatchPredictor, TournamentSimulator
from src.utils import load_config, resolve_path, setup_logger, ensure_dir

logger = setup_logger("main")


def make_random_groups(teams: list, group_size: int = 4, seed: int = 42) -> dict:
    """Chia ngẫu nhiên các đội vào bảng đấu (đơn giản hoá bốc thăm thật của FIFA,
    vốn có seeding theo pot). Có thể thay bằng bốc thăm thật nếu bạn có dữ liệu."""
    rng = random.Random(seed)
    shuffled = teams.copy()
    rng.shuffle(shuffled)

    groups = {}
    n_groups = len(shuffled) // group_size
    letters = [chr(ord("A") + i) for i in range(n_groups)]
    for i, letter in enumerate(letters):
        groups[letter] = shuffled[i * group_size: (i + 1) * group_size]
    return groups


def build_match_training_set(matches: pd.DataFrame, team_stats: pd.DataFrame,
                              bt_model: BradleyTerryModel) -> tuple:
    """Chuyển dữ liệu lịch sử đối đầu -> tập huấn luyện đặc trưng cho LightGBM."""
    X_rows, y_rows = [], []
    known_teams = set(team_stats["national_team"])
    skipped = 0

    for row in matches.itertuples():
        if row.team_a not in known_teams or row.team_b not in known_teams:
            skipped += 1
            continue  # bỏ qua trận đấu có đội không nằm trong bộ dữ liệu cầu thủ hiện tại
        feats = build_match_features(str(row.team_a), str(row.team_b), team_stats, bt_model)
        X_rows.append(feats.iloc[0])
        y_rows.append(row.result)

    if skipped:
        logger.warning(
            f"Đã bỏ qua {skipped}/{len(matches)} trận vì tên đội không khớp với dataset cầu thủ.\n"
            f"-> Nếu số này lớn, kiểm tra mục 'team_name_aliases' trong config.yaml "
            f"để thêm ánh xạ tên đội còn thiếu."
        )

    X = pd.DataFrame(X_rows).reset_index(drop=True)
    y = np.array(y_rows)
    return X, y


def main():
    cfg = load_config()
    ensure_dir(resolve_path(cfg["output"]["results_dir"]))

    # ---------- BƯỚC 1: Data Processing ----------
    logger.info("=== BƯỚC 1/5: Đọc & gộp dữ liệu cầu thủ -> đội tuyển ===")
    team_stats = PlayerDataProcessor(cfg).run()

    # ---------- BƯỚC 2: Feature Engineering ----------
    logger.info("=== BƯỚC 2/5: Tính Power Index theo từng tuyến ===")
    team_stats = FeatureEngineer(cfg).run(team_stats)

    # ---------- BƯỚC 3: Bradley-Terry ----------
    logger.info("=== BƯỚC 3/5: Huấn luyện Bradley-Terry trên lịch sử đối đầu ===")
    matches_path = resolve_path(cfg["data"]["historical_matches_file"])
    if not matches_path.exists():
        raw_results_path = resolve_path(cfg["data"]["raw_historical_results_file"])
        if raw_results_path.exists():
            logger.info(f"Chưa có {matches_path.name} -> tự động chuyển đổi từ {raw_results_path.name}")
            from scripts.convert_historical_results import convert as convert_results
            result_df = convert_results(cfg)
            matches_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(matches_path, index=False)
        else:
            raise FileNotFoundError(
                f"Không tìm thấy {matches_path} lẫn {raw_results_path}.\n"
                f"-> Đặt results.csv (International football results) vào data/raw/,\n"
                f"   hoặc chạy: python scripts/generate_synthetic_matches.py để test bằng dữ liệu giả lập."
            )
    matches = pd.read_csv(matches_path)
    bt_model = BradleyTerryModel().fit(matches)

    # ---------- BƯỚC 4: LightGBM Hybrid ----------
    logger.info("=== BƯỚC 4/5: Huấn luyện LightGBM Hybrid Match Predictor ===")
    X, y = build_match_training_set(matches, team_stats, bt_model)
    split = int(len(X) * 0.85)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y[:split], y[split:]

    lgbm_model = LGBMMatchPredictor(cfg).fit(X_train, y_train, X_val, y_val)
    lgbm_model.save(str(resolve_path(cfg["output"]["results_dir"]) / "lgbm_model.txt"))

    # ---------- BƯỚC 5: Monte Carlo Tournament Simulation ----------
    logger.info("=== BƯỚC 5/5: Mô phỏng Monte Carlo toàn bộ giải đấu ===")
    predictor = HybridMatchPredictor(
        team_stats, bt_model, lgbm_model,
        draw_weight=cfg["simulation"]["draw_probability_weight"],
    )

    all_teams = team_stats["national_team"].tolist()
    groups = make_random_groups(all_teams, seed=cfg["simulation"]["random_seed"])
    logger.info(f"Đã chia {len(all_teams)} đội vào {len(groups)} bảng đấu")

    simulator = TournamentSimulator(
        predictor=predictor,
        groups=groups,
        n_runs=cfg["simulation"]["monte_carlo_runs"],
        seed=cfg["simulation"]["random_seed"],
    )
    odds = simulator.run()

    out_path = resolve_path(cfg["output"]["championship_odds_file"])
    odds.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("KẾT QUẢ DỰ ĐOÁN: TOP 10 ỨNG CỬ VIÊN VÔ ĐỊCH WORLD CUP 2026")
    print("=" * 60)
    odds["championship_probability"] = (odds["championship_probability"] * 100).round(2)
    print(odds.head(10).to_string(index=False))
    print(f"\nĐã lưu bảng đầy đủ vào: {out_path}")


if __name__ == "__main__":
    main()