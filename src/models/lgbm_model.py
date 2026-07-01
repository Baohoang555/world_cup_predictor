"""
lgbm_model.py — LightGBM Match Outcome Predictor

Dự đoán xác suất thắng của team_a trước team_b dựa trên:
- Chênh lệch các chỉ số sức mạnh theo tuyến (attack/midfield/defense) từ feature_engineering.py
- Xác suất gốc từ Bradley-Terry model (dùng làm 1 feature "prior" lịch sử)

Đây chính là kiến trúc Hybrid Ensemble: Bradley-Terry (lịch sử đối đầu)
đóng vai trò 1 feature đầu vào cho LightGBM (phong độ/chất lượng hiện tại),
để mô hình tự học cách cân bằng giữa 2 nguồn thông tin.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.utils import setup_logger

logger = setup_logger(__name__)

FEATURE_COLUMNS = [
    "diff_attack",
    "diff_midfield",
    "diff_defense",
    "diff_power_index",
    "diff_avg_rating",
    "diff_clutch",
    "bt_proba",
]


def build_match_features(
    team_a: str, team_b: str, team_stats: pd.DataFrame, bt_model=None
) -> pd.DataFrame:
    """Tạo vector đặc trưng cho 1 cặp đấu, dùng chung cho training & inference.
    (Dùng cho training set / các lệnh gọi đơn lẻ; xem TeamStatsIndex bên dưới
    để có phiên bản tối ưu tốc độ cho vòng lặp Monte Carlo hàng chục nghìn lần)."""
    row_a = team_stats[team_stats["national_team"] == team_a].iloc[0]
    row_b = team_stats[team_stats["national_team"] == team_b].iloc[0]

    bt_proba = bt_model.predict_proba(team_a, team_b) if bt_model is not None else 0.5

    feats = {
        "diff_attack": row_a["power_attack_norm"] - row_b["power_attack_norm"],
        "diff_midfield": row_a["power_midfield_norm"] - row_b["power_midfield_norm"],
        "diff_defense": row_a["power_defense_norm"] - row_b["power_defense_norm"],
        "diff_power_index": row_a["power_index"] - row_b["power_index"],
        "diff_avg_rating": row_a["squad_avg_rating"] - row_b["squad_avg_rating"],
        "diff_clutch": row_a["squad_avg_clutch"] - row_b["squad_avg_clutch"],
        "bt_proba": bt_proba,
    }
    return pd.DataFrame([feats])[FEATURE_COLUMNS]


class TeamStatsIndex:
    """Tra cứu chỉ số đội tuyển bằng numpy array thay vì pandas filtering,
    giúp vòng lặp Monte Carlo (hàng chục nghìn trận/lần chạy) nhanh hơn
    hàng trăm lần so với gọi build_match_features trực tiếp trên DataFrame."""

    STAT_COLS = [
        "power_attack_norm", "power_midfield_norm", "power_defense_norm",
        "power_index", "squad_avg_rating", "squad_avg_clutch",
    ]

    def __init__(self, team_stats: pd.DataFrame, bt_model=None):
        self.team_to_idx = {t: i for i, t in enumerate(team_stats["national_team"])}
        self.matrix = team_stats[self.STAT_COLS].to_numpy(dtype=np.float64)
        self.bt_ratings = None
        if bt_model is not None:
            self.bt_ratings = {t: bt_model.get_rating(t) for t in self.team_to_idx}

    def _bt_proba(self, team_a: str, team_b: str) -> float:
        if self.bt_ratings is None:
            return 0.5
        diff = self.bt_ratings.get(team_a, 0.0) - self.bt_ratings.get(team_b, 0.0)
        return float(1 / (1 + np.exp(-diff)))

    def features(self, team_a: str, team_b: str) -> np.ndarray:
        a = self.matrix[self.team_to_idx[team_a]]
        b = self.matrix[self.team_to_idx[team_b]]
        diff = a - b  # [attack, midfield, defense, power_index, rating, clutch]
        bt_p = self._bt_proba(team_a, team_b)
        return np.concatenate([diff, [bt_p]]).reshape(1, -1)  # khớp thứ tự FEATURE_COLUMNS


class LGBMMatchPredictor:
    def __init__(self, config: dict):
        self.params = dict(config["model"]["lightgbm"])
        self.num_boost_round = self.params.pop("num_boost_round", 200)
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 20)
        self.booster = None

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None):
        train_set = lgb.Dataset(X, label=y)
        valid_sets = [train_set]
        callbacks = []
        if X_val is not None and y_val is not None:
            valid_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
            valid_sets.append(valid_set)
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        self.booster = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            callbacks=callbacks if callbacks else None,
        )
        logger.info("Đã huấn luyện xong LightGBM Match Predictor")
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model chưa được huấn luyện. Gọi .fit() trước.")
        return self.booster.predict(X)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))
        logger.info(f"Đã lưu model LightGBM vào {path}")

    def load(self, path: str):
        self.booster = lgb.Booster(model_file=str(path))
        return self
