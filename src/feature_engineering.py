from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.utils import load_config, setup_logger

logger = setup_logger(__name__)


def _normalize_weights(weights: dict) -> dict:
    total = sum(weights.values())
    if total == 0:
        return weights
    return {k: v / total for k, v in weights.items()}


class FeatureEngineer:
    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or load_config()
        self.pos_weights = self.cfg["position_weights"]
        self.line_weights = _normalize_weights(self.cfg["line_weights"])

    # ------------------------------------------------------------------
    def learn_position_weights_from_benchmark(
        self, df_benchmark: pd.DataFrame, position: str,
        feature_cols: list, target_col: str = "overall_rating",
    ) -> dict:
        """
        Học trọng số tối ưu cho 1 vị trí bằng RandomForest Feature Importance,
        dùng bộ dữ liệu benchmark bên ngoài (VD: FBref, EA Sports FC Ratings).

        df_benchmark: DataFrame chỉ chứa cầu thủ ở đúng vị trí `position`
        feature_cols: các cột chỉ số đầu vào (VD: goals, xg, assists, tackles...)
        target_col: cột điểm tổng (ground-truth) để học, VD Overall Rating trong FIFA/EA FC
        """
        X = df_benchmark[feature_cols].fillna(0)
        y = df_benchmark[target_col].fillna(0)

        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X, y)

        importances = pd.Series(model.feature_importances_, index=feature_cols)
        weights = _normalize_weights(importances.to_dict())

        logger.info(f"Trọng số học được cho vị trí {position}: {weights}")
        self.pos_weights[position] = weights
        return weights

    # ------------------------------------------------------------------
    def compute_line_scores(self, team_stats: pd.DataFrame) -> pd.DataFrame:
        """Tính điểm attack/midfield/defense cho mỗi đội dựa trên chỉ số theo tuyến."""
        df = team_stats.copy()

        # Attack (FW) — trọng số theo rating / goals / xg
        w_fw = self.pos_weights.get("FW", {})
        df["power_attack"] = (
            df.get("FW_avg_rating", 0) * w_fw.get("rating", 0.35)
            + df.get("FW_total_goals", 0) * w_fw.get("goals", 0.35)
            + df.get("FW_total_xg", 0) * w_fw.get("xg", 0.30)
        )

        # Midfield (MF) — trọng số theo rating / assists / clutch
        w_mf = self.pos_weights.get("MF", {})
        df["power_midfield"] = (
            df.get("MF_avg_rating", 0) * w_mf.get("rating", 0.4)
            + df.get("MF_total_assists", 0) * w_mf.get("assists", 0.35)
            + df.get("MF_avg_clutch", 0) * w_mf.get("clutch", 0.25)
        )

        # Defense (DF + một phần GK) — trọng số theo rating / clutch / (nghịch đảo bàn thua-proxy)
        w_df = self.pos_weights.get("DF", {})
        w_gk = self.pos_weights.get("GK", {})
        defense_field = (
            df.get("DF_avg_rating", 0) * w_df.get("rating", 0.5)
            + df.get("DF_avg_clutch", 0) * w_df.get("clutch", 0.3)
        )
        defense_gk = (
            df.get("GK_avg_rating", 0) * w_gk.get("rating", 0.6)
            + df.get("GK_avg_clutch", 0) * w_gk.get("clutch", 0.4)
        )
        df["power_defense"] = 0.65 * defense_field + 0.35 * defense_gk

        # Chuẩn hoá về thang 0-100 cho từng tuyến để công bằng khi gộp
        for col in ["power_attack", "power_midfield", "power_defense"]:
            min_v, max_v = df[col].min(), df[col].max()
            if max_v > min_v:
                df[col + "_norm"] = 100 * (df[col] - min_v) / (max_v - min_v)
            else:
                df[col + "_norm"] = 50.0

        # Power Index tổng thể = trung bình trọng số của 3 tuyến (đã chuẩn hoá)
        df["power_index"] = (
            df["power_attack_norm"] * self.line_weights.get("attack", 0.35)
            + df["power_midfield_norm"] * self.line_weights.get("midfield", 0.35)
            + df["power_defense_norm"] * self.line_weights.get("defense", 0.30)
        )

        return df

    # ------------------------------------------------------------------
    def run(self, team_stats: pd.DataFrame) -> pd.DataFrame:
        df = self.compute_line_scores(team_stats)
        df = df.sort_values("power_index", ascending=False).reset_index(drop=True)
        logger.info("Top 5 đội theo Power Index:\n" +
                     df[["national_team", "power_index", "power_attack_norm",
                         "power_midfield_norm", "power_defense_norm"]].head(5).to_string(index=False))
        return df


if __name__ == "__main__":
    from src.data_processing import PlayerDataProcessor

    team_stats = PlayerDataProcessor().run(save=False)
    fe = FeatureEngineer()
    result = fe.run(team_stats)
    print(result.head(10))
