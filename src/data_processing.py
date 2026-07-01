"""
data_processing.py — Giai đoạn 1 của pipeline

Chức năng:
1. Đọc dữ liệu cầu thủ thô từ file CSV (Kaggle dataset)
2. Kiểm tra & chuẩn hoá tên cột theo config.yaml (phòng khi tên cột thật khác)
3. Lọc top-N cầu thủ theo phút thi đấu cho mỗi đội (đội hình chính)
4. Gộp (aggregate) dữ liệu cầu thủ -> dữ liệu cấp đội tuyển, tách riêng theo
   từng tuyến (GK / DF / MF / FW) để phục vụ feature engineering ở bước sau
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger(__name__)


class PlayerDataProcessor:
    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or load_config()
        self.cols = self.cfg["data"]["columns"]
        self.top_n = self.cfg["aggregation"]["top_n_players_per_team"]

    # ------------------------------------------------------------------
    def load_raw(self, path: Optional[str] = None) -> pd.DataFrame:
        resolved_path = resolve_path(path or self.cfg["data"]["raw_player_file"])
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file dữ liệu cầu thủ tại {resolved_path}.\n"
                f"-> Hãy tải file CSV từ Kaggle và đặt vào data/raw/, "
                f"hoặc chạy 'python scripts/generate_synthetic_data.py' để test bằng dữ liệu giả lập."
            )
        df = pd.read_csv(resolved_path)
        logger.info(f"Đã đọc {len(df)} dòng cầu thủ từ {resolved_path}")
        return self._validate_columns(df)

    def _validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Chỉ các cột "lõi" là bắt buộc; các cột total_* là tuỳ chọn (fallback)
        core_keys = ["team", "player_id", "player_name", "position", "rating",
                     "minutes", "goals", "assists", "clutch", "xg"]
        core_cols = [self.cols[k] for k in core_keys if k in self.cols]
        missing = [c for c in core_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Các cột sau không tồn tại trong dataset: {missing}\n"
                f"-> Kiểm tra lại tên cột thật của file CSV và sửa trong config.yaml (mục data.columns)."
            )
        return df

    # ------------------------------------------------------------------
    def collapse_match_rows_to_player_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dataset gốc là cấp TRẬN ĐẤU: mỗi cầu thủ có nhiều dòng (1 dòng/trận,
        gắn với match_id). Hàm này gộp về 1 dòng/cầu thủ trước khi lọc đội
        hình chính & gộp cấp đội tuyển.

        Ưu tiên dùng các cột tổng-cả-giải có sẵn (total_goals_tournament,...)
        nếu dataset cung cấp, vì đáng tin cậy hơn tự cộng dồn thủ công
        (tránh lỗi đếm trùng nếu 1 match_id có nhiều dòng phụ). Nếu không có,
        tự động fallback sang cộng dồn từ các cột theo-trận.
        """
        c = self.cols
        team_col, id_col, name_col, pos_col = c["team"], c["player_id"], c["player_name"], c["position"]

        has_totals = all(
            k in c and c[k] in df.columns
            for k in ["total_minutes_tournament", "total_goals_tournament",
                      "total_assists_tournament", "tournament_rating"]
        )

        if has_totals:
            logger.info("Phát hiện cột tổng-cả-giải có sẵn -> dùng trực tiếp thay vì tự cộng dồn")
            agg = df.groupby(id_col).agg(**{
                "team_tmp": (team_col, "first"),
                "name_tmp": (name_col, "first"),
                "pos_tmp": (pos_col, "first"),
                "rating_tmp": (c["tournament_rating"], "max"),
                "minutes_tmp": (c["total_minutes_tournament"], "max"),
                "goals_tmp": (c["total_goals_tournament"], "max"),
                "assists_tmp": (c["total_assists_tournament"], "max"),
                "clutch_tmp": (c["clutch"], "mean"),
                "xg_tmp": (c["xg"], "sum"),
            }).reset_index()
        else:
            logger.info("Không có cột tổng-cả-giải -> tự cộng dồn từ các dòng theo trận")
            agg = df.groupby(id_col).agg(**{
                "team_tmp": (team_col, "first"),
                "name_tmp": (name_col, "first"),
                "pos_tmp": (pos_col, "first"),
                "rating_tmp": (c["rating"], "mean"),
                "minutes_tmp": (c["minutes"], "sum"),
                "goals_tmp": (c["goals"], "sum"),
                "assists_tmp": (c["assists"], "sum"),
                "clutch_tmp": (c["clutch"], "mean"),
                "xg_tmp": (c["xg"], "sum"),
            }).reset_index()

        agg = agg.rename(columns={
            "team_tmp": team_col, "name_tmp": name_col, "pos_tmp": pos_col,
            "rating_tmp": c["rating"], "minutes_tmp": c["minutes"],
            "goals_tmp": c["goals"], "assists_tmp": c["assists"],
            "clutch_tmp": c["clutch"], "xg_tmp": c["xg"],
        })
        logger.info(f"Đã gộp {len(df)} dòng cấp trận đấu -> {len(agg)} dòng cấp cầu thủ")
        return agg

    # ------------------------------------------------------------------
    def filter_core_squad(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chỉ giữ lại top-N cầu thủ có phút thi đấu cao nhất mỗi đội."""
        team_col = self.cols["team"]
        minutes_col = self.cols["minutes"]

        df_core = (
            df.sort_values(minutes_col, ascending=False)
            .groupby(team_col, group_keys=False)
            .head(self.top_n)
            .reset_index(drop=True)
        )
        logger.info(
            f"Đã lọc còn {len(df_core)} cầu thủ (top {self.top_n} theo phút thi đấu / đội)"
        )
        return df_core

    # ------------------------------------------------------------------
    def aggregate_team_stats(self, df_core: pd.DataFrame) -> pd.DataFrame:
        """Gộp dữ liệu cầu thủ -> chỉ số cấp đội tuyển, tách theo từng vị trí."""
        c = self.cols
        team_col, pos_col = c["team"], c["position"]

        # 1) Chỉ số tổng thể toàn đội
        overall = df_core.groupby(team_col).agg(
            squad_avg_rating=(c["rating"], "mean"),
            squad_max_rating=(c["rating"], "max"),
            squad_avg_clutch=(c["clutch"], "mean"),
            total_goals=(c["goals"], "sum"),
            total_assists=(c["assists"], "sum"),
            total_xg=(c["xg"], "sum"),
            squad_size=(c["player_name"], "count"),
        )

        # 2) Chỉ số riêng theo từng tuyến (GK/DF/MF/FW)
        pos_frames = {}
        for pos in ["GK", "DF", "MF", "FW"]:
            sub = df_core[df_core[pos_col].astype(str).str.upper() == pos]
            if sub.empty:
                continue
            agg = sub.groupby(team_col).agg(**{
                f"{pos}_avg_rating": (c["rating"], "mean"),
                f"{pos}_avg_clutch": (c["clutch"], "mean"),
                f"{pos}_total_goals": (c["goals"], "sum"),
                f"{pos}_total_assists": (c["assists"], "sum"),
                f"{pos}_total_xg": (c["xg"], "sum"),
                f"{pos}_count": (c["player_name"], "count"),
            })
            pos_frames[pos] = agg

        team_stats = overall
        for pos, frame in pos_frames.items():
            team_stats = team_stats.join(frame, how="left")

        team_stats = team_stats.fillna(0.0).reset_index()
        # Chuẩn hoá tên cột đội tuyển về "national_team" bất kể tên cột gốc
        # trong file CSV là gì (VD: "team") — để các module downstream
        # (feature_engineering, simulation, main) dùng chung 1 tên cột cố định.
        # Rename all columns: if a column name matches team_col, rename to "national_team"
        team_stats.columns = ["national_team" if c == team_col else c for c in team_stats.columns]
        logger.info(f"Đã gộp dữ liệu cho {len(team_stats)} đội tuyển")
        return team_stats

    # ------------------------------------------------------------------
    def run(self, save: bool = True) -> pd.DataFrame:
        df_raw = self.load_raw()

        if self.cfg["data"].get("is_match_level", False):
            df_raw = self.collapse_match_rows_to_player_level(df_raw)

        df_core = self.filter_core_squad(df_raw)
        team_stats = self.aggregate_team_stats(df_core)

        if save:
            out_path = resolve_path(self.cfg["data"]["processed_team_stats_file"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            team_stats.to_csv(out_path, index=False)
            logger.info(f"Đã lưu team_stats vào {out_path}")

        return team_stats


if __name__ == "__main__":
    processor = PlayerDataProcessor()
    stats = processor.run()
    print(stats.head(10))