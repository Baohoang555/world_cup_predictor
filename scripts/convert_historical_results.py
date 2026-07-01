import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, resolve_path, setup_logger

logger = setup_logger(__name__)


def convert(cfg: dict) -> pd.DataFrame:
    hc = cfg["data"]["historical_results_columns"]
    min_year = cfg["data"].get("historical_results_min_year", 2010)

    src_path = resolve_path(cfg["data"]["raw_historical_results_file"])
    if not src_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {src_path}.\n"
            f"-> Đặt file results.csv (International football results) vào data/raw/"
        )

    df = pd.read_csv(src_path)
    logger.info(f"Đã đọc {len(df)} trận đấu lịch sử từ {src_path}")

    df[hc["date"]] = pd.to_datetime(df[hc["date"]], errors="coerce")
    df = df[df[hc["date"]].dt.year >= min_year].copy()
    logger.info(f"Còn {len(df)} trận từ năm {min_year} trở về sau")

    aliases = cfg["data"].get("team_name_aliases", {})
    if aliases:
        df[hc["home_team"]] = df[hc["home_team"]].replace(aliases)
        df[hc["away_team"]] = df[hc["away_team"]].replace(aliases)

    rows = []
    for row in df.itertuples():
        home = getattr(row, hc["home_team"])
        away = getattr(row, hc["away_team"])
        home_score = getattr(row, hc["home_score"])
        away_score = getattr(row, hc["away_score"])

        if pd.isna(home_score) or pd.isna(away_score):
            continue

        if home_score > away_score:
            rows.append({"team_a": home, "team_b": away, "result": 1})
        elif home_score < away_score:
            rows.append({"team_a": home, "team_b": away, "result": 0})
        else:
            # Trận hoà -> sinh 2 dòng đối nghịch (xem docstring ở trên)
            rows.append({"team_a": home, "team_b": away, "result": 1})
            rows.append({"team_a": home, "team_b": away, "result": 0})

    result_df = pd.DataFrame(rows)
    logger.info(f"Đã chuyển thành {len(result_df)} dòng team_a/team_b/result")
    return result_df


if __name__ == "__main__":
    cfg = load_config()
    result_df = convert(cfg)

    out_path = resolve_path(cfg["data"]["historical_matches_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"Đã lưu {len(result_df)} dòng vào {out_path}")
    print(result_df.head())