"""
generate_synthetic_data.py

Sinh dữ liệu cầu thủ GIẢ LẬP theo đúng schema THẬT của dataset Kaggle
"FIFA World Cup 2026 Player Performance Dataset" (đã đối chiếu trực tiếp với
file mẫu người dùng cung cấp) để test pipeline khi CHƯA có file thật.

Đặc điểm quan trọng của schema thật: đây là dữ liệu CẤP TRẬN ĐẤU — mỗi cầu
thủ có NHIỀU dòng (1 dòng / trận đấu, gắn với match_id), kèm theo các cột
tổng-cả-giải (total_goals_tournament,...) đã tính sẵn.

Cách dùng:
    python scripts/generate_synthetic_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, resolve_path

TEAMS = [
    "Argentina", "France", "Brazil", "England", "Spain", "Portugal",
    "Germany", "Netherlands", "Belgium", "Italy", "Croatia", "Uruguay",
    "Colombia", "Morocco", "Japan", "South Korea", "United States", "Mexico",
    "Canada", "Senegal", "Vietnam", "Australia", "Ecuador", "Switzerland",
]

POSITIONS = {"GK": 3, "DF": 8, "MF": 8, "FW": 6}

TEAM_STRENGTH = {
    "Argentina": 88, "France": 87, "Brazil": 86, "England": 85, "Spain": 85,
    "Portugal": 84, "Germany": 83, "Netherlands": 83, "Belgium": 81, "Italy": 81,
    "Croatia": 80, "Uruguay": 79, "Colombia": 78, "Morocco": 77, "Japan": 76,
    "South Korea": 75, "United States": 75, "Mexico": 74, "Canada": 72, "Senegal": 74,
    "Vietnam": 62, "Australia": 71, "Ecuador": 73, "Switzerland": 77,
}

MATCHES_PER_TEAM = 3  # giả lập mỗi đội đá 3 trận vòng bảng


def generate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    player_id = 1

    for team in TEAMS:
        base_strength = TEAM_STRENGTH[team]
        opponents = rng.choice(
            [t for t in TEAMS if t != team], size=MATCHES_PER_TEAM, replace=False
        )

        for pos, n_players in POSITIONS.items():
            for _ in range(n_players):
                pid = f"P{player_id:05d}"
                base_rating = np.clip(rng.normal(base_strength, 4), 45, 95)

                cum_goals, cum_assists, cum_minutes = 0, 0, 0
                match_rows = []

                for m_idx, opponent in enumerate(opponents):
                    match_id = f"M{team[:3].upper()}{m_idx}"
                    rating = np.clip(rng.normal(base_rating, 3), 40, 99)
                    minutes = int(np.clip(rng.normal(75, 15), 0, 90))
                    clutch = np.clip(rng.normal(base_strength - 5, 6), 30, 95)

                    if pos == "FW":
                        goals = max(0, int(rng.poisson(base_strength / 60)))
                        assists = max(0, int(rng.poisson(base_strength / 80)))
                        xg = round(max(0, rng.normal(goals + 0.3, 0.5)), 2)
                    elif pos == "MF":
                        goals = max(0, int(rng.poisson(base_strength / 120)))
                        assists = max(0, int(rng.poisson(base_strength / 70)))
                        xg = round(max(0, rng.normal(goals + 0.15, 0.3)), 2)
                    elif pos == "DF":
                        goals = max(0, int(rng.poisson(base_strength / 200)))
                        assists = max(0, int(rng.poisson(base_strength / 150)))
                        xg = round(max(0, rng.normal(goals + 0.05, 0.15)), 2)
                    else:  # GK
                        goals, assists, xg = 0, 0, 0.0

                    cum_goals += goals
                    cum_assists += assists
                    cum_minutes += minutes

                    match_rows.append({
                        "player_id": pid,
                        "player_name": f"{team}_Player_{player_id}",
                        "age": int(np.clip(rng.normal(26, 4), 18, 38)),
                        "nationality": team,
                        "team": team,
                        "jersey_number": rng.integers(1, 26),
                        "position": pos,
                        "match_id": match_id,
                        "match_date": "2026-07-10",
                        "opponent_team": opponent,
                        "tournament_stage": "Group Stage",
                        "goals_team": rng.integers(0, 4),
                        "goals_opponent": rng.integers(0, 4),
                        "minutes_played": minutes,
                        "goals": goals,
                        "assists": assists,
                        "expected_goals_xg": xg,
                        "player_rating": round(rating, 1),
                        "clutch_performance_score": round(clutch, 1),
                        "total_goals_tournament": cum_goals,
                        "total_assists_tournament": cum_assists,
                        "total_minutes_tournament": cum_minutes,
                        "tournament_rating": round(rating, 1),
                    })

                rows.extend(match_rows)
                player_id += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = load_config()
    out_path = resolve_path(cfg["data"]["raw_player_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate()
    df.to_csv(out_path, index=False)
    print(f"Đã sinh {len(df)} dòng dữ liệu cấp trận đấu -> {out_path}")
    print(f"({df['player_id'].nunique()} cầu thủ, {df['match_id'].nunique()} trận đấu)")
    print(df.head())