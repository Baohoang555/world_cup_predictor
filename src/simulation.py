import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.bradley_terry import BradleyTerryModel
from src.models.lgbm_model import LGBMMatchPredictor, TeamStatsIndex
from src.utils import setup_logger

logger = setup_logger(__name__)


class HybridMatchPredictor:
    """Kết hợp Bradley-Terry (lịch sử) + LightGBM (chất lượng đội hình hiện tại)."""

    def __init__(self, team_stats: pd.DataFrame, bt_model: BradleyTerryModel,
                 lgbm_model: LGBMMatchPredictor, draw_weight: float = 0.25):
        self.team_stats = team_stats
        self.bt_model = bt_model
        self.lgbm_model = lgbm_model
        self.draw_weight = draw_weight  # trọng số ước lượng xác suất hòa cho vòng bảng
        # Index numpy hoá để tra cứu đặc trưng cực nhanh trong vòng lặp Monte Carlo
        self.stats_index = TeamStatsIndex(team_stats, bt_model)

    def win_probability(self, team_a: str, team_b: str) -> float:
        """Xác suất team_a thắng team_b, bỏ qua khả năng hòa (dùng cho knock-out)."""
        X = self.stats_index.features(team_a, team_b)
        p = self.lgbm_model.predict_proba(pd.DataFrame(X, columns=["diff_attack", "diff_midfield", "diff_defense", "diff_power_index", "diff_avg_rating", "diff_clutch", "bt_proba"]))[0]
        return float(np.clip(p, 0.01, 0.99))

    def match_outcome_probs(self, team_a: str, team_b: str) -> dict:
        """Trả về xác suất Thắng/Hòa/Thua cho vòng bảng (có khả năng hòa).
        Xấp xỉ: lấy p_win từ LightGBM rồi 'cắt' một phần xác suất cho kèo hòa,
        theo tỷ lệ tương quan với mức độ cân bằng giữa 2 đội (càng cân bằng,
        xác suất hòa càng cao)."""
        p_a = self.win_probability(team_a, team_b)
        # Mức độ cân bằng: càng gần 0.5 thì càng dễ hòa
        balance = 1 - 2 * abs(p_a - 0.5)  # 1 nếu 50-50, 0 nếu 1 đội áp đảo
        p_draw = self.draw_weight * balance
        p_a_final = p_a * (1 - p_draw)
        p_b_final = (1 - p_a) * (1 - p_draw)
        return {"win_a": p_a_final, "draw": p_draw, "win_b": p_b_final}


@dataclass
class TournamentSimulator:
    predictor: HybridMatchPredictor
    groups: dict  # {"A": ["Team1", "Team2", "Team3", "Team4"], ...}
    n_runs: int = 10000
    seed: int = 42
    knockout_qualifiers_per_group: int = 2  # World Cup 2026: top 2 + 8 đội hạng 3 xuất sắc
    third_place_wildcards: int = 8

    def _simulate_group_stage(self, rng: np.random.Generator) -> dict:
        """Trả về bảng xếp hạng (điểm số) của từng bảng đấu."""
        standings = {}
        for group_name, teams in self.groups.items():
            points = {t: 0 for t in teams}
            goal_diff_proxy = {t: 0.0 for t in teams}  # dùng để phá vỡ đồng điểm

            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    a, b = teams[i], teams[j]
                    probs = self.predictor.match_outcome_probs(a, b)
                    r = rng.random()
                    if r < probs["win_a"]:
                        points[a] += 3
                        goal_diff_proxy[a] += probs["win_a"] - probs["win_b"]
                        goal_diff_proxy[b] -= probs["win_a"] - probs["win_b"]
                    elif r < probs["win_a"] + probs["draw"]:
                        points[a] += 1
                        points[b] += 1
                    else:
                        points[b] += 3
                        goal_diff_proxy[b] += probs["win_b"] - probs["win_a"]
                        goal_diff_proxy[a] -= probs["win_b"] - probs["win_a"]

            ranking = sorted(
                teams, key=lambda t: (points[t], goal_diff_proxy[t]), reverse=True
            )
            standings[group_name] = ranking
        return standings

    def _select_knockout_bracket(self, standings: dict) -> list:
        """Chọn các đội vào vòng knock-out: top-2 mỗi bảng + các đội hạng 3 xuất sắc nhất."""
        qualifiers = []
        third_placed = []
        for group_name, ranking in standings.items():
            qualifiers.extend(ranking[: self.knockout_qualifiers_per_group])
            if len(ranking) > self.knockout_qualifiers_per_group:
                third_placed.append(ranking[self.knockout_qualifiers_per_group])

        # Lấy ngẫu nhiên (đơn giản hoá) các đội hạng 3 xuất sắc thay vì tính chi tiết
        random.shuffle(third_placed)
        qualifiers.extend(third_placed[: self.third_place_wildcards])
        return qualifiers

    def _simulate_knockout(self, teams: list, rng: np.random.Generator) -> str:
        """Mô phỏng vòng loại trực tiếp, trả về đội vô địch. Không có hòa —
        nếu 'hòa ảo' xảy ra thì xử lý luân lưu bằng đồng xu có trọng số nhẹ."""
        random.shuffle(teams)  # bốc thăm ngẫu nhiên đơn giản hoá
        current_round = teams

        while len(current_round) > 1:
            next_round = []
            for i in range(0, len(current_round) - 1, 2):
                a, b = current_round[i], current_round[i + 1]
                p_a = self.predictor.win_probability(a, b)
                winner = a if rng.random() < p_a else b
                next_round.append(winner)
            # nếu số đội lẻ, đội cuối cùng được bye (hiếm khi xảy ra nếu bracket chuẩn)
            if len(current_round) % 2 == 1:
                next_round.append(current_round[-1])
            current_round = next_round

        return current_round[0]

    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        champion_counts = {}

        all_teams = [t for teams in self.groups.values() for t in teams]
        for t in all_teams:
            champion_counts[t] = 0

        for _ in tqdm(range(self.n_runs), desc="Monte Carlo simulation"):
            standings = self._simulate_group_stage(rng)
            bracket = self._select_knockout_bracket(standings)
            champion = self._simulate_knockout(bracket, rng)
            champion_counts[champion] += 1

        result = pd.DataFrame({
            "national_team": list(champion_counts.keys()),
            "championship_count": list(champion_counts.values()),
        })
        result["championship_probability"] = result["championship_count"] / self.n_runs
        result = result.sort_values("championship_probability", ascending=False).reset_index(drop=True)
        return result
