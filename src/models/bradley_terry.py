import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.utils import setup_logger

logger = setup_logger(__name__)


class BradleyTerryModel:
    def __init__(self, l2_penalty: float = 1.0):
        self.model = LogisticRegression(
            penalty="l2", C=1.0 / l2_penalty, fit_intercept=False, max_iter=1000
        )
        self.teams_ = None
        self.team_to_idx_ = None
        self.ratings_ = {}
        self.is_fitted = False

    # ------------------------------------------------------------------
    def _build_design_matrix(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        teams = sorted(set(matches["team_a"]) | set(matches["team_b"]))
        self.teams_ = teams
        self.team_to_idx_ = {t: i for i, t in enumerate(teams)}

        n_matches = len(matches)
        n_teams = len(teams)
        X = np.zeros((n_matches, n_teams))
        y = matches["result"].values.astype(float)

        for i, row in enumerate(matches.itertuples()):
            X[i, self.team_to_idx_[row.team_a]] = 1
            X[i, self.team_to_idx_[row.team_b]] = -1

        return X, y

    # ------------------------------------------------------------------
    def fit(self, matches: pd.DataFrame) -> "BradleyTerryModel":
        X, y = self._build_design_matrix(matches)
        self.model.fit(X, y)

        coefs = self.model.coef_[0]
        self.ratings_ = {team: coefs[idx] for team, idx in self.team_to_idx_.items()}
        self.is_fitted = True

        logger.info(f"Đã huấn luyện Bradley-Terry trên {len(matches)} trận, {len(self.teams_)} đội")
        return self

    # ------------------------------------------------------------------
    def get_rating(self, team: str) -> float:
        if team in self.ratings_:
            return self.ratings_[team]
        if self.ratings_:
            return float(np.median(list(self.ratings_.values())))
        return 0.0

    def predict_proba(self, team_a: str, team_b: str) -> float:
        """Xác suất team_a thắng team_b (bỏ qua khả năng hòa)."""
        if not self.is_fitted:
            raise RuntimeError("Model chưa được huấn luyện. Gọi .fit() trước.")
        diff = self.get_rating(team_a) - self.get_rating(team_b)
        return float(1 / (1 + np.exp(-diff)))

    # ------------------------------------------------------------------
    def ratings_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {"national_team": list(self.ratings_.keys()), "bt_rating": list(self.ratings_.values())}
        )
        return df.sort_values("bt_rating", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    demo = pd.DataFrame({
        "team_a": ["Argentina", "France", "Brazil", "Argentina"],
        "team_b": ["Brazil", "Argentina", "France", "France"],
        "result": [1, 0, 1, 1],
    })
    bt = BradleyTerryModel().fit(demo)
    print(bt.ratings_table())
    print("P(Argentina thắng France) =", bt.predict_proba("Argentina", "France"))
