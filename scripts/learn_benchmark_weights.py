"""
learn_benchmark_weights.py

Học trọng số tối ưu cho từng vị trí (GK/DF/MF/FW) bằng Random Forest Feature
Importance, dùng dataset benchmark THẬT "EA Sports FC 24/25 Player Ratings"
(male_players.csv) làm ground-truth: dự đoán điểm OVR (Overall Rating) của
EA Sports từ các chỉ số kỹ năng thành phần, sau đó lấy độ quan trọng của
từng chỉ số làm trọng số.

Đây chính là "Bước A" trong phần "Phương pháp lập trình để tích hợp vào mô
hình dự đoán" mà bạn đã brainstorm — EA Sports đã mất hàng chục năm tối ưu
công thức chấm OVR, nên trọng số học được từ đây là một benchmark đáng tin cậy.

Cách dùng:
    python scripts/learn_benchmark_weights.py

Sau khi chạy, trọng số học được sẽ được in ra và LƯU ĐÈ vào config.yaml
(mục position_weights) — hãy backup config.yaml trước nếu bạn muốn giữ trọng
số thủ công ban đầu.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.feature_engineering import FeatureEngineer
from src.utils import load_config, resolve_path, setup_logger, PROJECT_ROOT

logger = setup_logger(__name__)


def load_benchmark(cfg: dict) -> pd.DataFrame:
    path = resolve_path(cfg["data"]["benchmark_players_file"])
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}.\n"
            f"-> Đặt file male_players.csv (EA Sports FC Ratings) vào data/raw/"
        )
    df = pd.read_csv(path)
    logger.info(f"Đã đọc {len(df)} cầu thủ từ dataset benchmark EA Sports FC")
    return df


def main():
    cfg = load_config()
    bench_cfg = cfg["benchmark"]
    bc = bench_cfg["columns"]

    df = load_benchmark(cfg)
    fe = FeatureEngineer(cfg)

    learned_weights = {}
    for pos_group, ea_codes in bench_cfg["position_group_map"].items():
        subset = df[df[bc["position"]].isin(ea_codes)].copy()
        if subset.empty:
            logger.warning(f"Không tìm thấy cầu thủ nào cho nhóm vị trí {pos_group} "
                            f"(mã EA FC: {ea_codes}) -> bỏ qua, giữ trọng số cũ")
            continue

        feature_cols = bench_cfg["feature_columns"][pos_group]
        missing_cols = [c for c in feature_cols if c not in subset.columns]
        if missing_cols:
            logger.warning(f"Thiếu cột {missing_cols} cho nhóm {pos_group} trong "
                            f"male_players.csv -> bỏ qua, kiểm tra lại config.yaml")
            continue

        weights = fe.learn_position_weights_from_benchmark(
            subset, pos_group, feature_cols, target_col=bc["overall"]
        )
        learned_weights[pos_group] = {k: round(v, 4) for k, v in weights.items()}
        print(f"\n[{pos_group}] Trọng số học được từ {len(subset)} cầu thủ:")
        for feat, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"   {feat:25s} {w:.3f}")

    if not learned_weights:
        print("\nKhông học được trọng số nào — kiểm tra lại tên cột trong config.yaml (mục benchmark).")
        return

    # Lưu kết quả ra 1 file riêng (KHÔNG tự động ghi đè config.yaml để an toàn)
    out_path = PROJECT_ROOT / "outputs" / "learned_position_weights.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"position_weights": learned_weights}, f, allow_unicode=True)

    print(f"\nĐã lưu trọng số học được vào: {out_path}")
    print("\n LƯU Ý QUAN TRỌNG:")
    print("   Trọng số học được ở trên nằm trên không gian đặc trưng của EA Sports FC")
    print("   (VD: 'Finishing', 'GK Diving'...), khác với các đặc trưng dùng để tính")
    print("   Power Index trong feature_engineering.py (VD: 'rating', 'goals', 'xg'...).")
    print("   -> Không copy trực tiếp key-value vào config.yaml.")
    print("   Giá trị thực sự của bước này là cho bạn biết THỨ TỰ ƯU TIÊN của các nhóm")
    print("   kỹ năng theo từng vị trí (VD: Finishing > Positioning > Shot Power cho FW),")
    print("   từ đó bạn điều chỉnh trọng số thủ công trong position_weights một cách có")
    print("   căn cứ, thay vì đoán mò.")


if __name__ == "__main__":
    main()