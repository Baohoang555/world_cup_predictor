# 🏆 World Cup 2026 Championship Predictor

Hệ thống dự đoán đội vô địch World Cup 2026 dựa trên dữ liệu cầu thủ (Kaggle),
sử dụng kiến trúc **Hybrid Ensemble: Bradley-Terry + LightGBM**, kết hợp
**mô phỏng Monte Carlo** để tính tỷ lệ % vô địch cho từng đội tuyển.

## Kiến trúc hệ thống

```
Dữ liệu cầu thủ (Kaggle CSV)
        │
        ▼
[1] data_processing.py   → Lọc top-15 cầu thủ/đội, gộp thành chỉ số cấp đội tuyển
        │
        ▼
[2] feature_engineering.py → Tính power_attack / power_midfield / power_defense
        │                     + power_index tổng thể (theo trọng số config.yaml)
        ▼
[3] models/bradley_terry.py → Học "năng lực tiềm ẩn" θ mỗi đội từ lịch sử đối đầu
        │
        ▼
[4] models/lgbm_model.py  → LightGBM học cách kết hợp Power Index (hiện tại)
        │                    + xác suất Bradley-Terry (lịch sử) → xác suất thắng
        ▼
[5] simulation.py         → Mô phỏng Monte Carlo 10,000 lần toàn bộ giải đấu
        │                    (vòng bảng + knock-out) theo đúng thể thức 48 đội
        ▼
outputs/championship_odds.csv → Bảng % vô địch của từng đội
```

## Cài đặt

```bash
cd world_cup_predictor
pip install -r requirements.txt
```



