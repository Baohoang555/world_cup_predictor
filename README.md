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

## Cách chạy nhanh (test bằng dữ liệu giả lập)

Vì hệ thống chưa có sẵn file Kaggle thật, bạn có thể test toàn bộ pipeline bằng
dữ liệu giả lập trước:

```bash
python scripts/generate_synthetic_data.py       # sinh dữ liệu cầu thủ giả lập
python scripts/generate_synthetic_matches.py    # sinh dữ liệu lịch sử đối đầu giả lập
python main.py                                  # chạy toàn bộ pipeline
```

Kết quả sẽ in ra Top 10 ứng viên vô địch và lưu đầy đủ vào
`outputs/championship_odds.csv`.

## Cách dùng với dữ liệu THẬT

### Bước 1 — Tải dataset chính
Tải **"FIFA World Cup 2026 Player Performance Dataset"** từ Kaggle
(tác giả Ra'uf Fauzan Rambe), giải nén, đặt file CSV vào:
```
data/raw/fifa_world_cup_2026_player_performance.csv
```

### Bước 2 — Kiểm tra tên cột
Mở file CSV, đối chiếu tên cột thật với mục `data.columns` trong `config.yaml`.
Nếu tên cột khác (VD: dataset dùng `country` thay vì `national_team`), **chỉ cần
sửa trong config.yaml**, không cần sửa code.

### Bước 3 — Bổ sung dữ liệu lịch sử đối đầu (bắt buộc cho Bradley-Terry)
Dataset Kaggle gốc **chỉ có dữ liệu cấp cầu thủ**, không có kết quả trận đấu
lịch sử. Bradley-Terry cần dữ liệu này để học "năng lực tiềm ẩn" của mỗi đội.

Khuyến nghị tải thêm 1 trong các bộ dữ liệu sau trên Kaggle:
- *"International football results from 1872 to 2024"* (tác giả martj42)
- *"FIFA World Ranking"*

Chuẩn hoá về 3 cột: `team_a, team_b, result` (result = 1 nếu team_a thắng,
0 nếu team_b thắng — trận hòa nên loại bỏ hoặc tách thành 2 dòng 0.5/0.5),
lưu vào:
```
data/raw/international_matches.csv
```

### Bước 4 — Chạy pipeline
```bash
python main.py
```

## Tuỳ chỉnh mô hình

Tất cả tham số quan trọng đều nằm trong `config.yaml`, KHÔNG cần sửa code:

| Tham số | Ý nghĩa |
|---|---|
| `aggregation.top_n_players_per_team` | Số cầu thủ "đội hình chính" lấy theo phút thi đấu |
| `position_weights` | Trọng số các chỉ số theo từng vị trí (GK/DF/MF/FW) |
| `line_weights` | Trọng số gộp attack/midfield/defense → power_index |
| `model.lightgbm` | Hyperparameters của LightGBM |
| `simulation.monte_carlo_runs` | Số lần mô phỏng (càng cao càng chính xác, càng chậm) |

## Học trọng số vị trí tự động (nâng cao)

Thay vì gán trọng số thủ công trong `position_weights`, bạn có thể dùng
`FeatureEngineer.learn_position_weights_from_benchmark()` trong
`src/feature_engineering.py` để học trọng số tối ưu bằng Random Forest
Feature Importance, nếu có thêm dataset benchmark như:
- *"Football Players Stats 2024-2025"* (FBref, Kaggle)
- *"EA Sports FC 24/25 Player Ratings"* (Kaggle)

Xem ví dụ chi tiết trong docstring của hàm đó.

## Cấu trúc thư mục

```
world_cup_predictor/
├── config.yaml                    # Cấu hình trung tâm
├── main.py                        # Điểm chạy pipeline đầy đủ
├── requirements.txt
├── data/
│   ├── raw/                       # Đặt file CSV gốc (Kaggle) vào đây
│   └── processed/                 # Dữ liệu đã gộp cấp đội tuyển
├── src/
│   ├── data_processing.py         # Bước 1: gộp cầu thủ -> đội tuyển
│   ├── feature_engineering.py     # Bước 2: tính Power Index theo tuyến
│   ├── simulation.py              # Bước 4-5: Hybrid predictor + Monte Carlo
│   ├── utils.py                   # Helper: load config, logging
│   └── models/
│       ├── bradley_terry.py       # Bước 3: Bradley-Terry model
│       └── lgbm_model.py          # Bước 4: LightGBM predictor
├── scripts/
│   ├── generate_synthetic_data.py     # Sinh dữ liệu cầu thủ giả lập để test
│   └── generate_synthetic_matches.py  # Sinh dữ liệu lịch sử đối đầu giả lập
├── notebooks/
│   └── 01_eda.ipynb               # Notebook khám phá dữ liệu (EDA)
└── outputs/
    ├── championship_odds.csv      # KẾT QUẢ CUỐI: tỷ lệ % vô địch mỗi đội
    └── lgbm_model.txt             # Model LightGBM đã huấn luyện
```

## Giới hạn cần lưu ý

- **Bốc thăm bảng đấu**: `main.py` hiện chia bảng đấu ngẫu nhiên
  (`make_random_groups`), không theo đúng pot/seeding thật của FIFA. Khi có
  kết quả bốc thăm World Cup 2026 chính thức, hãy thay bằng bảng đấu thật.
- **Dữ liệu lịch sử đối đầu**: bắt buộc phải có để Bradley-Terry hoạt động
  đúng nghĩa — dùng dữ liệu giả lập chỉ để test pipeline, không dùng để dự
  đoán thật.
- **Vòng loại trực tiếp**: hiện mô phỏng bốc thăm ngẫu nhiên đơn giản hoá,
  chưa áp đúng sơ đồ nhánh đấu chính thức của FIFA (có thể mở rộng thêm ở
  `TournamentSimulator._simulate_knockout`).
