"""
utils.py — Các hàm tiện ích dùng chung cho toàn pipeline:
- Load config.yaml
- Thiết lập logging
- Resolve đường dẫn tương đối theo root dự án
"""

import logging
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str = None) -> dict:
    """Đọc file config.yaml và trả về dict cấu hình."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config tại: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def resolve_path(relative_path: str) -> Path:
    """Chuyển đường dẫn tương đối trong config.yaml thành đường dẫn tuyệt đối
    tính từ thư mục gốc của dự án."""
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def setup_logger(name: str = "wc_predictor", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def ensure_dir(path) -> None:
    os.makedirs(path, exist_ok=True)
