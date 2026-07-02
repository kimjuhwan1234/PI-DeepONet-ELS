# -*- coding: utf-8 -*-
"""모든 파일 경로 중앙 관리. 하드코딩 경로 대신 이 모듈만 사용."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"

DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
RESULT = ROOT / "result"
PRED = RESULT / "predictions"
IMAGE = RESULT / "image"
STAT = RESULT / "statistics"
SCRATCH = ROOT / "scratch"

# 모델링 데이터셋 3종
DATASETS = {
    "ml": DATA / "ml.parquet",  # 직접 tabular 벤치마크 + 하이브리드 마진모델
    "deeponet": DATA / "deeponet.parquet",  # DeepONet/PI-DeepONet(MLP)
    "deeponet_curve": DATA / "deeponet_curve.parquet",  # DeepONet-Curve
}


def ensure_dirs():
    for d in (DATA, RAW, CACHE, RESULT, PRED, IMAGE, STAT, SCRATCH):
        d.mkdir(parents=True, exist_ok=True)


def dataset(name: str) -> Path:
    """모델링 데이터셋 경로. name in {ml, deeponet, deeponet_curve}."""
    return DATASETS[name]


def source() -> Path:
    """원천(피처 완성본) 경로."""
    return CACHE / "els3_dataset.parquet"


def prediction(name: str) -> Path:
    PRED.mkdir(parents=True, exist_ok=True)
    return PRED / f"{name}.csv"


def image(name: str) -> Path:
    IMAGE.mkdir(parents=True, exist_ok=True)
    return IMAGE / f"{name}.png"


def stat(name: str) -> Path:
    STAT.mkdir(parents=True, exist_ok=True)
    return STAT / f"{name}.csv"


def list_predictions():
    """저장된 예측 csv 경로 목록 (2_evaluate 에서 사용)."""
    return sorted(PRED.glob("*.csv"))
