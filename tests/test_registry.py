import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import registry


def test_discover_finds_known_models():
    names = [n for n, _ in registry.discover()]
    assert "deeponet" in names
    assert "benchmark" in names


def test_discover_skips_private_registry_and_noninrun():
    names = [n for n, _ in registry.discover()]
    assert "registry" not in names   # 자기 자신 제외
    assert "_template" not in names  # 언더스코어 제외
    assert "stage2" not in names     # run() 없는 모듈(잔차모델) 제외
    assert all(callable(fn) for _, fn in registry.discover())


def test_run_all_rejects_duplicate_names():
    import pytest
    from types import SimpleNamespace
    # 두 가짜 모델이 같은 이름을 반환하면 ValueError
    def fn_a(D, cfg): return {"dup": 1}
    def fn_b(D, cfg): return {"dup": 2}
    orig = registry.discover
    registry.discover = lambda: [("a", fn_a), ("b", fn_b)]
    try:
        with pytest.raises(ValueError):
            registry.run_all(SimpleNamespace(), {}, verbose=False)
    finally:
        registry.discover = orig
