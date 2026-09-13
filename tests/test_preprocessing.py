"""
tests/test_preprocessing.py
Chạy: pytest tests/test_preprocessing.py -v
"""
import numpy as np
import pytest
from sklearn.datasets import load_wine
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import chia_train_test, chuan_hoa


@pytest.fixture
def du_lieu():
    wine = load_wine()
    return wine.data, wine.target


# ------------------------------------------------------------------
# Kiểm tra chia_train_test
# ------------------------------------------------------------------

def test_chia_ti_le(du_lieu):
    X, y = du_lieu
    X_train, X_test, _, _ = chia_train_test(X, y, ti_le_test=0.2)
    expected_test = round(len(X) * 0.2)
    # cho phép ±1 mẫu do làm tròn
    assert abs(len(X_test) - expected_test) <= 1, (
        f"Tập test nên có ~{expected_test} mẫu, nhưng có {len(X_test)}"
    )


def test_chia_tong_so_mau(du_lieu):
    X, y = du_lieu
    X_train, X_test, y_train, y_test = chia_train_test(X, y)
    assert len(X_train) + len(X_test) == len(X), "Tổng số mẫu sau khi chia phải bằng ban đầu"


def test_chia_stratify(du_lieu):
    """Kiểm tra rằng stratify giữ nguyên tỉ lệ nhóm."""
    X, y = du_lieu
    _, _, y_train, y_test = chia_train_test(X, y, ti_le_test=0.2)
    # tỉ lệ mỗi nhóm trong train và toàn bộ không được lệch quá 5%
    for cls in np.unique(y):
        ratio_all   = np.mean(y == cls)
        ratio_train = np.mean(y_train == cls)
        assert abs(ratio_train - ratio_all) < 0.05, (
            f"Nhóm {cls}: tỉ lệ train={ratio_train:.2f} quá khác với toàn bộ={ratio_all:.2f}"
        )


def test_chia_khong_rong(du_lieu):
    X, y = du_lieu
    X_train, X_test, y_train, y_test = chia_train_test(X, y)
    assert len(X_train) > 0 and len(X_test) > 0, "Tập train và test không được rỗng"


# ------------------------------------------------------------------
# Kiểm tra chuan_hoa
# ------------------------------------------------------------------

def test_chuan_hoa_trung_binh(du_lieu):
    X, y = du_lieu
    X_train, X_test, _, _ = chia_train_test(X, y)
    _, X_train_sc, _ = chuan_hoa(X_train, X_test)
    means = np.abs(X_train_sc.mean(axis=0))
    assert np.all(means < 1e-9), (
        f"Sau chuẩn hóa, trung bình train phải ≈ 0. Lớn nhất: {means.max():.6f}"
    )


def test_chuan_hoa_do_lech_chuan(du_lieu):
    X, y = du_lieu
    X_train, X_test, _, _ = chia_train_test(X, y)
    _, X_train_sc, _ = chuan_hoa(X_train, X_test)
    stds = X_train_sc.std(axis=0)
    assert np.all(np.abs(stds - 1.0) < 1e-6), (
        f"Sau chuẩn hóa, std train phải ≈ 1. Lệch lớn nhất: {np.abs(stds - 1).max():.6f}"
    )


def test_chuan_hoa_khong_data_leakage(du_lieu):
    """Kiểm tra rằng scaler chỉ được fit trên X_train (không phải X_test)."""
    X, y = du_lieu
    X_train, X_test, _, _ = chia_train_test(X, y)
    scaler, X_train_sc, X_test_sc = chuan_hoa(X_train, X_test)
    # mean_ và scale_ của scaler phải khớp với X_train
    assert np.allclose(scaler.mean_, X_train.mean(axis=0), atol=1e-6), (
        "scaler.mean_ phải bằng mean của X_train, không phải X_test hay toàn bộ X"
    )


def test_chuan_hoa_tra_ve_ba_gia_tri(du_lieu):
    X, y = du_lieu
    X_train, X_test, _, _ = chia_train_test(X, y)
    ket_qua = chuan_hoa(X_train, X_test)
    assert len(ket_qua) == 3, "chuan_hoa() phải trả về đúng 3 giá trị: (scaler, X_train_sc, X_test_sc)"
