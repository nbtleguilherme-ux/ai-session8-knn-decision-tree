"""
tests/test_decision_tree.py
Chạy: pytest tests/test_decision_tree.py -v
"""
import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import chia_train_test
from src.decision_tree import (
    tim_do_sau_tot_nhat, huan_luyen_cay,
    ve_cay_quyet_dinh, ve_tam_quan_trong,
)


@pytest.fixture
def du_lieu():
    wine = load_wine()
    X_tr, X_te, y_tr, y_te = chia_train_test(wine.data, wine.target)
    return X_tr, X_te, y_tr, y_te, wine.feature_names, wine.target_names


# ------------------------------------------------------------------
# Kiểm tra tim_do_sau_tot_nhat
# ------------------------------------------------------------------

def test_tim_do_sau_tra_ve_tuple(du_lieu):
    X_tr, X_te, y_tr, y_te, *_ = du_lieu
    ket_qua = tim_do_sau_tot_nhat(X_tr, X_te, y_tr, y_te, do_sau_max=8)
    assert isinstance(ket_qua, tuple) and len(ket_qua) == 3


def test_tim_do_sau_nam_trong_pham_vi(du_lieu):
    X_tr, X_te, y_tr, y_te, *_ = du_lieu
    d, _, _ = tim_do_sau_tot_nhat(X_tr, X_te, y_tr, y_te, do_sau_max=10)
    assert 1 <= d <= 10, f"max_depth tốt nhất phải trong [1, 10], nhận được {d}"


def test_tim_do_sau_lich_su_dung_do_dai(du_lieu):
    X_tr, X_te, y_tr, y_te, *_ = du_lieu
    D_MAX = 8
    _, ls_tr, ls_te = tim_do_sau_tot_nhat(X_tr, X_te, y_tr, y_te, do_sau_max=D_MAX)
    assert len(ls_tr) == D_MAX and len(ls_te) == D_MAX


def test_tim_do_sau_do_chinh_xac_hop_ly(du_lieu):
    X_tr, X_te, y_tr, y_te, *_ = du_lieu
    _, _, ls_te = tim_do_sau_tot_nhat(X_tr, X_te, y_tr, y_te, do_sau_max=10)
    assert max(ls_te) >= 0.85, (
        f"DT nên đạt ≥85% test accuracy, nhận được {max(ls_te):.2f}"
    )


# ------------------------------------------------------------------
# Kiểm tra huan_luyen_cay
# ------------------------------------------------------------------

def test_huan_luyen_cay_tra_ve_dt(du_lieu):
    X_tr, _, y_tr, *_ = du_lieu
    mo_hinh = huan_luyen_cay(X_tr, y_tr, max_depth=3)
    assert isinstance(mo_hinh, DecisionTreeClassifier)


def test_huan_luyen_cay_max_depth(du_lieu):
    X_tr, _, y_tr, *_ = du_lieu
    mo_hinh = huan_luyen_cay(X_tr, y_tr, max_depth=4)
    assert mo_hinh.max_depth == 4, f"max_depth phải là 4, nhận được {mo_hinh.max_depth}"


def test_huan_luyen_cay_da_fit(du_lieu):
    X_tr, X_te, y_tr, *_ = du_lieu
    mo_hinh = huan_luyen_cay(X_tr, y_tr, max_depth=3)
    try:
        mo_hinh.predict(X_te[:3])
    except Exception as e:
        pytest.fail(f"Mô hình chưa được fit: {e}")


# ------------------------------------------------------------------
# Kiểm tra ve_cay_quyet_dinh
# ------------------------------------------------------------------

def test_ve_cay_tra_ve_figure(du_lieu):
    import matplotlib.pyplot as plt
    X_tr, _, y_tr, _, feat, cls = du_lieu
    mo_hinh = huan_luyen_cay(X_tr, y_tr, max_depth=3)
    fig = ve_cay_quyet_dinh(mo_hinh, feat, cls)
    assert hasattr(fig, "savefig"), "ve_cay_quyet_dinh() phải trả về matplotlib.figure.Figure"
    plt.close("all")


# ------------------------------------------------------------------
# Kiểm tra ve_tam_quan_trong
# ------------------------------------------------------------------

def test_ve_tam_quan_trong_tra_ve_series(du_lieu):
    import pandas as pd
    import matplotlib.pyplot as plt
    X_tr, _, y_tr, _, feat, _ = du_lieu
    mo_hinh = huan_luyen_cay(X_tr, y_tr, max_depth=3)
    ket_qua = ve_tam_quan_trong(mo_hinh, feat)
    assert isinstance(ket_qua, pd.Series), "ve_tam_quan_trong() phải trả về pd.Series"
    plt.close("all")


def test_ve_tam_quan_trong_index(du_lieu):
    import matplotlib.pyplot as plt
    X_tr, _, y_tr, _, feat, _ = du_lieu
    mo_hinh = huan_luyen_cay(X_tr, y_tr, max_depth=3)
    series = ve_tam_quan_trong(mo_hinh, feat)
    assert set(series.index) == set(feat), "Index của Series phải là feature_names"
    plt.close("all")


def test_ve_tam_quan_trong_tong_bang_1(du_lieu):
    import matplotlib.pyplot as plt
    X_tr, _, y_tr, _, feat, _ = du_lieu
    mo_hinh = huan_luyen_cay(X_tr, y_tr, max_depth=3)
    series = ve_tam_quan_trong(mo_hinh, feat)
    assert abs(series.sum() - 1.0) < 1e-6, (
        f"Tổng feature importances phải = 1.0, nhận được {series.sum():.6f}"
    )
    plt.close("all")
