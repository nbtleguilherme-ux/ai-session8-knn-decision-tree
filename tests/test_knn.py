"""
tests/test_knn.py
Chạy: pytest tests/test_knn.py -v
"""
import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import chia_train_test, chuan_hoa
from src.knn import tim_k_tot_nhat, huan_luyen_knn, danh_gia_mo_hinh


@pytest.fixture
def du_lieu_da_xu_ly():
    wine = load_wine()
    X_tr, X_te, y_tr, y_te = chia_train_test(wine.data, wine.target)
    _, X_tr_sc, X_te_sc = chuan_hoa(X_tr, X_te)
    return X_tr_sc, X_te_sc, y_tr, y_te, wine.target_names


# ------------------------------------------------------------------
# Kiểm tra tim_k_tot_nhat
# ------------------------------------------------------------------

def test_tim_k_tra_ve_tuple(du_lieu_da_xu_ly):
    X_tr, X_te, y_tr, y_te, _ = du_lieu_da_xu_ly
    ket_qua = tim_k_tot_nhat(X_tr, X_te, y_tr, y_te, k_max=10)
    assert isinstance(ket_qua, tuple) and len(ket_qua) == 3, (
        "tim_k_tot_nhat() phải trả về tuple (k_tot_nhat, lich_su_train, lich_su_test)"
    )


def test_tim_k_nam_trong_pham_vi(du_lieu_da_xu_ly):
    X_tr, X_te, y_tr, y_te, _ = du_lieu_da_xu_ly
    k, _, _ = tim_k_tot_nhat(X_tr, X_te, y_tr, y_te, k_max=20)
    assert 1 <= k <= 20, f"k tốt nhất phải nằm trong [1, 20], nhận được {k}"


def test_tim_k_lich_su_do_dai(du_lieu_da_xu_ly):
    X_tr, X_te, y_tr, y_te, _ = du_lieu_da_xu_ly
    K_MAX = 15
    _, ls_train, ls_test = tim_k_tot_nhat(X_tr, X_te, y_tr, y_te, k_max=K_MAX)
    assert len(ls_train) == K_MAX and len(ls_test) == K_MAX, (
        f"Lịch sử accuracy phải có đúng {K_MAX} phần tử"
    )


def test_tim_k_do_chinh_xac_hop_ly(du_lieu_da_xu_ly):
    X_tr, X_te, y_tr, y_te, _ = du_lieu_da_xu_ly
    _, _, ls_test = tim_k_tot_nhat(X_tr, X_te, y_tr, y_te, k_max=10)
    assert max(ls_test) >= 0.85, (
        f"KNN trên wine dataset nên đạt ≥85% test accuracy, nhận được {max(ls_test):.2f}"
    )


# ------------------------------------------------------------------
# Kiểm tra huan_luyen_knn
# ------------------------------------------------------------------

def test_huan_luyen_knn_tra_ve_knn(du_lieu_da_xu_ly):
    X_tr, _, y_tr, _, _ = du_lieu_da_xu_ly
    mo_hinh = huan_luyen_knn(X_tr, y_tr, k=5)
    assert isinstance(mo_hinh, KNeighborsClassifier), (
        "huan_luyen_knn() phải trả về KNeighborsClassifier"
    )


def test_huan_luyen_knn_da_fit(du_lieu_da_xu_ly):
    X_tr, X_te, y_tr, _, _ = du_lieu_da_xu_ly
    mo_hinh = huan_luyen_knn(X_tr, y_tr, k=5)
    # mô hình đã fit thì predict không được raise error
    try:
        mo_hinh.predict(X_te[:3])
    except Exception as e:
        pytest.fail(f"Mô hình chưa được fit đúng: {e}")


def test_huan_luyen_knn_k_dung(du_lieu_da_xu_ly):
    X_tr, _, y_tr, _, _ = du_lieu_da_xu_ly
    k_mong_muon = 7
    mo_hinh = huan_luyen_knn(X_tr, y_tr, k=k_mong_muon)
    assert mo_hinh.n_neighbors == k_mong_muon, (
        f"n_neighbors phải là {k_mong_muon}, nhận được {mo_hinh.n_neighbors}"
    )


# ------------------------------------------------------------------
# Kiểm tra danh_gia_mo_hinh
# ------------------------------------------------------------------

def test_danh_gia_tra_ve_dict(du_lieu_da_xu_ly):
    X_tr, X_te, y_tr, y_te, class_names = du_lieu_da_xu_ly
    mo_hinh = huan_luyen_knn(X_tr, y_tr, k=5)
    ket_qua = danh_gia_mo_hinh(mo_hinh, X_te, y_te, class_names, "test")
    assert isinstance(ket_qua, dict), "danh_gia_mo_hinh() phải trả về dict"


def test_danh_gia_cac_khoa(du_lieu_da_xu_ly):
    X_tr, X_te, y_tr, y_te, class_names = du_lieu_da_xu_ly
    mo_hinh = huan_luyen_knn(X_tr, y_tr, k=5)
    ket_qua = danh_gia_mo_hinh(mo_hinh, X_te, y_te, class_names, "test")
    for khoa in ("do_chinh_xac", "du_doan", "ma_tran"):
        assert khoa in ket_qua, f"Khóa '{khoa}' phải có trong dict trả về"


def test_danh_gia_do_chinh_xac_hop_ly(du_lieu_da_xu_ly):
    X_tr, X_te, y_tr, y_te, class_names = du_lieu_da_xu_ly
    mo_hinh = huan_luyen_knn(X_tr, y_tr, k=5)
    ket_qua = danh_gia_mo_hinh(mo_hinh, X_te, y_te, class_names, "test")
    assert ket_qua["do_chinh_xac"] >= 0.80, (
        f"Độ chính xác KNN nên ≥80%, nhận được {ket_qua['do_chinh_xac']:.2f}"
    )
