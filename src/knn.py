"""
src/knn.py — Xây dựng và đánh giá mô hình K-Nearest Neighbors.

Sinh viên cần hoàn thiện các hàm được đánh dấu TODO.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------------
# TODO 1 — Tìm k tối ưu
# ---------------------------------------------------------------------------

def tim_k_tot_nhat(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k_max: int = 30,
) -> tuple:
    """
    Huấn luyện KNN với k từ 1 đến k_max, ghi lại độ chính xác và vẽ biểu đồ.

    Parameters
    ----------
    X_train, X_test : dữ liệu đã chuẩn hóa
    y_train, y_test : nhãn
    k_max           : giá trị k lớn nhất cần thử

    Returns
    -------
    k_tot_nhat     : int — giá trị k cho độ chính xác test cao nhất
    lich_su_train  : list — độ chính xác train cho mỗi k
    lich_su_test   : list — độ chính xác test cho mỗi k

    Yêu cầu
    -------
    - Lặp k từ 1 đến k_max (bao gồm k_max).
    - Với mỗi k: tạo KNeighborsClassifier(n_neighbors=k), fit, ghi score.
    - Vẽ biểu đồ đường (train + test) với đường dọc tại k tốt nhất.
    - Trục x: 'k (số lượng hàng xóm)', trục y: 'Độ chính xác'.
    - Tiêu đề: 'KNN: Độ chính xác theo k'.
    """
    dai_k = range(1, k_max + 1)
    lich_su_train = []
    lich_su_test = []

    for k in dai_k:
        mo_hinh = KNeighborsClassifier(n_neighbors=k)
        mo_hinh.fit(X_train, y_train)
        lich_su_train.append(mo_hinh.score(X_train, y_train))
        lich_su_test.append(mo_hinh.score(X_test, y_test))

    k_tot_nhat = 1 + int(np.argmax(lich_su_test))

    plt.figure(figsize=(12, 5))
    plt.plot(dai_k, lich_su_train, "b-o", markersize=5, label="Tập huấn luyện")
    plt.plot(dai_k, lich_su_test, "r-s", markersize=5, label="Tập kiểm tra")
    plt.axvline(
        k_tot_nhat, color="green", linestyle="--", linewidth=1.5,
        label=f"k tốt nhất = {k_tot_nhat}",
    )
    plt.xlabel("k (số lượng hàng xóm)")
    plt.ylabel("Độ chính xác")
    plt.title("KNN: Độ chính xác theo k")
    plt.xticks(list(dai_k))
    plt.legend()
    plt.tight_layout()
    plt.show()

    return k_tot_nhat, lich_su_train, lich_su_test


# ---------------------------------------------------------------------------
# TODO 2 — Huấn luyện mô hình KNN
# ---------------------------------------------------------------------------

def huan_luyen_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
) -> KNeighborsClassifier:
    """
    Tạo và huấn luyện KNeighborsClassifier với k đã cho.

    Parameters
    ----------
    X_train : dữ liệu huấn luyện đã chuẩn hóa
    y_train : nhãn huấn luyện
    k       : số lượng hàng xóm

    Returns
    -------
    KNeighborsClassifier đã được fit.
    """
    mo_hinh = KNeighborsClassifier(n_neighbors=k)
    mo_hinh.fit(X_train, y_train)
    return mo_hinh


# ---------------------------------------------------------------------------
# TODO 3 — Đánh giá mô hình
# ---------------------------------------------------------------------------

def danh_gia_mo_hinh(
    mo_hinh,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    tieu_de: str = "Mô hình",
) -> dict:
    """
    Đánh giá mô hình phân loại: in báo cáo và vẽ ma trận nhầm lẫn.

    Parameters
    ----------
    mo_hinh     : mô hình đã huấn luyện (KNN hoặc DT)
    X_test      : dữ liệu kiểm tra
    y_test      : nhãn kiểm tra
    class_names : danh sách tên nhóm
    tieu_de     : chuỗi xuất hiện trong tiêu đề biểu đồ

    Returns
    -------
    dict với các khóa:
        'do_chinh_xac' : float
        'du_doan'      : np.ndarray
        'ma_tran'      : np.ndarray (confusion matrix)

    Yêu cầu
    -------
    - Dự đoán bằng mo_hinh.predict(X_test).
    - In classification_report với target_names=class_names.
    - Vẽ ConfusionMatrixDisplay (figsize=(6,5)).
    - Tiêu đề biểu đồ: f'Ma trận nhầm lẫn — {tieu_de}'.
    """
    du_doan = mo_hinh.predict(X_test)
    do_cx = accuracy_score(y_test, du_doan)
    ma_tran = confusion_matrix(y_test, du_doan)

    print(f"=== Báo cáo phân loại — {tieu_de} ===")
    print(classification_report(y_test, du_doan, target_names=class_names))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=ma_tran, display_labels=class_names).plot(
        cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Ma trận nhầm lẫn — {tieu_de}", fontsize=13)
    plt.tight_layout()
    plt.show()

    return {"do_chinh_xac": do_cx, "du_doan": du_doan, "ma_tran": ma_tran}


# ---------------------------------------------------------------------------
# Hàm hỗ trợ — KHÔNG cần chỉnh sửa
# ---------------------------------------------------------------------------

def kiem_dinh_cheo(mo_hinh, X: np.ndarray, y: np.ndarray, cv: int = 5) -> np.ndarray:
    """Kiểm định chéo k-fold, trả về mảng điểm accuracy."""
    diem = cross_val_score(mo_hinh, X, y, cv=cv, scoring="accuracy")
    print(f"Điểm CV ({cv}-fold) : {diem.round(4)}")
    print(f"Trung bình           : {diem.mean():.4f}")
    print(f"Độ lệch chuẩn        : {diem.std():.4f}")
    return diem
