"""
src/decision_tree.py — Xây dựng và đánh giá mô hình Decision Tree.

Sinh viên cần hoàn thiện các hàm được đánh dấu TODO.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------------
# TODO 1 — Tìm max_depth tối ưu
# ---------------------------------------------------------------------------

def tim_do_sau_tot_nhat(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    do_sau_max: int = 14,
    random_state: int = 42,
) -> tuple:
    """
    Thử max_depth từ 1 đến do_sau_max, vẽ biểu đồ và trả về depth tốt nhất.

    Parameters
    ----------
    X_train, X_test : dữ liệu GỐC (Decision Tree không cần chuẩn hóa)
    y_train, y_test : nhãn
    do_sau_max      : max_depth lớn nhất cần thử
    random_state    : hạt giống

    Returns
    -------
    do_sau_tot_nhat : int
    lich_su_train   : list[float]
    lich_su_test    : list[float]

    Yêu cầu
    -------
    - Lặp depth từ 1 đến do_sau_max (bao gồm do_sau_max).
    - Với mỗi depth: DecisionTreeClassifier(max_depth=depth, random_state=...).
    - Fit trên X_train (dữ liệu gốc, không chuẩn hóa).
    - Vẽ biểu đồ đường tương tự KNN.
    - Tiêu đề: 'Decision Tree: Độ chính xác theo max_depth'.
    """
    dai_depth = range(1, do_sau_max + 1)
    lich_su_train = []
    lich_su_test = []

    for d in dai_depth:
        dt = DecisionTreeClassifier(max_depth=d, random_state=random_state)
        dt.fit(X_train, y_train)
        lich_su_train.append(dt.score(X_train, y_train))
        lich_su_test.append(dt.score(X_test, y_test))

    do_sau_tot_nhat = 1 + int(np.argmax(lich_su_test))

    plt.figure(figsize=(12, 5))
    plt.plot(dai_depth, lich_su_train, "b-o", markersize=5, label="Tập huấn luyện")
    plt.plot(dai_depth, lich_su_test, "r-s", markersize=5, label="Tập kiểm tra")
    plt.axvline(
        do_sau_tot_nhat, color="green", linestyle="--", linewidth=1.5,
        label=f"max_depth tốt nhất = {do_sau_tot_nhat}",
    )
    plt.xlabel("max_depth")
    plt.ylabel("Độ chính xác")
    plt.title("Decision Tree: Độ chính xác theo max_depth")
    plt.xticks(list(dai_depth))
    plt.legend()
    plt.tight_layout()
    plt.show()

    return do_sau_tot_nhat, lich_su_train, lich_su_test


# ---------------------------------------------------------------------------
# TODO 2 — Huấn luyện Decision Tree
# ---------------------------------------------------------------------------

def huan_luyen_cay(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """
    Tạo và huấn luyện DecisionTreeClassifier.

    Parameters
    ----------
    X_train      : dữ liệu huấn luyện GỐC (không chuẩn hóa)
    y_train      : nhãn huấn luyện
    max_depth    : độ sâu tối đa
    random_state : hạt giống

    Returns
    -------
    DecisionTreeClassifier đã được fit.
    """
    mo_hinh = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    mo_hinh.fit(X_train, y_train)
    return mo_hinh


# ---------------------------------------------------------------------------
# TODO 3 — Trực quan hóa cây quyết định
# ---------------------------------------------------------------------------

def ve_cay_quyet_dinh(
    mo_hinh: DecisionTreeClassifier,
    feature_names: list,
    class_names: list,
) -> plt.Figure:
    """
    Vẽ cây quyết định với tất cả thông tin phân nhánh.

    Parameters
    ----------
    mo_hinh      : DecisionTreeClassifier đã huấn luyện
    feature_names: danh sách tên đặc trưng (wine.feature_names)
    class_names  : danh sách tên nhóm (wine.target_names)

    Returns
    -------
    fig : matplotlib.figure.Figure

    Yêu cầu
    -------
    - Dùng plot_tree() với: feature_names, class_names,
      filled=True, rounded=True, fontsize=9.
    - figsize=(20, 8).
    - Tiêu đề: f'Cây quyết định (max_depth={mo_hinh.max_depth})'.
    """
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        mo_hinh,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
    )
    ax.set_title(
        f"Cây quyết định (max_depth={mo_hinh.max_depth})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# TODO 4 — Tầm quan trọng của đặc trưng
# ---------------------------------------------------------------------------

def ve_tam_quan_trong(
    mo_hinh: DecisionTreeClassifier,
    feature_names: list,
) -> pd.Series:
    """
    Vẽ biểu đồ cột ngang (barh) thể hiện tầm quan trọng Gini của từng đặc trưng.

    Parameters
    ----------
    mo_hinh      : DecisionTreeClassifier đã huấn luyện
    feature_names: danh sách tên đặc trưng

    Returns
    -------
    pd.Series : index=tên đặc trưng, values=feature_importances_, sorted ascending.

    Yêu cầu
    -------
    - Lấy mo_hinh.feature_importances_.
    - Tạo pd.Series với index=feature_names, sắp xếp ascending=True.
    - Vẽ barh với palette 'muted'.
    - Nhãn trục x: 'Chỉ số Gini Importance'.
    - Tiêu đề: 'Tầm quan trọng của đặc trưng (Decision Tree)'.
    - In top 3 đặc trưng quan trọng nhất.
    """
    tam_quan_trong = pd.Series(
        mo_hinh.feature_importances_, index=feature_names
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    tam_quan_trong.plot(
        kind="barh", ax=ax,
        color=sns.color_palette("muted", len(tam_quan_trong)),
    )
    ax.set_xlabel("Chỉ số Gini Importance")
    ax.set_title("Tầm quan trọng của đặc trưng (Decision Tree)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print("Top 3 đặc trưng quan trọng nhất:")
    for ten, val in tam_quan_trong.sort_values(ascending=False).head(3).items():
        print(f"  {ten}: {val:.4f}")

    return tam_quan_trong


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
