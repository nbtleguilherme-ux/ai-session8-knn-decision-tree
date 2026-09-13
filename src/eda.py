"""
src/eda.py — Các hàm khám phá dữ liệu (EDA) cho bộ dữ liệu Wine.

Sinh viên cần hoàn thiện các hàm được đánh dấu TODO.
Chạy tests/test_eda.py để kiểm tra kết quả.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import LabelEncoder, StandardScaler

sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Hàm hỗ trợ — KHÔNG cần chỉnh sửa
# ---------------------------------------------------------------------------

def tai_du_lieu():
    """
    Tải bộ dữ liệu wine và trả về DataFrame đã được xử lý cùng đối tượng gốc.

    Returns
    -------
    df : pd.DataFrame
        178 hàng × 15 cột (13 đặc trưng + 'nhan' + 'loai_ruou').
    wine : sklearn.utils.Bunch
        Đối tượng gốc (feature_names, target_names, data, target).
    """
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["nhan"] = wine.target
    df["loai_ruou"] = df["nhan"].map(
        {i: ten for i, ten in enumerate(wine.target_names)}
    )
    return df, wine


# ---------------------------------------------------------------------------
# TODO 1 — Kiểm tra giá trị thiếu
# ---------------------------------------------------------------------------

def kiem_tra_gia_tri_thieu(df: pd.DataFrame) -> pd.Series:
    """
    Đếm số giá trị thiếu (NaN) trong mỗi cột và in kết quả.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.Series : số lượng NaN mỗi cột (kể cả cột = 0).
    """
    so_nan = df.isnull().sum()
    if so_nan.any():
        print("Các cột có giá trị thiếu:")
        print(so_nan[so_nan > 0])
    else:
        print("Không có giá trị thiếu. Dữ liệu đầy đủ.")
    return so_nan


# ---------------------------------------------------------------------------
# TODO 2 — Phân phối nhãn
# ---------------------------------------------------------------------------

def ve_phan_phoi_nhan(df: pd.DataFrame, col: str = "loai_ruou") -> plt.Figure:
    """
    Vẽ biểu đồ cột và biểu đồ tròn thể hiện phân phối nhãn.

    Parameters
    ----------
    df  : pd.DataFrame
    col : tên cột nhãn dạng chuỗi (mặc định 'loai_ruou')

    Returns
    -------
    fig : matplotlib.figure.Figure — figure gồm 2 subplot cạnh nhau.

    Yêu cầu
    -------
    - Subplot trái  : biểu đồ cột, mỗi cột có nhãn số bên trên.
    - Subplot phải  : biểu đồ tròn với autopct='%1.1f%%'.
    - Cùng bảng màu sns 'muted'.
    - Tiêu đề chung : 'Phân phối nhãn trong tập dữ liệu'.
    """
    so_mau = df[col].value_counts()
    mau_sac = sns.color_palette("muted", len(so_mau))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(so_mau.index, so_mau.values, color=mau_sac)
    axes[0].set_title("Số mẫu theo nhóm rượu")
    axes[0].set_xlabel("Nhóm rượu")
    axes[0].set_ylabel("Số mẫu")
    for i, v in enumerate(so_mau.values):
        axes[0].text(i, v + 0.5, str(v), ha="center", fontweight="bold")

    axes[1].pie(
        so_mau.values,
        labels=so_mau.index,
        autopct="%1.1f%%",
        colors=mau_sac,
        startangle=90,
    )
    axes[1].set_title("Tỉ lệ phần trăm")

    plt.suptitle("Phân phối nhãn trong tập dữ liệu", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# TODO 3 — Histogram các đặc trưng theo nhóm
# ---------------------------------------------------------------------------

def ve_histogram_dac_trung(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "loai_ruou",
) -> plt.Figure:
    """
    Vẽ histogram chồng nhau (alpha=0.6) cho từng đặc trưng, phân theo nhóm.

    Parameters
    ----------
    df          : pd.DataFrame
    feature_cols: danh sách tên cột đặc trưng
    target_col  : tên cột nhóm (mặc định 'loai_ruou')

    Returns
    -------
    fig : matplotlib.figure.Figure — lưới subplot 3 cột, n_rows hàng.

    Yêu cầu
    -------
    - Lưới 3 cột, số hàng = ceil(len(feature_cols) / 3).
    - Mỗi subplot: histogram alpha=0.6, bins=15, edgecolor='white', legend.
    - Ẩn các subplot thừa.
    - Tiêu đề chung: 'Phân phối đặc trưng theo nhóm rượu'.
    """
    nhom_list = sorted(df[target_col].unique())
    n_cols = 3
    n_rows = -(-len(feature_cols) // n_cols)
    mau_sac = sns.color_palette("muted", len(nhom_list))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()

    for i, dac_trung in enumerate(feature_cols):
        for j, nhom in enumerate(nhom_list):
            subset = df[df[target_col] == nhom][dac_trung]
            axes[i].hist(
                subset, bins=15, alpha=0.6,
                color=mau_sac[j], label=nhom, edgecolor="white",
            )
        axes[i].set_title(dac_trung, fontsize=11)
        axes[i].set_xlabel("Giá trị")
        axes[i].set_ylabel("Số mẫu")
        axes[i].legend(fontsize=8)

    for k in range(len(feature_cols), len(axes)):
        axes[k].set_visible(False)

    plt.suptitle(
        "Phân phối đặc trưng theo nhóm rượu", fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# TODO 4 — Heatmap tương quan
# ---------------------------------------------------------------------------

def ve_heatmap_tuong_quan(df: pd.DataFrame, feature_cols: list) -> plt.Figure:
    """
    Vẽ heatmap ma trận tương quan (chỉ hiển thị tam giác dưới).

    Parameters
    ----------
    df           : pd.DataFrame
    feature_cols : danh sách tên cột đặc trưng

    Returns
    -------
    fig : matplotlib.figure.Figure

    Yêu cầu
    -------
    - Tính corr() trên df[feature_cols].
    - Tạo mask che tam giác trên bằng np.triu(..., dtype=bool).
    - Dùng sns.heatmap: annot=True, fmt='.2f', cmap='RdBu_r', center=0,
      vmin=-1, vmax=1, linewidths=0.5.
    - Tiêu đề: 'Ma trận tương quan giữa các đặc trưng'.
    """
    tuong_quan = df[feature_cols].corr()
    mat_na = np.triu(np.ones_like(tuong_quan, dtype=bool))

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        tuong_quan, mask=mat_na,
        annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Ma trận tương quan giữa các đặc trưng", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# TODO 5 — Top đặc trưng phân biệt nhóm
# ---------------------------------------------------------------------------

def top_dac_trung_phan_biet(
    df: pd.DataFrame, feature_cols: list, n: int = 5
) -> pd.Series:
    """
    Tìm n đặc trưng có tương quan tuyệt đối cao nhất với nhãn.

    Parameters
    ----------
    df           : pd.DataFrame (phải có cột 'nhan' kiểu int)
    feature_cols : danh sách tên cột đặc trưng
    n            : số đặc trưng cần trả về

    Returns
    -------
    pd.Series : index = tên đặc trưng, values = |tương quan|, sắp xếp giảm dần,
                độ dài = n.

    Yêu cầu
    -------
    - Mã hóa 'nhan' thành số bằng LabelEncoder.
    - Tính corr() của df[[*feature_cols, 'nhan_so']].
    - Lấy cột 'nhan_so', bỏ chính nó, abs(), sort_values, head(n).
    """
    le = LabelEncoder()
    df_tmp = df[list(feature_cols)].copy()
    df_tmp["nhan_so"] = le.fit_transform(df["nhan"])

    return (
        df_tmp.corr()["nhan_so"]
        .drop("nhan_so")
        .abs()
        .sort_values(ascending=False)
        .head(n)
    )
