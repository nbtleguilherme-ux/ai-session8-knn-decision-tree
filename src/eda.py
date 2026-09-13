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


def thong_ke_mo_ta(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Trả về bảng thống kê mô tả (transpose) đã làm tròn 4 chữ số.

    Parameters
    ----------
    df           : pd.DataFrame
    feature_cols : danh sách tên cột đặc trưng

    Returns
    -------
    pd.DataFrame : describe().T.round(4)
    """
    stats = df[feature_cols].describe().T.round(4)
    return stats


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

    Yêu cầu
    -------
    - Dùng df.isnull().sum() để đếm NaN mỗi cột.
    - Nếu có NaN: in các cột bị thiếu; nếu không: in "Không có giá trị thiếu."
    - Trả về pd.Series kết quả (kể cả cột có giá trị 0).
    """
    # TODO: Triển khai hàm này
    # Bước 1: Tính số NaN mỗi cột
    # Bước 2: Kiểm tra và in kết quả
    # Bước 3: return Series
    raise NotImplementedError("TODO 1: Hoàn thiện hàm kiem_tra_gia_tri_thieu()")


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
    # TODO: Triển khai hàm này
    # Bước 1: Đếm số mẫu mỗi nhóm: df[col].value_counts()
    # Bước 2: Tạo bảng màu: sns.color_palette("muted", ...)
    # Bước 3: Tạo figure với 2 subplot: plt.subplots(1, 2, figsize=(12, 5))
    # Bước 4: Vẽ bar chart ở axes[0] — thêm nhãn số trên mỗi cột
    # Bước 5: Vẽ pie chart ở axes[1] — autopct='%1.1f%%'
    # Bước 6: plt.suptitle(...), plt.tight_layout(), return fig
    raise NotImplementedError("TODO 2: Hoàn thiện hàm ve_phan_phoi_nhan()")


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
    # TODO: Triển khai hàm này
    # Bước 1: Lấy danh sách nhóm: df[target_col].unique()
    # Bước 2: Tính n_rows = -(-len(feature_cols) // 3)  (ceiling division)
    # Bước 3: Tạo figure lưới: plt.subplots(n_rows, 3, figsize=(18, n_rows*4))
    # Bước 4: axes = axes.flatten()
    # Bước 5: Lặp qua từng đặc trưng, với mỗi đặc trưng lặp qua từng nhóm → vẽ hist
    # Bước 6: Ẩn subplot thừa: axes[k].set_visible(False)
    # Bước 7: plt.suptitle(...), plt.tight_layout(), return fig
    raise NotImplementedError("TODO 3: Hoàn thiện hàm ve_histogram_dac_trung()")


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
    # TODO: Triển khai hàm này
    # Bước 1: tuong_quan = df[feature_cols].corr()
    # Bước 2: mat_na = np.triu(np.ones_like(tuong_quan, dtype=bool))
    # Bước 3: Tạo figure: fig, ax = plt.subplots(figsize=(13, 10))
    # Bước 4: sns.heatmap(tuong_quan, mask=mat_na, annot=True, fmt='.2f',
    #                     cmap='RdBu_r', center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    # Bước 5: Đặt tiêu đề, tight_layout, return fig
    raise NotImplementedError("TODO 4: Hoàn thiện hàm ve_heatmap_tuong_quan()")


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
    # TODO: Triển khai hàm này
    # Bước 1: le = LabelEncoder()
    #         df_tmp = df[list(feature_cols)].copy()
    #         df_tmp["nhan_so"] = le.fit_transform(df["nhan"])
    # Bước 2: Tính corr() của df_tmp
    # Bước 3: Lấy cột "nhan_so", drop("nhan_so"), .abs(), .sort_values(ascending=False), .head(n)
    # Bước 4: return kết quả
    raise NotImplementedError("TODO 5: Hoàn thiện hàm top_dac_trung_phan_biet()")
