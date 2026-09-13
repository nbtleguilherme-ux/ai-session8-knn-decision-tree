"""
src/recommender.py — Hệ thống gợi ý rượu vang.

Sinh viên cần hoàn thiện hàm goi_y_ruou().
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# TODO — Hàm gợi ý rượu vang
# ---------------------------------------------------------------------------

def goi_y_ruou(
    thanh_phan: dict,
    mo_hinh,
    scaler: StandardScaler,
    feature_names: list,
    class_names: list,
) -> str:
    """
    Gợi ý nhóm rượu dựa trên thành phần hóa học.

    Parameters
    ----------
    thanh_phan   : dict — key khớp với feature_names, value là giá trị thực.
    mo_hinh      : mô hình đã huấn luyện (KNN hoặc Decision Tree).
    scaler       : StandardScaler đã fit trên tập train.
                   Nếu mo_hinh là DecisionTreeClassifier → không dùng scaler.
    feature_names: danh sách tên đặc trưng theo thứ tự.
    class_names  : danh sách tên nhóm rượu.

    Returns
    -------
    str : ví dụ 'Gợi ý: class_0  (độ tự tin: 85.7%)'

    Yêu cầu
    -------
    1. Chuyển thanh_phan thành mảng 2D shape (1, n_dac_trung) theo thứ tự feature_names.
    2. Nếu mo_hinh là KNeighborsClassifier:
       - Chuẩn hóa mẫu bằng scaler.transform().
       - Dự đoán bằng mo_hinh.predict() và mo_hinh.predict_proba().
    3. Nếu mo_hinh là DecisionTreeClassifier:
       - KHÔNG chuẩn hóa.
       - Dự đoán trực tiếp.
    4. Trả về chuỗi: f'Gợi ý: {tên_nhóm}  (độ tự tin: {xác_suất:.1%})'.

    Lưu ý
    -----
    predict() trả về mảng → lấy phần tử [0].
    predict_proba() trả về ma trận xác suất → lấy hàng [0], sau đó cột [du_doan].
    """
    mau = np.array([[thanh_phan[f] for f in feature_names]])

    if isinstance(mo_hinh, KNeighborsClassifier):
        mau_xu_ly = scaler.transform(mau)
    else:
        mau_xu_ly = mau

    du_doan = mo_hinh.predict(mau_xu_ly)[0]
    xac_suat = mo_hinh.predict_proba(mau_xu_ly)[0]

    ten_nhom = class_names[du_doan]
    do_tin_tuong = xac_suat[du_doan]
    return f"Gợi ý: {ten_nhom}  (độ tự tin: {do_tin_tuong:.1%})"
