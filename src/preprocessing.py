"""
src/preprocessing.py — Tiền xử lý dữ liệu trước khi huấn luyện mô hình.

Sinh viên cần hoàn thiện các hàm được đánh dấu TODO.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# TODO 1 — Chia tập dữ liệu
# ---------------------------------------------------------------------------

def chia_train_test(
    X: np.ndarray,
    y: np.ndarray,
    ti_le_test: float = 0.2,
    random_state: int = 42,
):
    """
    Chia dữ liệu thành tập huấn luyện và tập kiểm tra với stratify.

    Parameters
    ----------
    X            : ma trận đặc trưng, shape (n_mau, n_dac_trung)
    y            : vector nhãn, shape (n_mau,)
    ti_le_test   : tỉ lệ dữ liệu dành cho tập test (mặc định 0.2)
    random_state : hạt giống ngẫu nhiên

    Returns
    -------
    X_train, X_test, y_train, y_test — bốn mảng numpy.

    Yêu cầu
    -------
    - Dùng train_test_split với stratify=y để giữ nguyên tỉ lệ nhóm.
    - Không chuẩn hóa ở đây — hàm này chỉ chia dữ liệu.
    """
    return train_test_split(
        X, y,
        test_size=ti_le_test,
        random_state=random_state,
        stratify=y,
    )


# ---------------------------------------------------------------------------
# TODO 2 — Chuẩn hóa đặc trưng
# ---------------------------------------------------------------------------

def chuan_hoa(X_train: np.ndarray, X_test: np.ndarray):
    """
    Chuẩn hóa đặc trưng bằng StandardScaler.

    Parameters
    ----------
    X_train : dữ liệu huấn luyện (chưa chuẩn hóa)
    X_test  : dữ liệu kiểm tra  (chưa chuẩn hóa)

    Returns
    -------
    scaler         : StandardScaler đã được fit trên X_train
    X_train_scaled : X_train sau khi chuẩn hóa
    X_test_scaled  : X_test sau khi chuẩn hóa

    Yêu cầu
    -------
    - Tạo StandardScaler().
    - fit_transform trên X_train  → X_train_scaled.
    - Chỉ transform (KHÔNG fit lại) trên X_test → X_test_scaled.
    - Trả về (scaler, X_train_scaled, X_test_scaled).

    Lưu ý quan trọng
    ----------------
    Nếu fit scaler trên cả X_test, bạn đã gây ra "data leakage" —
    thông tin từ tập test rò rỉ vào quá trình huấn luyện.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled
