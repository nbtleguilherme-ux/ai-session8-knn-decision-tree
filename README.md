# Buổi 8 — Hệ thống gợi ý rượu vang với KNN và Decision Tree

## Mô tả dự án

Bạn sẽ xây dựng một hệ thống gợi ý rượu vang: cho trước thành phần hóa học của một chai rượu, hệ thống xác định loại rượu đó thuộc nhóm nào trong 3 nhóm giống nho.

Dữ liệu: `sklearn.datasets.load_wine` — 178 mẫu, 13 đặc trưng hóa học.

---

## Cấu trúc dự án

```
session8/
├── src/                        ← Code bạn cần hoàn thiện
│   ├── eda.py                  ← Hàm khám phá dữ liệu         (5 TODO)
│   ├── preprocessing.py        ← Tiền xử lý dữ liệu           (2 TODO)
│   ├── knn.py                  ← Mô hình KNN                  (3 TODO)
│   ├── decision_tree.py        ← Mô hình Decision Tree        (4 TODO)
│   └── recommender.py          ← Hàm gợi ý rượu vang         (1 TODO)
├── notebooks/                  ← Chạy theo thứ tự
│   ├── 01_EDA.ipynb
│   ├── 02_KNN.ipynb
│   ├── 03_DecisionTree.ipynb
│   └── 04_Goi_Y_So_Sanh.ipynb
├── tests/                      ← Kiểm tra bài làm của bạn
│   ├── test_preprocessing.py
│   ├── test_knn.py
│   └── test_decision_tree.py
└── requirements.txt
```

---

## Hướng dẫn thực hiện

### 1. Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### 2. Quy trình làm bài

Làm lần lượt theo thứ tự:

| Bước | File cần chỉnh sửa | Notebook kiểm tra |
|------|-------------------|-------------------|
| 1 | `src/eda.py` | `notebooks/01_EDA.ipynb` |
| 2 | `src/preprocessing.py` | `notebooks/02_KNN.ipynb` |
| 3 | `src/knn.py` | `notebooks/02_KNN.ipynb` |
| 4 | `src/decision_tree.py` | `notebooks/03_DecisionTree.ipynb` |
| 5 | `src/recommender.py` | `notebooks/04_Goi_Y_So_Sanh.ipynb` |

### 3. Chạy notebook

```bash
cd notebooks
jupyter notebook
```

### 4. Kiểm tra bài làm bằng tests

```bash
# Từ thư mục gốc session8/
pytest tests/ -v
```

Tất cả tests phải **PASSED** trước khi nộp bài.

---

## Quy tắc quan trọng

- **Chỉ được chỉnh sửa** các file trong `src/`.
- **Không được thay đổi** notebooks, tests, hoặc `requirements.txt`.
- Mỗi hàm có docstring giải thích rõ **đầu vào**, **đầu ra**, và **yêu cầu**.
- Nếu một ô notebook báo `NotImplementedError` → bạn chưa hoàn thiện hàm tương ứng.

---

## Gợi ý khi bị kẹt

1. Đọc kỹ docstring của hàm — nó nêu rõ từng bước cần làm.
2. Kiểm tra phần **Yêu cầu** trong docstring.
3. Chạy test cụ thể để xem lỗi chi tiết:
   ```bash
   pytest tests/test_preprocessing.py::test_chuan_hoa_khong_data_leakage -v
   ```
4. Xem lại slide lý thuyết hoặc hỏi giảng viên.
