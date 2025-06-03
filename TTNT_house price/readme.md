# 🏠 Dự Án Dự Đoán Giá Nhà - House Price Prediction

## 📌 Mô tả

Dự án sử dụng mô hình học máy Random Forest để dự đoán giá nhà dựa trên 10 đặc trưng có tương quan cao nhất với giá bán (SalePrice). Dự án sử dụng Streamlit để tạo giao diện người dùng trực quan, cho phép nhập thông tin và hiển thị kết quả dự đoán.

## 📁 Cấu trúc thư mục

```
├── data/                        # Dữ liệu gốc và dữ liệu mẫu
│   └── train.csv
│
├── models/                      # Mô hình đã huấn luyện
│   └── rf_model.pkl
│
├── notebooks/                  # Phân tích và trực quan dữ liệu
│   └── EDA_Preprocessing.ipynb
│
├── outputs/                    # Lưu metric đánh giá mô hình
│   └── metrics.txt
│
├── app/                        # Ứng dụng Streamlit
│   └── streamlit_app.py
│
├── src/                        # Mã nguồn huấn luyện
│   └── train_model.py
│
├── requirements.txt            # Thư viện cần thiết
└── README.md                   # Hướng dẫn dự án
```

## 📊 Các bước thực hiện

1. Phân tích tương quan dữ liệu, chọn 10 đặc trưng mạnh nhất.
2. Tiền xử lý dữ liệu: Impute missing values, StandardScaler, OneHotEncoder.
3. Huấn luyện mô hình với RandomForestRegressor và GridSearchCV.
4. Đánh giá mô hình bằng MSE, R2.
5. Lưu mô hình vào thư mục `models/`.
6. Tạo ứng dụng Streamlit để nhập thông số và dự đoán.

## 🚀 Cách chạy

### 1. Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

### 2. Huấn luyện mô hình (nếu muốn huấn luyện lại):

```bash
python src/train_model.py
```

### 3. Chạy ứng dụng Streamlit:

```bash
streamlit run app/streamlit_app.py
```

## 📈 Độ tin cậy mô hình

Dự án hiện hiển thị độ tin cậy giả định (R² \~ 85%) để người dùng có cảm nhận về chất lượng mô hình. Nếu đầu vào là tất cả 0, dự đoán sẽ trả về 0 và cảnh báo độ tin cậy rất thấp.

## 🧪 Notebook EDA

File `show_data.ipynb` giúp trực quan dữ liệu trước và sau tiền xử lý, sử dụng biểu đồ tương quan và PCA.

## 🛠 Công nghệ sử dụng

* Python 3
* Scikit-learn
* Pandas / NumPy
* Streamlit
* Matplotlib / Seaborn

## ✅ Tác giả

* 💼 Tên: \[Nhóm 13]
* 📅 Năm: 2025
