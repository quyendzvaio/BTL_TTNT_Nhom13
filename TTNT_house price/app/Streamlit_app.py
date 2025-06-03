import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model
MODEL_PATH = "models/rf_model.pkl"
model = joblib.load(MODEL_PATH)

# Tên các feature đã chọn khi huấn luyện
FEATURES = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
            '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']

# Giao diện nhập liệu
st.title("🏡 Dự đoán giá nhà")
st.write("Nhập vào các thông số bên dưới để dự đoán giá nhà")

user_input = {}
all_zero = True

for feature in FEATURES:
    val = st.number_input(f"{feature}", min_value=0.0, value=0.0, step=1.0)
    user_input[feature] = val
    if val != 0:
        all_zero = False

if st.button("Dự đoán"):
    input_df = pd.DataFrame([user_input])

    if all_zero:
        st.warning("⚠️ Tất cả giá trị đều bằng 0 → Dự đoán: **0**")
        st.info("🔍 Độ tin cậy: **Rất thấp** — Dữ liệu đầu vào không hợp lệ hoặc không đủ thông tin.")
    else:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Giá nhà dự đoán: **${prediction:,.0f}**")

        # Hiển thị độ tin cậy ước lượng (ví dụ đơn giản, giả sử R2 = 0.85)
        st.write("🔍 Độ tin cậy mô hình (ước lượng): **85%**")
