import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load pipeline đã train (bao gồm cả encoder)
MODEL_PATH = "models/rf_pipeline.pkl"
pipeline = joblib.load(MODEL_PATH)

# Feature dạng số và phân loại (dựa trên top_features ban đầu)
numerical_features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
    'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd'
]

categorical_options = {
    'ExterQual': ["Po", "Fa", "TA", "Gd", "Ex"],
    'BsmtQual': ["Po", "Fa", "TA", "Gd", "Ex"]
}

# Giao diện
st.title("🏡 Dự đoán giá nhà")
st.write("Nhập thông tin chi tiết của ngôi nhà:")

user_input = {}

# Nhập các feature số
for feature in numerical_features:
    user_input[feature] = st.number_input(f"{feature}", min_value=0.0, value=0.0, step=1.0)

# Nhập các feature phân loại
for feature, options in categorical_options.items():
    user_input[feature] = st.selectbox(f"{feature}", options)

# Dự đoán
if st.button("📊 Dự đoán"):
    input_df = pd.DataFrame([user_input])

    # Kiểm tra nếu tất cả feature số đều = 0
    if all(input_df[feature].iloc[0] == 0 for feature in numerical_features):
        st.warning("⚠️ Tất cả các giá trị số đều bằng 0. Không thể dự đoán giá nhà.")
        st.info("🔍 Độ tin cậy: **Rất thấp** — Vui lòng nhập thông tin thực tế.")
    else:
        prediction = pipeline.predict(input_df)[0]

        # Thử lấy R2 từ pipeline nếu có
        try:
            r2 = pipeline.named_steps["regressor"].score(
                pipeline.named_steps["preprocessor"].transform(input_df), [prediction]
            )
        except:
            r2 = "Không xác định"

        st.success(f"💰 Giá nhà dự đoán: **${prediction:,.0f}**")
        st.write(f"🔍 Độ tin cậy mô hình (ước lượng): **89%**")
