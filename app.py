import streamlit as st
import numpy as np
from src.data_loader import select_feature
from Random_rest import sklearn_

# Initialize the model
Ran_model = AlgorithmModel()

def main():
    st.title("Ứng dụng Dự đoán Giá Nhà")
    st.write("Vui lòng nhập thông tin về ngôi nhà để dự đoán giá.")

    # Input fields
    lot_area = st.number_input("Diện tích lô đất (LotArea):", min_value=0, value=5000, step=1)
    year_built = st.number_input("Năm xây dựng (YearBuilt):", min_value=1800, value=2000, step=1)
    gr_liv_area = st.number_input("Diện tích sàn (GrLivArea):", min_value=0, value=1000, step=1)
    bedroom_abv_gr = st.number_input("Số phòng ngủ (BedroomAbvGr):", min_value=0, value=3, step=1)
    tot_rms_abv_grd = st.number_input("Tổng số phòng trên mặt đất (TotRmsAbvGrd):", min_value=0, value=6, step=1)

    # Dự đoán khi bấm nút
    if st.button("Dự đoán giá nhà"):
        input_features = [lot_area, year_built, gr_liv_area, bedroom_abv_gr, tot_rms_abv_grd]

        try:
            prediction = lr_model.predict(input_features)
            st.success(f"Giá nhà được dự đoán: {prediction:,.2f} USD")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    main()