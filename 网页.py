# -*- coding: utf-8 -*-
"""
Spyder Editor

这是一个临时的脚本文件。
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

# 加载模型
model = joblib.load('XGBoost.pkl')

# 定义新的 12 个特征选项及名称
cp_options = {
    1: '重度职业紧张 (1)',
    2: '中度职业紧张 (2)',
    3: '轻度职业紧张 (3)',
    4: '无症状 (4)'
}

feature_names = ['年龄', '在职工龄', 'A2', 'A3', 'A4', 'A6', 'B4', 'B5', '工时分组', '生活满意度', '睡眠状况', '工作负担度']

# Streamlit 用户界面
st.title("职业紧张预测 app")

# 年龄
age = st.number_input("年龄：", min_value=1, max_value=120, value=50)

# 在职工龄
service_years = st.number_input("在职工龄（年）：", min_value=0, max_value=40, value=5)

# A2（性别）
A2_options = {0: '女性', 1: '男性'}
A2 = st.selectbox(
    "性别：",
    options=list(A2_options.keys()),
    format_func=lambda x: A2_options[x]
)

# A3（学历）
A3_options = {1: '初中及以下', 2: '高中或中专', 3: '大专或高职', 4: '大学本科', 5: '研究生及以上'}
A3 = st.selectbox(
    "学历：",
    options=list(A3_options.keys()),
    format_func=lambda x: A3_options[x]
)

# A4（婚姻状况）
A4_options = {0: '未婚', 1: '已婚住在一起', 2: '已婚分居或异地', 3: '离婚', 4: '丧偶'}
A4 = st.selectbox(
    "婚姻状况：",
    options=list(A4_options.keys()),
    format_func=lambda x: A4_options[x]
)

# A6（月收入）
A6_options = {1: '少于 3000 元', 2: '3000 - 4999 元', 3: '5000 - 6999 元', 4: '7000 - 8999 元', 5: '9000 - 10999 元', 6: '11000 元及以上'}
A6 = st.selectbox(
    "月收入：",
    options=list(A6_options.keys()),
    format_func=lambda x: A6_options[x]
)

# B4（是否轮班）
B4_options = {0: '否', 1: '是'}
B4 = st.selectbox(
    "是否轮班：",
    options=list(B4_options.keys()),
    format_func=lambda x: B4_options[x]
)

# B5（是否需要上夜班）
B5_options = {0: '否', 1: '是'}
B5 = st.selectbox(
    "是否需要上夜班：",
    options=list(B5_options.keys()),
    format_func=lambda x: B5_options[x]
)

# 工时分组
working_hours_group_options = {1: '少于 20 小时', 2: '20 - 30 小时', 3: '30 - 40 小时', 4: '40 - 50 小时', 5: '多于 50 小时'}
working_hours_group = st.selectbox(
    "工时分组：",
    options=list(working_hours_group_options.keys()),
    format_func=lambda x: working_hours_group_options[x]
)

# 生活满意度
life_satisfaction = st.slider("生活满意度（1 - 5）：", min_value=1, max_value=5, value=3)

# 睡眠状况
sleep_status = st.slider("睡眠状况（1 - 5）：", min_value=1, max_value=5, value=3)

# 工作负担度
work_load = st.slider("工作负担度（1 - 5）：", min_value=1, max_value=5, value=3)

def predict():
    try:
        feature_values = [
            age, service_years, A2, A3, A4, A6, B4, B5, working_hours_group, life_satisfaction, sleep_status, work_load
        ]
        dmatrix = xgb.DMatrix(np.array([feature_values]))
        predicted_class = model.predict(dmatrix)[0]
        predicted_proba = model.predict_proba(dmatrix)[0]

        # 显示预测结果
        st.write(f"**预测类别：** {predicted_class}")
        st.write(f"**预测概率：** {predicted_proba}")

        # 根据预测结果生成建议
        probability = predicted_proba[predicted_class] * 100

        if predicted_class == 1:
            advice = (
                f"根据我们的模型，您有较高的职业紧张。"
                f"模型预测该员工有职业紧张症状的概率为 {probability:.1f}%。"
                "建议管理层关注该员工的工作状态，提供必要的支持和关怀。"
            )
        else:
            advice = (
                f"根据我们的模型，您患有职业紧张可能性较低。"
                "请继续保持良好的工作氛围，鼓励员工的积极性。"
            )

        st.write(advice)

        # 将 DMatrix 转换为 DataFrame
        df = pd.DataFrame(dmatrix.get_data().toarray()[0], columns=feature_names)

        # 计算 SHAP 值并显示力图
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)

        shap.force_plot(
            explainer.expected_value, shap_values[0],
            df,
            matplotlib=True
        )
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

        st.image("shap_force_plot.png")
    except Exception as e:
        st.write(f"出现错误：{e}")

if st.button("预测"):
    predict()
