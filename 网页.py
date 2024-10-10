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

# 加载模型
file_path = 'C:\\Users\\18657\\Desktop\\长工时\\XGBoost.pkl'
model = joblib.load(file_path)

# 定义特征选项
cp_options = {
    1: '重度职业紧张 (1)',
    2: '中度职业紧张 (2)',
    3: '轻度职业紧张 (3)',
    4: '无症状 (4)'
}

# 定义特征名称
feature_names = ['年龄', '在职工龄', 'A2', 'A3', 'A4', 'A6', 'B4', 'B5',  '工时分组', '生活满意度', '睡眠状况', '工作负担度'
]

# Streamlit用户界面
st.title("职业紧张预测app")

# 年龄
年龄 = st.number_input("年龄：", min_value=1, max_value=120, value=50)


# 在职工龄
在职工龄 = st.number_input("在职工龄（年）：", min_value=0, max_value=40, value=5)

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
A4_options = {0: '未婚',1: '已婚住在一起',2: '已婚分居或异地',3: '离婚', 4: '丧偶'}
A4 = st.selectbox(
    "婚姻状况：",
    options=list(A4_options.keys()),
    format_func=lambda x: A4_options[x]
)

# A6（月收入）
A6_options = {1: '少于3000元', 2: '3000-4999 元', 3: '5000-6999 元',4: '7000-8999 元', 5: '9000-10999 元',6:'11000 元及以上'}
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
工时分组_options = {1: '少于20小时', 2: '20-30小时', 3: '30-40小时', 4: '40-50小时', 5: '多于50小时'}
工时分组 = st.selectbox(
    "工时分组：",
    options=list(工时分组_options.keys()),
    format_func=lambda x: 工时分组_options[x]
)

# 生活满意度
生活满意度 = st.slider("生活满意度（1-5）：", min_value=1, max_value=5, value=3)

# 睡眠状况
睡眠状况 = st.slider("睡眠状况（1-5）：", min_value=1, max_value=5, value=3)

# 工作负担度
工作负担度 = st.slider("工作负担度（1-5）：", min_value=1, max_value=5, value=3)

# 处理输入并进行预测
feature_values = [
    年龄, 在职工龄, A2, A3, A4, A6, B4, B5, 工时分组, 生活满意度, 睡眠状况, 工作负担度
]
features = np.array([feature_values])

if st.button("预测"):
    # 预测类别和概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

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

    # 计算SHAP值并显示力图
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(
        pd.DataFrame([feature_values], columns=feature_names)
    )

    shap.force_plot(
        explainer.expected_value, shap_values[0],
        pd.DataFrame([feature_values], columns=feature_names),
        matplotlib=True
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")
