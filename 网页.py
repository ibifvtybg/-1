import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import xgboost as xgb

# 加载模型
model = joblib.load('XGBoost.pkl')

# 获取模型输入特征数量
model_input_features = model.feature_names_in_
expected_feature_count = len(model_input_features)

# 定义新的 12 个特征选项及名称
cp_options = {
    1: '重度职业紧张 (1)',
    2: '中度职业紧张 (2)',
    3: '轻度职业紧张 (3)',
    4: '无症状 (4)'
}

feature_names = ['年龄', '在职工龄', 'A2', 'A3', 'A4', 'A6', 'B4', 'B5', '工时分组', '生活满意度', '睡眠状况', '工作负担度']

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Streamlit 用户界面
st.title("职业紧张预测 app")

# 年龄
age = st.number_input("年龄：", min_value=1, max_value=120, value=50)

# 在职工龄
service_years = st.number_input("在职工龄（年）：", min_value=0, max_value=40, value=5)

# A2（性别）
A2_options = {1: '女性', 0: '男性'}
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
        features = np.array([feature_values])
        if len(features[0])!= expected_feature_count:
            # 如果特征数量不匹配，使用零填充来达到模型期望的特征数量
            padded_features = np.pad(features, ((0, 0), (0, expected_feature_count - len(features[0]))), 'constant')
            predicted_class = model.predict(padded_features)[0]
            predicted_proba = model.predict_proba(padded_features)[0]
        else:
            predicted_class = model.predict(features)[0]
            predicted_proba = model.predict_proba(features)[0]

        # 显示预测结果
        st.write(f"**预测类别：** {predicted_class}")

        # 将概率转换为小数并显示
        decimal_probabilities = [f'{prob:.4f}' for prob in predicted_proba]
        st.write(f"**预测概率：** {decimal_probabilities}")

        # 根据预测结果生成建议
        probability = predicted_proba[predicted_class] * 100

        if predicted_class == 4:
            advice = (
                f"根据我们的模型，您无职业紧张症状。"
                f"模型预测该员工有职业紧张症状的概率为 {probability:.1f}%。"
                "请继续保持良好的工作和生活状态。"
            )
        elif predicted_class == 3:
            advice = (
                f"根据我们的模型，您有轻度职业紧张症状。"
                f"模型预测该员工有职业紧张症状的概率为 {probability:.1f}%。"
                "建议您适当调整工作节奏，关注自身身心健康。"
            )
        elif predicted_class == 2:
            advice = (
                f"根据我们的模型，您有中度职业紧张症状。"
                f"模型预测该员工有职业紧张症状的概率为 {probability:.1f}%。"
                "建议您寻求专业帮助，如心理咨询或与上级沟通调整工作安排。"
            )
        elif predicted_class == 1:
            advice = (
                f"根据我们的模型，您有重度职业紧张症状。"
                f"模型预测该员工有职业紧张症状的概率为 {probability:.1f}%。"
                "强烈建议您立即采取行动，如休假、寻求医疗支持或与管理层协商改善工作环境。"
            )
        else:
            advice = "预测结果出现未知情况。"

        st.write(advice)

        # 进行 SHAP 值计算，不直接使用 DMatrix
        if len(features[0])!= expected_feature_count:
            data_df = pd.DataFrame(padded_features[0].reshape(1, -1), columns=model_input_features)
        else:
            data_df = pd.DataFrame(features[0].reshape(1, -1), columns=feature_names)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data_df)

        # 更加谨慎地处理 expected_value
        base_value = explainer.expected_value if not isinstance(explainer.expected_value, list) else (explainer.expected_value[0] if len(explainer.expected_value) > 0 else None)
        if base_value is None:
            raise ValueError("Unable to determine base value for SHAP force plot.")

        try:
            shap.plots.force(base_value, shap_values[0], data_df)
            plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        except Exception as e:
            print(f"Error in force plot: {e}")
            # 如果 force plot 失败，尝试其他绘图方法
            shap.summary_plot(shap_values, data_df, show=False)
            plt.title('SHAP 值汇总图')
            plt.xlabel('特征')
            plt.ylabel('SHAP 值')
            plt.savefig("shap_summary_plot.png", bbox_inches='tight', dpi=1200)

        st.image("shap_summary_plot.png")
    except Exception as e:
        st.write(f"出现错误：{e}")

if st.button("预测"):
    predict()
