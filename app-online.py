import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import distutils
from sklearn.ensemble import RandomForestRegressor

# 初始化 session_state 中的 data
# 创建一个空的DataFrame来存储预测数据
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns=['Age', 'Discharge ferritin', 'Discharge WBC', 'Discharge HB', 'Discharge PLT', 'Discharge AST', 'Discharge DB', 'Discharge TP', 'Discharge ALB', 'Discharge Ca', 'Prediction', 'Label'])

# 在主页面上显示数据
st.header('6-month mortality of adult HLH patients based on RT')

#st.markdown("### 本地图片示例")
#创建两列布局
left_column, col1,col2,col3,right_column = st.columns(5)


# 在左侧列中添加其他内容
left_column.write("")

# 在右侧列中显示图像
right_column.image('logo.png', caption='', width=100)

#with open("F:\\model\\jssrm\logo2.png", "rb") as f:
#    local_image = f.read()
#    st.image(local_image, caption='', width=100)
    

# 创建一个侧边栏
st.sidebar.header('输入参数')

# Input bar 1
a = st.sidebar.number_input('Age (year)', min_value=0, max_value=150, value=47)
b = st.sidebar.number_input('Discharge ferritin (μg/L)', min_value=0.0, max_value=20000.0, value=123.0)
c = st.sidebar.number_input('Discharge WBC (×109/L)', min_value=0.00, max_value=100.00, value=20.80)
d = st.sidebar.number_input('Discharge HB (g/L)', min_value=0.00, max_value=100.00, value=45.96)
e = st.sidebar.number_input('Discharge PLT (×109/L)', min_value=0.00, max_value=1000.00, value=47.00)
f = st.sidebar.number_input('Discharge AST (U/L)', min_value=0.00, max_value=4000.00, value=2300.00)
g = st.sidebar.number_input('Discharge DB (μmol/L)', min_value=0.0, max_value=300.0, value=1.3)
h = st.sidebar.number_input('Discharge TP (g/L)', min_value=0.0, max_value=100.0, value=38.9)
i = st.sidebar.number_input('Discharge ALB (g/L)', min_value=0.0, max_value=100.0, value=15.9)
j = st.sidebar.number_input('Discharge Ca (mmol/L)', min_value=0.00, max_value=100.00, value=1.75)


# Unpickle classifier    
# dirs = 'F:\\model\\jssrm\\文章二'
mm = joblib.load('random_forest.pkl')
    
# If button is pressed
if st.sidebar.button("Submit"):
    # Store inputs into dataframe
    X = pd.DataFrame([[a, b, c, d,e,f,g,h,i,j]], 
                     columns = ["age","D  FERR","D  WBC","D  Hb","D  PLT","D  AST","D  DB","D  TP","D  Alb","D Ca"])
    #X = X.replace(["Brown", "Blue"], [1, 0])
    
    # Get prediction
    for index, row in X.iterrows():
        data1 = row.to_frame()
        data2 = pd.DataFrame(data1.values.T, columns=data1.index)
        result111 = mm.predict(data2)
        result222 = str(result111).replace("[", "")
        result = str(result222).replace("]", "")  #预测结果
        result333 = mm.predict_proba(data2)
        result444 = str(result333).replace("[[", "")
        result555 = str(result444).replace("]]", "")
        strlist = result555.split(' ')
        # print(index,row['OUTPATIENT_ID'],result)
        result_prob_neg = round(float(strlist[0]) * 100, 2)
        if len(strlist[1]) == 0:
            result_prob_pos = 'The conditions do not match and cannot be predicted'
        else:
            result_prob_pos = round(float(strlist[1]) * 100, 2) #预测概率        
    explainer = shap.TreeExplainer(mm) 
    shap_values = explainer.shap_values(data2)
    shap_values = shap_values.reshape((1, -1)) 
    # output_index = 0  # 假设我们选择第一个输出来解释
    # 绘制 SHAP 分析图
    # 构建特征向量
    # features = np.array([a, b, c, d, e,f,g])  # 假设只有 7 个参数
    # shap.force_plot(explainer.expected_value[0], shap_values[0], data2.iloc[0])
    # st.pyplot()
    
    
    # Output prediction
    st.text(f"The 6-month probability of death in adult patients with HLH is:：{str(result_prob_pos)}%")
    # st.text({str(shap_values[0])})
    

    # 创建一个新的DataFrame来存储用户输入的数据
    new_data = pd.DataFrame([[a, b, c, d, e, f, g, h, i, j, result_prob_pos/100, None]], 
                            columns=st.session_state['data'].columns)
    
    # 将预测结果添加到新数据中
    st.session_state['data'] = pd.concat([st.session_state['data'], new_data], ignore_index=True)

# 上传文件按钮
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # 读取 Excel 文件
    df = pd.read_excel(uploaded_file)
    
    # 列名映射字典,左为Excel字段，右为模型参数名
    column_mapping = {
        'Age (year)': 'age',
        'Discharge ferritin (μg/L)': 'D  FERR',
        'Discharge WBC (×109/L)': 'D  WBC',
        'Discharge HB (g/L)': 'D  Hb',
        'Discharge PLT (×109/L)': 'D  PLT',
        'Discharge AST (U/L)': 'D  AST',
        'Discharge DB (μmol/L)': 'D  DB',
        'Discharge TP (g/L)': 'D  TP',
        'Discharge ALB (g/L)': 'D  Alb',
        'Discharge Ca (mmol/L)': 'D Ca'
    }
    
     # 假设 'Label' 列在 Excel 文件中存在并且不参与计算
    label_column = 'label'  # 这是 Excel 文件中未参与计算的列名
    
    # 进行列名映射
    df = df.rename(columns=column_mapping)
    
    # 检查是否所有必需的列都存在
    missing_cols = [col for col in ["age","D  FERR","D  WBC","D  Hb","D  PLT","D  AST","D  DB","D  TP","D  Alb","D Ca"] if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing columns in the uploaded file: {', '.join(missing_cols)}")
    else:
        # 逐行读取数据并进行预测
        for _, row in df.iterrows():
            # 提取每一行数据并转换为模型输入格式
            X = pd.DataFrame([row], columns=["age","D  FERR","D  WBC","D  Hb","D  PLT","D  AST","D  DB","D  TP","D  Alb","D Ca"])
            
            # 进行预测
            result = mm.predict(X)[0]
            result_prob = mm.predict_proba(X)[0][1]
            
            # 获取标签列的值
            label = row[label_column] if label_column in row else None
            
            # 将结果添加到 session_state 的 data 中
            new_data = pd.DataFrame([[row["age"], row["D  FERR"], row["D  WBC"], row["D  Hb"], row["D  PLT"], row["D  AST"], row["D  DB"], row["D  TP"], row["D  Alb"], row["D Ca"], result_prob, label]], 
                                    columns=st.session_state['data'].columns)
            st.session_state['data'] = pd.concat([st.session_state['data'], new_data], ignore_index=True)
        
# 显示更新后的 data
st.write(st.session_state['data'])

# Footer
st.write("<p style='font-size: 12px;'>Disclaimer: This mini app is designed to provide general information and is not a substitute for professional medical advice or diagnosis. Always consult with a qualified healthcare professional if you have any concerns about your health.</p>", unsafe_allow_html=True)
st.markdown('<div style="font-size: 12px; text-align: right;">Powered by MyLab+ i-Research Consulting Team</div>', unsafe_allow_html=True)
