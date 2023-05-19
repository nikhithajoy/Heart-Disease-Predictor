import streamlit as st
import pandas as pd
from sklearn.svm import SVC

st.write("""
# Heart Disease Predictor
This app is for informational purposes only and should not replace medical consultation.
""")

data = pd.read_csv("C:/Users/nikhi/OneDrive/Documents/Self Projects/Heart Disease Prediction/heart - heart - heart - heart.csv")

st.sidebar.header('User Input Parameters')
st.sidebar.write("Please adjust the sliders below to provide information about health metrics")


def user_input_features():
    age = st.sidebar.slider('Age', 0, 150, 50)
    cp = st.sidebar.slider('CP', 0, 3, 0)
    trestbps = st.sidebar.slider('trestbps', 70, 200, 90)
    chol = st.sidebar.slider('Cholesterol', 120, 600, 246)
    fbs = st.sidebar.slider('FBS', 0, 1, 0)
    restecg = st.sidebar.slider('restecg', 0, 2, 0)
    thalach = st.sidebar.slider('thalach', 70, 210, 149)
    exang = st.sidebar.slider('exang', 0, 1, 0)
    oldpeak = st.sidebar.slider('oldpeak', 0.0, 10.0, 1.07)
    slope = st.sidebar.slider('slope', 0, 2, 1)
    ca = st.sidebar.slider('ca', 0, 5, 0)
    thal = st.sidebar.slider('thal', 0, 5, 0)

    features = pd.DataFrame({
            'age': [age],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalach': [thalach],
            'exang': [exang],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'ca': [ca],
            'thal': [thal]
        })
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

ml = SVC()
ml.fit(X, Y)

df = df.reindex(columns=X.columns, fill_value=0)

prediction = ml.predict(df)

st.subheader('Prediction')
st.write(prediction)

if prediction == 0:
    st.write(" ** The result is negative. You are healthy! **")
else:
    st.write(" ** The result is positive. You have heart disease! **")