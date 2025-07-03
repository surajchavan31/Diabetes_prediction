import numpy as np
import pickle
import streamlit as st

st.title('DIABETES PREDICTION ')

loaded_model = pickle.load(open("Diabetesmodel.pkl","rb"))

def Disease(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_reshape = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_reshape)

    if (prediction[0]==0):
        return st.success("the person is not diabetes")
    else:
        return st.error("the person is diabetes")
    

def main():
    st.write("prediction model")

    Pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
    Glucose = st.number_input("Glucose Level", min_value=0)
    BloodPressure = st.number_input("Blood Pressure value", min_value=0)
    SkinThickness = st.number_input("Skin Thickness value", min_value=0)
    Insulin = st.number_input("Insulin Level", min_value=0)
    BMI = st.number_input("BMI", min_value=0.0, format="%.2f")
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.4f")
    Age = st.number_input("Age", min_value=0, step=1)

    diagnosis = ""

    if st.button("predict"):
        diagnosis = Disease([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])


if __name__=="__main__":
    main()