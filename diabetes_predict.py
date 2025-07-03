# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open("E:/Machine Learning/Deploying ML model/Diabetesmodel.pkl",'rb'))

input_data1 = (4,103,60,33,192,24,0.966,33)

# changing the input data to numpy array
input_data_as_numpy_array1 = np.asarray(input_data1)

# reshape the array as we are predicting  for one instance
input_reshape1 = input_data_as_numpy_array1.reshape(1,-1)

prediction1 = loaded_model.predict(input_reshape1)
print(prediction1)

if (prediction1[0]==0):
  print("the person is not diabetic")
else:
  print("the person is diabetic")