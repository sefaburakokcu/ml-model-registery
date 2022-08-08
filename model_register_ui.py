"""
Created on Mon Aug  8 22:49:23 2022

@author: Sefaburak
"""

import os
import streamlit as st


MODEL_EXTS =[".pth", ".pt", ".onnx", ".pb", ".wk", ".caffemodel", ".prototxt",
             ".engine"]

def save_uploadedfile(uploaded_file, save_dir):
     if not os.path.exists(save_dir):
          os.makedirs(save_dir)
          
     with open(os.path.join(save_dir, uploaded_file.name), "wb") as f:
         f.write(uploaded_file.getbuffer())
     return st.success(f"Saved File:{uploaded_file.name} to save_dir")
 
    
save_dir = r'D:\Users\Sefaburak\Desktop\projects\ml-model-registery\savem' 
 
file = st.file_uploader("Upload A Model",type=MODEL_EXTS)

if file is not None:
    file_details = {"File Name":file.name,"File Type":file.type}
    st.write(file_details)
    save_uploadedfile(file, save_dir)
    
    
