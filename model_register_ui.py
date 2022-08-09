"""
Created on Mon Aug  8 22:49:23 2022

@author: Sefaburak
"""

import os
import datetime
import pathlib
import json
import streamlit as st

from opensearchpy import helpers
from elastic_db_handler import DbHandle

st.set_page_config(page_title="Machine Learning Model Registery",
                   #page_icon="logo.png",
                   layout="centered")

DB_NAME = "ml_model_registery_db"
PAIR_MODEL_EXTS = [[".caffemodel", ".prototxt"], [".params", ".json"]]
SINGLE_MODEL_EXTS = [".pth", ".pt", ".onnx", ".pb", ".wk", ".engine", ".trt"]
MODEL_EXTS = SINGLE_MODEL_EXTS + [ i  for pair in PAIR_MODEL_EXTS for i in pair]
ADDITIONAL_FILES_EXTS = [".txt", ".json", ".yaml", ".png", ".jpg", ".jpeg"]

MODEL_MAPPER = {".caffemodel" : "Caffe",
                ".prototxt": "Caffe",
                ".params": "Mxnet", 
                ".json": "Mxnet",
                ".pth": "Pytorch",
                ".pt": "Pytorch",
                ".onnx": "Onnx", 
                ".pb": "Tensorflow",
                ".wk": "Hisi",
                ".engine": "Tensorrt",
                ".trt": "Tensorrt"}

tasks_json_file = "ui_info.json"
try:
    with open(tasks_json_file, "r") as f:
        info = json.load(f)
        task_names = info["task_names"]
except:
    task_names = ["face_detection", "object_detection", "face_recognition", "Person_reid",
                  "Vehicle_reid", "object_detection_color", "object_detection_angle",
                  "facial_landmarks_extraction", "face_quality_assessment", "face_tracking",
                  "object_tracking", "face_detection_landmark_quality", "face_detection_landmark",
                  "other"]

def create_model_archieve_folder(model_root_folder, model_sub_folder):
    model_folder = os.path.join(model_root_folder, model_sub_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    return model_folder

def save_uploaded_file(uploaded_file, model_folder):
     with open(os.path.join(model_folder, uploaded_file.name), "wb") as f:
         f.write(uploaded_file.getbuffer())
     return st.success(f"Saved model file: {uploaded_file.name} to {model_folder}")

def download_files():
    pass

# check whether model pair is correct or not
def check_pair_model(model_files, single_model_exts, pair_model_exts):
    is_pair_has_both_files = False
    is_not_pair = False
    pair_model_ext = []
    for model_file in model_files:
        ext = pathlib.Path(model_file.name).suffix
        if ext in single_model_exts:
            is_not_pair = True
            return is_not_pair, is_pair_has_both_files
        
        if len(pair_model_ext) == 1:
            if ext in pair_model_ext:
                is_pair_has_both_files = True
        for ext_list in pair_model_exts:
            if ext in ext_list:
                pair_model_ext = list(ext_list)
                pair_model_ext.remove(ext) 
                break
    return is_not_pair, is_pair_has_both_files

def check_single_model(model_files, single_model_exts, pair_model_exts):
    is_not_pair = True
    is_correct_model = True
    for model_file in model_files:
        ext = pathlib.Path(model_file.name).suffix
        if ext not in single_model_exts:
            is_correct_model = False
            break
    return is_not_pair, is_correct_model


def check_model_files(model_files, single_model_exts, pair_model_exts):
    if len(model_files) == 1:
        is_not_pair, is_correct_model = check_single_model(model_files,
                                                           single_model_exts,
                                                           pair_model_exts)
    elif len(model_files) == 2:
        is_not_pair, is_correct_model = check_pair_model(model_files,
                                                         single_model_exts,
                                                         pair_model_exts)
    return is_not_pair, is_correct_model
    

def get_model_sub_folder_name(version_info, main_task, model_type,
                              production, model_name_without_suffix):
    if production:
        model_for = "production"
    else:
        model_for = "development"
    model_sub_folder = f"{version_info}/{model_for}/{model_type}/{main_task}/{model_name_without_suffix}"
    return model_sub_folder


def main():
    pass
    
# Initialize connection.
# Uses st.cache to only run once.
@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection(ip=None, port=None, user=None, password=None):
    db = DbHandle(ip, port, user, password)
    return db
    
if __name__ == "__main__":
    
    producer_mail = "burakokcu@aselsan.com.tr"
    producer = producer_mail.split("@")[0]
    
    info = "example model upload"
    
    ip = '192.170.100.119'
    port = 9200
    db = init_connection(ip, port)
    
    
    if not db.es.indices.exists(index=DB_NAME):
        db.create_index(DB_NAME, force=False)
        st.info(f"Created a new index with {DB_NAME} name")
        
        
    save_root = "/home/sefa/workspace/projects/.model_archieve"
    
    # Upload model files
    model_files = st.file_uploader("Upload A Model",type=MODEL_EXTS,
                            accept_multiple_files=True)
    # Upload additional files
    additional_files = st.file_uploader("Upload Additional Files",
                                        type=ADDITIONAL_FILES_EXTS,
                                        accept_multiple_files=True)
    
    version_info = st.number_input("Version Number", min_value=0.1, 
                                   max_value=100.0, value=0.1, step=0.1, format="%.1f")
    
    main_task = st.selectbox("Main Task", task_names)
    if main_task == "other":
        main_task = st.text_input("Enter A New Task Name(No space between words, use underscore)",
                                  value="New_task")
        if not main_task in task_names:
            task_names.append(main_task)
            with open(tasks_json_file, "w") as f:
                json.dump({"task_names":task_names}, f, indent=6)
        else:
            st.warning(f"{main_task} is already available!")

        
    sub_tasks = st.multiselect("Sub Tasks", task_names)
    
    production = st.checkbox("Model For Production", value=False)
    
    time_format='%Y-%m-%d %H:%M:%S'
    register_time = datetime.datetime.now() - datetime.timedelta(hours=3)
    register_time = register_time.strftime(time_format)

    register_button = st.button("Register Model")

    
    if register_button:
        model_files_length = len(model_files)
        if model_files_length == 0:
            st.warning("Please upload a model file")
        elif model_files_length > 2:
            st.error("Model files should be one or two files")
        else:
            is_not_pair, is_correct_model = check_model_files(model_files,
                                                              SINGLE_MODEL_EXTS,
                                                              PAIR_MODEL_EXTS)
            if not is_correct_model:
                if is_not_pair:
                    st.error("Model file has not a valid extension.")
                else:
                    st.error(f"Model pair has not valid extensions. Please \
                             select a correct pair, e.g {PAIR_MODEL_EXTS}")
            else:
                save_model_files_successfully = False
                for i, model_file in enumerate(model_files):
                    status_bar = st.progress(0)
                
                    model_name_without_suffix = pathlib.Path(model_file.name).stem
                    model_suffix = pathlib.Path(model_file.name).suffix
                    file_details = {"File Name":model_file.name,
                                    "File Type":model_file.type}
                    
                    model_format = MODEL_MAPPER.get(model_suffix, "Unknown")
                    
                    model_sub_folder = get_model_sub_folder_name(version_info,
                                        main_task, model_format, production,
                                        model_name_without_suffix)
                    st.info(model_sub_folder)
                    save_dir = create_model_archieve_folder(save_root, model_sub_folder)
                    save_uploaded_file(model_file, save_dir)
                    if not save_model_files_successfully:
                        save_model_files_successfully = True
                    
                    if i == 0:
                        model_name = model_file.name
                        model_name_extra = ""
                    elif i == 1:
                        model_name_extra = model_file.name
                        
                if save_model_files_successfully:
                    additional_files_names = []
                    for additional_file in additional_files:
                        save_uploaded_file(additional_file, save_dir)
                        additional_files_names.append(additional_file.name)
                        
                    doc_id =  len(list(helpers.scan(db.es,
                                                 scroll = '3m',
                                                 size = 10,
                                                 index=DB_NAME)))+1
                    db.add_doc(DB_NAME, doc_id, model_name, save_dir, 
                               model_name_extra, additional_files_names, model_format, 
                               production, producer, register_time, version_info, main_task, 
                               sub_tasks, info)
            
    
    
