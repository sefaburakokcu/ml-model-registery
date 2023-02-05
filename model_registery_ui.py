"""
Created on Mon Aug  8 22:49:23 2022

@author: Sefaburak
"""

import os
import glob
import datetime
import pathlib
import json
import yaml
import streamlit as st
import streamlit_authenticator as stauth
import numpy as np
import pandas as pd
from zipfile import ZipFile, ZIP_DEFLATED
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

from utils.json_db_handler import JsonDbHandle

st.set_page_config(page_title="Machine Learning Model Registry",
                   #page_icon="logo.png",
                   layout="wide")

PAIR_MODEL_EXTS = [[".caffemodel", ".prototxt"], [".params", ".json"]]
SINGLE_MODEL_EXTS = [".pth", ".pt", ".onnx", ".pb", ".wk", ".engine", ".trt",
                     ".tflite"]
MODEL_EXTS = SINGLE_MODEL_EXTS + [ i  for pair in PAIR_MODEL_EXTS for i in pair]
ADDITIONAL_FILES_EXTS = [".txt", ".json", ".yaml", ".png", ".jpg", ".jpeg"]

MODEL_MAPPER = {".caffemodel" : "caffe",
                ".prototxt": "caffe",
                ".params": "mxnet",
                ".json": "mxnet",
                ".pth": "pytorch",
                ".pt": "pytorch",
                ".onnx": "onnx",
                ".pb": "tensorflow",
                ".wk": "hisi",
                ".engine": "tensorrt",
                ".trt": "tensorrt",
                ".tflite": "tensorflowlite"}

tasks_json_file = "./assets/ui_info.json"
try:
    with open(tasks_json_file, "r") as f:
        info = json.load(f)
        task_names = info["task_names"]
except:
    task_names = ["face_detection", "object_detection", "face_recognition", "person_reid",
                  "vehicle_reid", "object_detection_color", "object_detection_angle",
                  "facial_landmarks_extraction", "face_quality_assessment", "face_tracking",
                  "object_tracking", "face_detection_landmark_quality", "face_detection_landmark",
                  "other"]

def create_archieve_folder(model_folder):
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    return model_folder

def save_uploaded_file(uploaded_file, model_folder):
     with open(os.path.join(model_folder, uploaded_file.name), "wb") as f:
         f.write(uploaded_file.getbuffer())
     st.success(f"Saved model file: {uploaded_file.name}")


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
    else:
        raise f"Number of model files should be 1 or 2 but {len(model_files)} is given."

    return is_not_pair, is_correct_model
    

def get_model_sub_folder_name(version_info, main_task, model_type,
                              production, model_name_without_suffix):
    if production:
        model_for = "production"
    else:
        model_for = "development"
    model_sub_folder = f"{version_info}/{model_for}/{model_type}/{main_task}/{model_name_without_suffix}"
    return model_sub_folder


# Uses st.cache to only run once.
@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection(db_name, index_name):
    db = JsonDbHandle(db_name, index_name)
    return db


def model_register(db):
    st.title("Model Register")
    
    # Upload model files
    model_files = st.file_uploader("Upload A Model",type=MODEL_EXTS,
                            accept_multiple_files=True)
    # Upload additional files
    extra_files = st.file_uploader("Upload Additional Files",
                                        type=ADDITIONAL_FILES_EXTS,
                                        accept_multiple_files=True)
    
    model_version = st.number_input("Version Number", min_value=0.1,
                                   max_value=100.0, value=0.1, step=0.1, format="%.1f")
    
    main_task = st.selectbox("Main Task", task_names)
    if main_task == "other":
        main_task = st.text_input("Enter A New Task Name(No space between words, use underscore)",
                                  value="New_task")
        main_task = main_task.lower()
        if not main_task in task_names:
            task_names.append(main_task)
            with open(tasks_json_file, "w") as f:
                json.dump({"task_names":task_names}, f, indent=6)
        else:
            st.warning(f"{main_task} is already available!")

        
    tags = st.multiselect("Tags", task_names)
    
    info = st.text_input("Enter Additional Info", value="")
    
    model_for = st.selectbox("Model For", ["development", "production"])
    
    time_format='%Y%m%d %H%M%S'
    register_date_time = datetime.datetime.now() - datetime.timedelta(hours=3)
    register_date_time = register_date_time.strftime(time_format)
    register_date, register_time = register_date_time.split()

    register_button = st.button("Register Model")

    registerer = st.session_state["username"]
    
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

                model_names = [model_file.name for model_file in model_files]
                extra_file_names = [extra_file.name for extra_file in extra_files]

                root_model_name = pathlib.Path(model_files[0].name).stem
                model_suffix = pathlib.Path(model_files[0].name).suffix
                # file_details = {"File Name":model_file.name,
                #                 "File Type":model_file.type}

                model_format = MODEL_MAPPER.get(model_suffix, "Unknown")

                model_doc, relative_model_sub_folder = db.create_model_doc(registerer, model_for, main_task, model_format,
                                 root_model_name,model_version, register_date,
                                 register_time, model_names, extra_file_names,
                                 tags, info)

                if db.check_doc_exists(relative_model_sub_folder):
                    st.warning(f"Model already exists! ")
                else:
                    st.info("Saving model(s)...")
                    model_sub_folder = db.get_full_model_path(relative_model_sub_folder)
                    extra_sub_folder = db.get_full_extras_path(relative_model_sub_folder)
                    model_save_dir = create_archieve_folder(model_sub_folder)
                    extra_save_dir = create_archieve_folder(extra_sub_folder)

                    for i, model_file in enumerate(model_files):
                        save_model_files_successfully = False
                        status_bar = st.progress(0)
                        save_uploaded_file(model_file, model_save_dir)
                        if not save_model_files_successfully:
                            save_model_files_successfully = True

                    if len(extra_files) == 0:
                        save_extra_files_successfully = True
                    else:
                        for i, extra_file in enumerate(extra_files):
                            save_extra_files_successfully = False
                            status_bar = st.progress(0)
                            save_uploaded_file(extra_file, extra_save_dir)
                            if not save_extra_files_successfully:
                                save_extra_files_successfully = True

                    if (save_model_files_successfully and save_extra_files_successfully):
                        db.add_model_doc(relative_model_sub_folder, model_doc)
                        st.success("Model is added to database successfully.")

def get_filtered_models(models_info, filtered_registerer, filtered_tasks, filtered_model_format,
                                            filtered_model_for, filtered_tags):
    filtered_model_info = {}

    for model_info_key, model_info_value in models_info.items():
        flag_provider = True
        flag_model_format = True
        flag_task = True
        flag_model_for = True
        flag_tags = True

        if len(filtered_registerer) > 0:
            if model_info_value["registerer"] not in filtered_registerer:
                flag_provider = False

        if len(filtered_model_format) > 0:
            if model_info_value["model_format"] not in filtered_model_format:
                flag_model_format = False

        if len(filtered_tasks) > 0:
            if model_info_value["main_task"] not in filtered_tasks:
                flag_task = False

        if len(filtered_model_for) > 0:
            if model_info_value["model_for"] not in filtered_model_for:
                flag_model_for = False

        if len(filtered_tags) > 0:
            model_tags = model_info_value["tags"]
            flag_tags = False
            for model_tag in model_tags:
                if model_tag in filtered_tags:
                    flag_tags = True

        if (flag_provider and flag_model_format and flag_task and flag_model_for and flag_tags):
            filtered_model_info[model_info_key] = model_info_value
    return filtered_model_info

def get_models_dataframe(models_info):
    models_dataframe_dict = {}
    models_dataframe_dict["model_name"] = []
    models_dataframe_dict["registerer"] = []
    models_dataframe_dict["model_for"] = []
    models_dataframe_dict["model_format"] = []
    models_dataframe_dict["main_task"] = []
    models_dataframe_dict["tags"] = []
    models_dataframe_dict["model_version"] = []
    models_dataframe_dict["register_date"] = []
    models_dataframe_dict["register_time"] = []
    models_dataframe_dict["info"] = []
    models_dataframe_dict["model_path"] = []

    for model_info_value, model_info_key in models_info.items():
        models_dataframe_dict["model_name"].append(model_info_key["model_names"][0])
        models_dataframe_dict["registerer"].append(model_info_key["registerer"])
        models_dataframe_dict["model_for"].append(model_info_key["model_for"])
        models_dataframe_dict["model_format"].append(model_info_key["model_format"])
        models_dataframe_dict["main_task"].append(model_info_key["main_task"])
        models_dataframe_dict["tags"].append(model_info_key["tags"])
        models_dataframe_dict["model_version"].append(model_info_key["model_version"])
        models_dataframe_dict["register_date"].append(model_info_key["register_date"])
        models_dataframe_dict["register_time"].append(model_info_key["register_time"])
        models_dataframe_dict["info"].append(model_info_key["info"])
        models_dataframe_dict["model_path"].append(model_info_value)
    return models_dataframe_dict


def compress_files(folder_list, zip_name="models.zip"):
    zip_object = ZipFile(file=zip_name, mode="w", compression=ZIP_DEFLATED)
    for folder in folder_list:
        for file_path in glob.glob(folder + "**/**"):
            file_path_split = file_path.split(os.path.sep)  # last 8 for the path
            new_path = "/".join(file_path_split[-8:])
            zip_object.write(file_path, arcname=new_path)
    zip_object.close()


def download_files(model_folders, zip_name="models.zip"):
    compress_files(model_folders, zip_name)

    st.info(f"Please use the button below in order to download models.")
    with open(zip_name, 'rb') as f:
        btn = st.download_button(
            label="download",
            data=f,
            file_name=zip_name,
            mime="application/zip"
        )
import shutil

def delete_files(db, model_folders):
    for model_path in model_folders:
        st.warning(f"Deleting {model_path}...")
        full_model_path = db.get_full_path(model_path)
        move_folder = full_model_path.replace(".db", ".tmp_models")
        if not os.path.exists(move_folder):
            os.makedirs(move_folder)
        shutil.move(full_model_path, move_folder)
        db.delete_model_doc(model_path)
    # remove also from db

@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None})
def load_credentials(config_file="./.credentials/config.yaml"):
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    return config


def update_credentials(config, config_file="./.credentials/config.yaml"):
    with open(config_file, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def model_hub():
    st.title("Model HUB")
    with st.sidebar.form(key='Form1'):
        filtered_registerer = st.multiselect("Registerer", REGISTERERS)
        filtered_tasks = st.multiselect("Tasks", task_names)
        filtered_model_for = st.multiselect("Model For", ["development", "production"])
        filtered_model_format = st.multiselect("Model Format", set(MODEL_MAPPER.values()))
        filtered_tags = st.multiselect("Tags", task_names)
        submitted = st.form_submit_button(label='filter models')

    filtered_models = get_filtered_models(db.index_info, filtered_registerer, filtered_tasks, filtered_model_format,
                                          filtered_model_for, filtered_tags)

    data = get_models_dataframe(filtered_models)
    df = pd.DataFrame(data)

    gd = GridOptionsBuilder.from_dataframe(df)
    gd.configure_selection(selection_mode='multiple', use_checkbox=True)
    gridoptions = gd.build()

    grid_table = AgGrid(df, height=250, gridOptions=gridoptions,
                        update_mode=GridUpdateMode.GRID_CHANGED,
                        reload_data=st.session_state["reload_data"],
                        fit_columns_on_grid_load=True)
    selected_rows = grid_table["selected_rows"]
    st.session_state["reload_data"] = False

    col1, _, _, col2 = st.columns([1, 0.5, 0.3, 0.2])

    with col1:
        download_button = st.button('prepare for download')
        if download_button:
            model_folders = []
            st.info(f"Preparing models for downloading...")
            for selected_row in selected_rows:
                row_idx = selected_row['_selectedRowNodeInfo']['nodeRowIndex']
                model_folders.append(db.get_full_path(selected_row["model_path"]))
            download_files(model_folders, zip_name="models.zip")
            st.session_state["reload_data"] = True

    with col2:
        delete_button = st.button('delete')
        if delete_button:
            model_folders = []
            for selected_row in selected_rows:
                row_idx = selected_row['_selectedRowNodeInfo']['nodeRowIndex']
                model_folders.append(selected_row["model_path"])
            delete_files(db, model_folders)
            st.session_state["reload_data"] = True


if __name__ == "__main__":

    db_name = "db_test"
    index_name = "db_models"

    if "reload_data" not in st.session_state:
        st.session_state["reload_data"] = False
    reload_data = st.session_state["reload_data"]

    db = init_connection(db_name, index_name)

    REGISTERERS = set([info["registerer"] for info in db.index_info.values()])

    config = load_credentials("./.credentials/config.yaml")

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = None

    if "name" not in st.session_state:
        st.session_state["name"] = None

    if "username" not in st.session_state:
        st.session_state["username"] = None

    TABS = ['Model Register', 'Model HUB']

    title = "Machine Learning Model Registry"

    _, col2, _ = st.columns([1, 1, 1])

    with col2:
        titleholder = st.empty()
        titleholder.title(title)
        st.session_state["name"], st.session_state["authentication_status"], st.session_state[
            "username"] = authenticator.login('Login', 'main')
        if st.session_state["authentication_status"] is None:
            st.warning('Please enter your username and password')
            with st.expander("*New to Model Registry? Sign up.*"):
                try:
                    if authenticator.register_user('Register user', preauthorization=False):
                        st.success('User registered successfully')
                    update_credentials(config, config_file="./.credentials/config.yaml")
                except Exception as e:
                    st.error(e)

        elif not st.session_state["authentication_status"]:
            st.error('Username/password is incorrect')
            with st.expander("*New to Model Registry? Sign up.*"):
                try:
                    if authenticator.register_user('Register user', preauthorization=False):
                        st.success('User registered successfully')
                    update_credentials(config, config_file="./.credentials/config.yaml")
                except Exception as e:
                    st.error(e)

    if st.session_state["authentication_status"]:
        titleholder.empty()
        st.title(title)
        with st.sidebar:
            st.title(title)
            active_tab = st.radio("Menu:", TABS)
            authenticator.logout('Logout', 'main')
            st.write(f"Welcome *{st.session_state['name']}*")

        if active_tab == 'Model Register':
            model_register(db)

        elif active_tab == 'Model HUB':
            model_hub()
