import os
import glob
import json


def load_index(index_path):
    with open(index_path,) as f:
        index_info = json.load(f)
    return index_info


def save_index(index_path, index_info):
    with open(index_path, "w") as f:
        json.dump(index_info, f, indent=6)


class JsonDbHandle:
    def __init__(self, db_name, index_name):
        self.db_name = db_name
        self.index_name = index_name

        self.db_path = "./.db"
        self.index_info = {}

        self.index_path = f"{self.db_path}/{self.db_name}/{self.index_name}.json"

        self.get_index_info()


    def get_index_info(self):
        if os.path.exists(self.index_path):
            self.index_info = load_index(self.index_path)

    def get_full_path(self, model_path):
        return f"{self.db_path}/{self.db_name}/{self.index_name}/{model_path}/"

    def get_full_model_path(self, model_path):
        return f"{self.get_full_path(model_path)}/models/"

    def get_full_extras_path(self, model_path):
        return f"{self.get_full_path(model_path)}/extras/"

    def check_doc_exists(self, doc_key):
        flag_exists = False
        if doc_key in self.index_info.keys():
            flag_exists = True
        return flag_exists


    def create_model_doc(self, registerer, model_for, main_task, model_format, root_model_name, model_version, register_date,
                         register_time, model_names, extra_file_names, tags, info):
        model_doc = {
            "registerer": registerer,
            "model_for": model_for,
            "main_task": main_task,
            "model_format": model_format,
            "root_model_name": root_model_name,
            "model_version": model_version,
            "register_date": register_date,
            "register_time": register_time,
            "model_names": model_names,
            "extra_file_names": extra_file_names,
            "tags": tags,
            "info": info
        }
        # relative_model_folder_path = f"{registerer}/{model_for}/{main_task}/{model_format}/{root_model_name}/{model_version}/{register_date}/{register_time}/models/"
        root_model_folder_path = f"{registerer}/{model_for}/{main_task}/{model_format}/{root_model_name}/{model_version}/"
        return model_doc, root_model_folder_path


    def create_db_folder(self):
        if not os.path.exists(self.db_name):
            os.makedirs(self.db_name)

    def add_model_doc(self, relative_model_folder_path, model_doc):
        self.index_info[relative_model_folder_path] = model_doc
        save_index(self.index_path, self.index_info)

    def delete_model_doc(self, root_model_folder_path):
        try:
            del self.index_info[root_model_folder_path]
            save_index(self.index_path, self.index_info)
        except KeyError:
            print(f"{root_model_folder_path} could not be found in model database.")
