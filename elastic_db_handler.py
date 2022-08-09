"""
Created on Tue Aug  9 07:33:30 2022

@author: sefa
"""

import os
import time
from opensearchpy import OpenSearch
from opensearchpy import TransportError, ConnectionError


def get_results(res):
    image_paths = []
    names = []
    encodings = []
    ids = []
    for hit in res['hits']['hits']:
        _image_path = hit['_source']['image_path']
        _name = hit['_source']['name']
        _encodings = hit['_source']['face_encoding']
        _id = hit['_id']
        image_paths.append(_image_path)
        names.append(_name)
        encodings.append(_encodings)
        ids.append(_id)
    return image_paths, names,  encodings, ids


def create_client(host, http_port=9200, timeout=30):
    for _ in range(0, timeout):
        try:
          client = OpenSearch([{'host': host, 'port': http_port}])
          client = OpenSearch(
              hosts = [{'host': host, 'port': http_port}],
              http_compress = True, # enables gzip compression for request bodies
              http_auth = ('admin', 'admin'),
              # client_cert = client_cert_path,
              # client_key = client_key_path,
              use_ssl = True,
              verify_certs = True,
              ssl_assert_hostname = False,
              ssl_show_warn = False,
              ca_certs = f"{os.path.dirname(os.path.abspath(__file__))}/external/root-ca.pem"
          )
          
          client.cluster.health(wait_for_nodes=1)
          client.count()
          return client
        except (ConnectionError, TransportError):
          pass
        time.sleep(1)
    assert False, 'Timed out waiting for node for %s seconds' % timeout


class DbHandle():
    def __init__(self, ip=None, port=None, user=None, password=None,
                 excluded_indices=["whatch_db", "security-auditlog"],
                 protected_indices=["aselsan_database", "celebrity_database"]):
        if ip is None:
            self.ip = 'localhost'
        else:
            self.ip = ip
        if port is None:
            self.port = '9200'
        else:
            self.port = port
        if (user is not None) and (password is not None):
            hosts=f"http://{user}:{password}@{ip}:{port}/"
        else:
            hosts=f"http://{ip}:{port}/"
        
        self.excluded_indices = excluded_indices
        self.protected_indices = protected_indices
        
        self.es = create_client(host=ip, http_port=port)
        self.indices = []
        self.update_indices()
        
        self.mappings = {
                      "properties": {
                          "model_name": {"type": "keyword"},
                          "model_folder": {"type": "keyword"},
                          "model_name_extra": {"type": "keyword"},
                          "additional_file_names": {"type": "text"},
                          "model_format": {"type": "keyword"},
                          "production": {"type": "boolean"},
                          "producer": {"type": "keyword"},
                          "register_time": {"type":   "date",
                                           "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd"},
                          "version": {"type": "float"},
                          "main_task": {"type": "keyword"},
                          "sub_tasks" : {"type": "text"},
                          "info" : {"type": "text"},
                          }
                      }
    
    def update_indices(self):
        self.indices = []
        protected_indices = []
        excluded_indices = []
        for index in self.es.indices.get('*'):
            if not index.startswith('.'):
                self.indices.append(index)
                for protected_index in self.protected_indices:
                    if index.startswith(protected_index):
                        protected_indices.append(index)
                for excluded_index in self.excluded_indices:
                    if index.startswith(excluded_index):
                        excluded_indices.append(index)
        for p_index in protected_indices:
            if p_index not in self.protected_indices:
                self.protected_indices.append(p_index) 
        for e_index in excluded_indices:
            if e_index not in self.excluded_indices:
                self.excluded_indices.append(e_index) 

    def create_index(self, index, force=False, mapping=None):
        if mapping is None:
            mapping = {
                      "settings": {
                        "index": {
                          "number_of_shards": 4
                        }
                      },
                      "mappings": self.mappings
                    }
        if index not in self.indices:
            self.es.indices.create(index=index, body=mapping)
            self.update_indices()
        else:
            if force:
                self.delete_index(index)
                self.es.indices.create(index=index, body=mapping)
                self.update_indices()

    def delete_index(self, index):
        res = False
        if index in self.indices:
            self.es.indices.delete(index=index)
            self.update_indices()
            res = True
        return res
            
    def add_doc(self, database_name, doc_id, model_name, model_folder, 
                model_name_extra, additional_file_names, model_format, 
                production, producer, register_time, version, main_task, 
                sub_tasks, info):
        doc = {"model_name": model_name,
                "model_folder": model_folder,
                "model_name_extra": model_name_extra,
                "additional_file_names": additional_file_names,
                "model_format": model_format,
                "production": production,
                "producer": producer,
                "register_time": register_time,
                "version": version,
                "main_task": main_task,
                "sub_tasks" : sub_tasks,
                "info" : info}
        self.es.create(index=database_name, id=doc_id, body=doc)


    def _delete_by_query(self, index, query_body):
        try:
            self.es.delete_by_query(index=index, body=query_body)
            res = True
        except:
            res = False
        return res

    def delete_by_ids(self, index, ids):
        query = {"query": {"terms": {"_id": ids}}}
        res = self.es.delete_by_query(index=index, body=query)
        return res

