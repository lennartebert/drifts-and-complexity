import json


def load_data_dictionary(path):
    with open(path, 'r') as f:
        return json.load(f)
    