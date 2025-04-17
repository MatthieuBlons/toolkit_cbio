import os
import json
import h5py
import pandas as pd

def print_attrs(name, obj):
    print(f"Object: {name}")
    for key, value in obj.attrs.items():
        print(f"    Attribute - {key}: {value}")

def print_dict(dict, name = None):
    if name:
        print(f"In {name}: ")
    else:
        print(f"In Dictionary: ")
    for key, value in dict.items():
        print(f"    {key}: {value}")