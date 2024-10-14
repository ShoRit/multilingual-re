import os, json, argparse, itertools, pickle, itertools, dill, random
from glob import glob
import numpy as np
import pandas as pd
from collections import defaultdict as ddict
from tqdm import tqdm
import regex as re
from joblib import Parallel, delayed
from intervals import *
from torch.nn.utils.rnn import pad_sequence
import pathlib
from pprint import pprint

def partition(lst, n):
    if n == 0: return lst
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def get_chunks(inp_list, chunk_size):
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))

def load_pickle(file):
    with open(file,'rb') as handle:
        return pickle.load(handle)
    
def dump_pickle(data, file):
    with open(file, 'wb') as handle:
        pickle.dump(data, handle)
    print('Pickle dumped successfully')

def load_dill(file):
    with open(file,'rb') as handle:
        return dill.load(handle)

def load_json(file):
    with open(file) as f:
        return json.load(f)

def dump_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)
    print('Json dumped successfully')


def dump_dill(data, file):
    with open(file, 'wb') as handle:
        dill.dump(data, handle)
    print('Dill dumped successfully')

def load_deprels(file='enh_dep_rel.txt', enhanced=False):
    dep_dict    = {}
    lines       = open(file).readlines()
    for line in tqdm(lines):
        data                    = line.strip().split()
        rel_name                = data[0]
        if enhanced:
            if rel_name.endswith(':'):
                rel_name        = rel_name[:-1]
        else:
            rel_name =  rel_name.split(':')[0]                
        if rel_name not in dep_dict:
            dep_dict[rel_name]      = len(dep_dict)
    
    if 'STAR' not in dep_dict:
        dep_dict['STAR']                = len(dep_dict)
    return dep_dict


def get_recursively(search_dict, field):
    """
    Takes a dict with nested lists and dicts,
    and searches all dicts for a key of the field
    provided.
    """
    fields_found = []

    for key, value in search_dict.items():

        if key == field:
            fields_found.append(value)

        elif isinstance(value, dict):
            results = get_recursively(value, field)
            for result in results:
                fields_found.append(result)

        elif isinstance(value, (list,tuple)):
            for item in value:
                if isinstance(item, dict):
                    more_results = get_recursively(item, field)
                    for another_result in more_results:
                        fields_found.append(another_result)

    return fields_found


def partition(lst, n):
    if n == 0: return lst
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def get_chunks(inp_list, chunk_size):
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))


def check_file(filename):
    return pathlib.Path(filename).is_file()
