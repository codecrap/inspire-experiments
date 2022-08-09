import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from typing import List
import itertools as it

DEFAULT_DATA_DIR = './data/'
DEFAULT_PLOT_DIR = './plots/'


def save_exp_result(
        job,
        exp_name: str,
        header: str=None,
        directory: str=DEFAULT_DATA_DIR
):
    np.savetxt(fname=directory + exp_name + '_RAW.csv',
               X=np.array(job.result().data()['memory_multiple_measurement']),
               delimiter=',',
               fmt='%s',
               header=header)

    with open(directory + exp_name + '_PROB.json', 'w') as f:
        f.write(header + '\n')
        f.write(json.dumps(job.result().get_probabilities_multiple_measurement()))

    with open(directory + exp_name + '_COUNTS.json', 'w') as f:
        f.write(header + '\n')
        f.write(json.dumps(job.result().data()['counts_multiple_measurement']))

    return


def get_json_data(
        filename: str,
        directory: str=DEFAULT_DATA_DIR,
        comment: str='#'
):
    with open(directory + filename, 'r') as f:
        data_str = [line for line in f.read().split('\n') if line[0] != comment]
        data_dict = json.loads(data_str[0])
    return data_dict


def get_csv_data(
        filename: str,
        directory: str=DEFAULT_DATA_DIR,
        comment: str='#',
        single_qubit: bool=False,
        use_string_repr: bool=True
):
    data_hex = np.loadtxt(directory + filename, comments=comment, dtype='<U3', delimiter=',')
    # create binary strings of length 5
    data_bin = np.array(list(map(lambda h: str(bin(int(h, 16)))[2:].zfill(5), data_hex.flatten()))).reshape(data_hex.shape)

    if not use_string_repr:
        # convert strings to arrays of binary integers, which creates an extra dimension in the data_bin array
        data_bin = np.array(map(lambda s: np.fromiter(s, dtype=int), data_bin.flatten())).reshape(data_hex.shape)

    # data_bin = np.array(["{0:05b}".format(b) for b in list(map(lambda d: int(d, 16), data_hex.flatten()))]).reshape(data_hex.shape)
    # data_bin = np.frombuffer(data_hex)
    # data_bin = data_hex.astype(np.int16)
    # data_bin = np.array(list(map(lambda x: bin(int(str(x), 16)), data_hex)))
    # measured = np.log2(data_bin) #if data_bin > 0 else 0
    # data_bin = data_bin > 0

    if single_qubit:
        # if only one qubit is measured, just create integer 0s/1s
        # this will disregard any specific qubits in the 1-state and just return 1 if any one qubit was in 1-state
        # the lambda function produces a decimal representation of the binary (hex) string
        data_bin = np.array(list(map(lambda d: int(d, 16), data_hex.flatten()))).reshape(data_hex.shape)
        data_bin = (data_bin > 0).astype(int)

    return data_bin
