from __future__ import annotations
from dataclasses import dataclass

import os
import time
import json

import numpy as np
from pathlib import Path
from datetime import datetime, date

from quantuminspire.qiskit.qi_job import QIJob
from matplotlib.figure import Figure

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


DEFAULT_DATA_DIR = './data/'
DEFAULT_PLOT_DIR = './plots/'
NUM_QUBITS = 5


@dataclass
class ExperimentData:
    _data_dir = DEFAULT_DATA_DIR
    _plot_dir = DEFAULT_PLOT_DIR

    def __init__(self, data_dir: str | Path = None, plot_dir: str | Path = None):
        self._data_dir = data_dir if data_dir is not None else Path(DEFAULT_DATA_DIR)
        self._plot_dir = plot_dir if plot_dir is not None else Path(DEFAULT_PLOT_DIR)
        self._data = None
        self._hist = None
        self._timestamp = date.today()


    @property
    def data_dir(self):
        return self._data_dir


    @data_dir.setter
    def data_dir(self, d: str | Path):
        self._data_dir = Path(d)


    @property
    def plot_dir(self):
        return self._plot_dir


    @plot_dir.setter
    def plot_dir(self, d: str | Path):
        self._plot_dir = Path(d)


    @property
    def timestamp(self):
        return self._timestamp


    @timestamp.setter
    def timestamp(self, t: str | datetime):
        if isinstance(t, str):
            self._timestamp = date.fromtimestamp(t)
        elif isinstance(t, datetime):
            self._timestamp = t
        else:
            raise ValueError("Wrong format for timestamp!")


    @property
    def data(self):
        return self._data


    @data.setter
    def data(self, d: np.ndarray | list):
        self._data = np.asarray(d)


    @property
    def hist(self):
        return self._hist


    @hist.setter
    def hist(self, d: dict):
        self._hist = d


    @classmethod
    def save_job_result(
            cls,
            job: QIJob,
            exp_name: str,
            header: str = None,
            save_counts_prob_separate: bool = False,
            directory: str | Path = _data_dir
    ) -> QIJob:
        log.info(f"Saving results for job {job.job_id()} in {directory}")
        Path(directory).mkdir(parents=True, exist_ok=True)
        np.savetxt(
            fname=directory / Path(exp_name + '_RAW.csv'),
            X=np.array(job.result().data()['memory_multiple_measurement']),
            delimiter=',',
            fmt='%s',
            header=header)

        probs = job.result().get_probabilities_multiple_measurement()
        counts = job.result().data()['counts_multiple_measurement']

        if save_counts_prob_separate:
            # old style: separate files for probability and hist dicts
            with open(directory / Path(exp_name + '_PROB.json'), 'w') as f:
                f.write(header + '\n')
                f.write(json.dumps(probs))

            with open(directory / Path(exp_name + '_COUNTS.json'), 'w') as f:
                f.write(header + '\n')
                f.write(json.dumps(counts))
        else:
            # new style: first line hist dict, second line probability dict
            with open(directory / Path(exp_name + '_HIST.json'), 'w') as f:
                f.write(header + '\n')
                f.write(json.dumps(counts))
                f.write('\n')
                f.write(json.dumps(probs))

        cls.timestamp = date.today()
        return job

    @classmethod
    def save_fig(
        cls,
        fig: Figure,
        name: str,
        directory: str | Path = _plot_dir
    ) -> Figure:
        log.info(f"Saving figure <{name} - {fig}> in {directory}")
        Path(directory).mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / Path(name + "_PLOT.png"), format='png', dpi=200)
        return fig

    @classmethod
    def get_json_data(
            cls,
            filename: str | Path,
            comment: str = '#',
            directory: str = _data_dir
    ) -> dict:
        with open(directory / Path(filename), 'r') as f:
            data_str = [line for line in f.read().split('\n') if line[0] != comment]

        # will contain either one line with dict of counts or probs for old saving style,
        # or two lines with one dict in each, first counts then probs
        # (see save_job_result())
        dicts = [json.loads(line) for line in data_str]

        cls.hist = dicts
        cls.timestamp = time.ctime(os.path.getmtime(directory / Path(filename)))
        return dicts


    @classmethod
    def get_csv_data(
            cls,
            filename: str,
            comment: str = '#',
            single_qubit: bool = False,
            use_string_repr: bool = True,
            directory: str = _data_dir
    ) -> np.ndarray:
        data_hex = np.loadtxt(directory / Path(filename), comments=comment, dtype=str, delimiter=',')
        msmt_shape = data_hex.shape
        # create binary strings of length NUM_QUBITS
        data_bin = np.array(list(map(lambda h: str(bin(int(h, 16)))[2:].zfill(NUM_QUBITS), data_hex.flatten()))).reshape(msmt_shape)

        if not use_string_repr:
            # convert strings to arrays of binary integers, which creates an extra dimension in the data_bin array
            data_bin = np.array(list(map(lambda s: np.fromiter(s, dtype=int), data_bin.flatten()))).reshape((*msmt_shape, NUM_QUBITS))

        if single_qubit:
            # if only one qubit is measured, just create integer 0s/1s
            # this will disregard any specific qubits in the 1-state and just return 1 if any one qubit was in 1-state
            # the lambda function produces a decimal representation of the binary (hex) string
            data_bin = np.array(list(map(lambda d: int(d, 16), data_hex.flatten()))).reshape(msmt_shape)
            data_bin = (data_bin > 0).astype(int)

        cls.data = data_bin
        cls.timestamp = time.ctime(os.path.getmtime(directory / Path(filename)))
        return data_bin
