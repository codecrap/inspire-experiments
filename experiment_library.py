import time
import json

import numpy as np
import pandas as pd
from pprint import pprint
from typing import List
import itertools as it
import matplotlib.pyplot as plt

import qiskit
import qiskit.tools.jupyter
from qiskit import QuantumCircuit
import qiskit.circuit.library.standard_gates as gates
from qiskit.tools.visualization import circuit_drawer, plot_histogram

from quantuminspire.credentials import get_token_authentication
from quantuminspire.api import QuantumInspireAPI
from quantuminspire.qiskit import QI

from tqdm import tqdm


def get_starmon_status(qi_api: QuantumInspireAPI) -> str:
    # entry for Starmon-5
    status = qi_api.get_backend_types()[1]['status']
    print(status)
    return status

def inspire_login() -> QuantumInspireAPI:
    QI_URL = r'https://api.quantum-inspire.com/'
    authentication = get_token_authentication()
    qi = QuantumInspireAPI(QI_URL, authentication)
    QI.set_authentication(authentication)
    starmon5 = QI.get_backend("Starmon-5")
    print(starmon5.status())
    get_starmon_status()
    return qi

def get_file_header(circuit: QuantumCircuit) -> str:
    header = '\n'.join(['# ' + line for line in str(circuit.draw(output='text')).split('\n')])
    return header
