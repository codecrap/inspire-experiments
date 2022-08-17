from __future__ import annotations

import itertools as it

from quantuminspire.credentials import get_token_authentication
from quantuminspire.api import QuantumInspireAPI
from quantuminspire.qiskit import QI
from quantuminspire.qiskit.backend_qx import QuantumInspireBackend
from quantuminspire.qiskit.qi_job import QIJob

import qiskit
from qiskit import QuantumCircuit

from .data_utils import ExperimentData

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def get_starmon_status(qi_api: QuantumInspireAPI) -> str:
    # entry for Starmon-5
    status = qi_api.get_backend_types()[1]['status']
    # print(status)
    return status

def inspire_login() -> tuple[QuantumInspireAPI, QuantumInspireBackend]:
    QI_URL = r'https://api.quantum-inspire.com/'
    authentication = get_token_authentication()
    qi = QuantumInspireAPI(QI_URL, authentication)
    QI.set_authentication(authentication)
    starmon5 = QI.get_backend('Starmon-5')
    print("Backend status: ", get_starmon_status(qi))
    return qi, starmon5

def get_file_header(circuit: QuantumCircuit) -> str:
    header = '\n'.join(['# ' + line for line in str(circuit.draw(output='text')).split('\n')])
    return header


def measure_readout_correction(
        qubits: tuple[int] | list[int],
        backend: QuantumInspireBackend,
        exp_basename: str="readout_correction",
        extra_label: str=None,
        show_circuit: bool=True,
        send_jobs: bool=True
) -> tuple[list[QIJob], list[QuantumCircuit]]:
    """ Measures readout correction calibration points for arbitrary combination of qubits."""
    jobs, circuits = [], []
    for state in it.product(['0', '1'], repeat=len(qubits)):
        circuit = QuantumCircuit(5,5)
        for i, s in enumerate(state):
            if s == '1':
                circuit.x(qubits[i])
            else:
                circuit.id(qubits[i])
        circuit.barrier(range(5))
        for qb in qubits:
            circuit.measure(qb, qb)

        circuits += [circuit]
        if show_circuit:
            display(circuit.draw(output='mpl', interactive=False))

        header = get_file_header(circuit)
        exp_name = exp_basename + f"_qbs{qubits}"
        exp_name = exp_name + extra_label if extra_label else exp_name
        exp_name += f"_state{''.join(state)}"
        log.info(exp_name)

        if send_jobs:
            log.info(f"Measuring readout correction: state {state}")
            job = qiskit.execute(circuit, shots=2**14, optimization_level=0, backend=backend)
            ExperimentData.save_job_result(job, exp_name, header)
            jobs += [job]

    return jobs, circuits
