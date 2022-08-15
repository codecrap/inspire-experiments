from quantuminspire.credentials import get_token_authentication
from quantuminspire.api import QuantumInspireAPI
from quantuminspire.qiskit import QI

from qiskit import QuantumCircuit

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
