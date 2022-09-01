
# inspire-experiments

#### Code and notebooks for running, saving and analyzing experiments on [QuTech Quantum Inspire](https://www.quantum-inspire.com/).

This package is mostly focused on using the [Starmon-5](https://www.quantum-inspire.com/backends/starmon-5/) hardware backend, 
but is easily extendable to using other backends of _Quantum Inspire_.

Here we use [_Qiskit_](https://github.com/Qiskit/qiskit) to generate the experiment circuits and send jobs just for convenience,
but it is also possible to use _Quantum Inspire's_ native cQUASM language as shown in [this example](https://www.quantum-inspire.com/kbase/using-api-and-sdk/). 

## Installation

To use this package, just clone it locally to a directory of your choice
```commandline
❯ git clone git@github.com:codecrap/inspire-experiments.git
```

and install it as local package in your python environment (`-e` for editable)
```commandline
❯ python -m  pip install -e ./inspire-experiments
```

## Usage

Once you have stored your API token locally (check the [Knowledge Base](https://www.quantum-inspire.com/kbase/using-api-and-sdk/) for how to do that)
you can login using 
```python
from inspire_experiments import *
api, backend = inspire_login()
```
and check for the backend's status using
```python
print(get_starmon_status())
```

After building the circuit you wish to run and sending the job using 
```python
job = qiskit.execute(circuit, shots=2**14, optimization_level=0, backend=backend)
```
Note that passing `optimization_level=0` is needed to disable the optimization part of the Qiskit circuit transpiler and interpret your circuit literally _as is_
(except for the mapping to hardware topology, check the [Qiskit pass manager](https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/transpiler/preset_passmanagers/level0.py) for details).


You can save the results (by default saves as CSV and JSON files under `./data/`) with
```python
ExperimentData.save_job_result(job, exp_name="my_experiment_V1", header=get_file_header(circuit))
```
By passing the output of `get_file_header()` as the optional `header` argument,
each file will contain a text representation of the circuit that was used in the job.
This is useful to help remember which data belongs to which experiment.


Check the `notebooks` directory for more detailed usage examples.
