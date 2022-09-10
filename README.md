[<img align="left" src="https://www.datocms-assets.com/5203/1586261721-starmon-5-no-border.svg" width="70px" height="105px">](https://www.quantum-inspire.com/backends/starmon-5/) 
# inspire-experiments
#### Helper library to conveniently run, save and analyze quantum experiments on [QuTech's Quantum Inspire](https://www.quantum-inspire.com/)

This package is mostly focused on using the [Starmon-5](https://www.quantum-inspire.com/backends/starmon-5/) hardware backend, 
but is easily applicable/extendable to using other backends of _Quantum Inspire_.
The main thing is that in most places, it is assumed/hardcoded that the backend has 5 qubits.

Here we use [_Qiskit_](https://github.com/Qiskit/qiskit) to generate the experiment circuits and send jobs just for convenience,
but it is also possible to use _Quantum Inspire's_ native cQUASM language as shown in [this example](https://www.quantum-inspire.com/kbase/using-api-and-sdk/). 
The main difference for the user is that one would have to operate on strings instead of objects to make the circuit 
(Qiskit can create a QASM representation of its circuit objects).

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

### Login

Once you have stored your API token locally (check the [Knowledge Base](https://www.quantum-inspire.com/kbase/using-api-and-sdk/) for how to do that)
you can login using 
```python
from inspire_experiments import *
api, backend = inspire_login()
```
and check for the Starmon-5 backend's status using
```python
print(get_starmon_status())
```

### Sending jobs

After building the circuit you wish to run and sending the job using 
```python
job = qiskit.execute(circuit, shots=2**14, optimization_level=0, backend=backend)
```
This method will return immediately after posting the job to the API, but this doesn't mean the job is already done.
It only gives you back a handle to the job.
The next time you try to access `job.result()`, the command will not return until `job.status()` is `FINISHED`.
This will either take ~40-60 seconds if the job started to execute immediately, or longer if there are other jobs 
in the queue (unfortunately it is currently not possible to get any information about the status of the queue from the API).
If you changed your mind, you can terminate a job with `job.cancel()`.

Note that passing `optimization_level=0` above is needed to disable the optimization part of the Qiskit circuit transpiler and interpret your circuit literally _as is_
(except for the mapping to hardware topology, check the [Qiskit `PassManager` source code](https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/transpiler/preset_passmanagers/level0.py) for details).

### Saving data

You can save the results with
```python
ExperimentData.save_job_result(job, exp_name="my_experiment_V1", header=get_file_header(circuit))
```
By default the `ExperimentData` class saves data as CSV and JSON files under `./data/` in the current directory.

By passing the output of `get_file_header()` as the optional `header` argument,
each file will contain a text representation of the circuit that was used in the job.
This is useful to help remember which data belongs to which experiment.

### Loading data

To load back experiment data saved earlier, simply call
```python
ExperimentData.get_csv_data("my_experiment_V1_RAW.csv")
```
This will return an array of binary strings representing the measurement result of each in [little endian](https://www.quantum-inspire.com/kbase/binary-register/)
format.

If the circuit contains multiple subsequent measurements on any qubit, the result format will be such that each row 
of the array contains all shots corresponding to one (or several) measurements executed at the same time.
To understand the scheduling of measurements (and other operations) see the scheduling section of the [Knowledge Base](https://www.quantum-inspire.com/kbase/starmon-5-operational-specifics/).


## Examples & more

Check the `./notebooks` directory for more detailed and advanced usage examples of actual experiments you can run.

For more documentation about _Quantum Inspire_ itself, visit the [Knowledge Base](https://www.quantum-inspire.com/kbase/introduction-to-quantum-computing).
It also contains detailed examples of experiments that you can run yourself from Jupyter Notebooks (which are hosted [here](https://github.com/QuTech-Delft/quantum-inspire-examples)).
