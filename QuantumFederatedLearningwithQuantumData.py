#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:36:16 2025

@author: maya

circuit creating functions modified from https://github.com/MahdiChehimi/Quantum-Federated-Learning-with-Quantum-Data/blob/main/qfl_paper.py
which is the paper's official github site. I also somewhat modified the argument names to make more sense.
"""
NUM_QUBITS = 8

import pennylane as qml
import numpy as np

# Cirq quantum circuit to create an arbitrary sinle-qubit unitary
def one_qubit_unitary(qbit, params):
    qml.RX(params[0]*np.pi,wires = qbit)
    qml.RY(params[1]*np.pi,wires = qbit)
    qml.RZ(params[2]*np.pi,wires = qbit)

# Cirq quantum circuit to create an arbitrary two-qubit unitary
def two_qubit_unitary(qbits, params):
    one_qubit_unitary(qbits[0], params[0:3])
    one_qubit_unitary(qbits[1], params[3:6])
    qml.IsingZZ(params[6],qbits)
    qml.IsingYY(params[7],qbits)
    qml.IsingXX(params[8],qbits)
    one_qubit_unitary(qbits[0], params[9:12])
    one_qubit_unitary(qbits[1], params[12:])

# Cirq quantum circuit to perform a parametrized pooling operation.
# This operation reduces entanglement down from two-qubits, to a single qubit.
def two_qubit_pool(source_qubit, sink_qubit, params):
    one_qubit_unitary(sink_qubit, params[0:3]) #sink_basis_selector
    one_qubit_unitary(source_qubit, params[3:6]) #source_basis_selector 
    
    qml.CNOT([source_qubit, sink_qubit])
    one_qubit_unitary(sink_qubit, params[0:3] ** -1) #sink_basis_selector ** -1

"""#### Define the Quantum convolution layer """

# A cascade application of the two-qubit unitary to all pairs of qubits in 'bits' 
def quantum_conv_circuit(qbits, params):
    for first, second in zip(qbits[0::2], qbits[1::2]):
        two_qubit_unitary([first, second], params)
    for first, second in zip(qbits[1::2], qbits[2::2] + [qbits[0]]):
        two_qubit_unitary([first, second], params)

"""#### Define the Quantum pooling layer"""

# A circuit that learns to pool the relevant information from two qubits onto 1
def quantum_pool_circuit(source_bits, sink_bits, params):
    for source, sink in zip(source_bits, sink_bits):
        two_qubit_pool(source, sink, params)

"""### Define the QCNN model"""

# The model consists of a sequence of convolution and pooling layers that 
# gradually shrink over time
def create_model_circuit(params):
    num_qubits = NUM_QUBITS
    qubits = [i for i in range(num_qubits)]
    quantum_conv_circuit(qubits, params[0:15])
    quantum_pool_circuit(qubits[:4], qubits[4:], params[15:21])
    quantum_conv_circuit(qubits[4:], params[21:36])
    quantum_pool_circuit(qubits[4:6], qubits[6:], params[36:42])
    quantum_conv_circuit(qubits[6:], params[42:57])
    quantum_pool_circuit([qubits[6]], [qubits[7]], params[57:63])





from functools import lru_cache


@lru_cache(maxsize=1)
def _get_ibm_service_and_backends():
    """Initialize IBM Runtime service and cache available backends."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()  # Requires IBM credentials set up
    backends = service.backends(simulator=False, operational=True)

    if not backends:
        raise RuntimeError("No operational IBM Quantum devices available.")

    print(f"🔗 IBM Runtime service initialized. {len(backends)} real devices available.")
    return service, backends


#@lru_cache(maxsize=None)
def get_device(backend: str, num_qubits: int, device_id: int = 0):
    """
    Returns a PennyLane device object depending on the backend.

    Parameters
    ----------
    backend : str
        Either 'sim' for simulation (Qiskit Aer) or 'real' for real IBM hardware.
    num_qubits : int
        Number of qubits for the device.
    device_id : int, optional
        Used to pick a specific real backend (default: 0).
    """

    if backend == "sim":
        # Use Qiskit Aer simulator (local backend)
        dev = qml.device("qiskit.aer", wires=num_qubits)
        print(f"Using Qiskit Aer simulator with {num_qubits} qubits.")
        return dev

    elif backend == "real":
        service, backends = _get_ibm_service_and_backends()
        
        backend_name = backends[device_id].name 
        print(f"Using IBM Quantum backend '{backend_name}' with {num_qubits} qubits.")

        dev = qml.device(
            "qiskit.ibmq",
            wires=num_qubits,
            backend=backend_name,
            ibmqx_token=None  # credentials handled by QiskitRuntimeService
        )
        return dev

    else:
        raise ValueError("Invalid backend specified. Choose 'sim' or 'real'.")


import optax
import jax
def make_federated_model(backend, num_classes = 1):
    def init_fun(rng): #create parameters
        return jax.random.uniform(rng, [num_classes, 64])
    
    def apply_fun(params, batch, rng=None): #use parameters
        client_id = batch['client_id'][0]
        x = batch['x']
        dev = get_device(backend, 8, client_id)
        @qml.qnode(dev, interface = "jax")
        def circuit(x0, p):
            qml.AmplitudeEmbedding(x0, wires = range(8), normalize = True)
            create_model_circuit(p)
            return qml.expval(qml.PauliZ(0))
        print(batch['x'][0], batch['y'][0], batch['client_id'])
        return jax.lax.map(lambda x_i : jax.lax.map(lambda p_c: circuit(x_i, p_c), params ), x)

    
    def loss_fun(batch, pred):
        target = jax.nn.one_hot(batch['y'], num_classes)
        return optax.losses.squared_error(pred, target)
        
    
    
    return fedjax.Model(
      init=init_fun,
      apply_for_train=apply_fun,
      apply_for_eval=apply_fun,
      train_loss=loss_fun,
      eval_metrics={'accuracy': fedjax.metrics.Accuracy()})

from dataset import EMNIST_dataset as make_dset
import fedjax
def run(**args):

    model = make_federated_model()

    dset = make_dset()
    
    grad_fn = fedjax.model_grad(model)
    client_opt = fedjax.optimizers.adam(0.1)
    server_opt = fedjax.optimizers.adam(0.1)
    batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=8)
    fed_alg = fedjax.algorithms.fed_avg.federated_averaging(grad_fn, client_opt, server_opt, batch_hparams)
    
    rng = jax.random.PRNGKey(0)
    init_params = model.init(rng)
    init_server_state = fed_alg.init(init_params)
    
    client_inputs = dset
    updated_server_state, client_diagnostics = fed_alg.apply(init_server_state, client_inputs)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Quantum computing experiment configuration"
    )

    parser.add_argument(
        "--backend",
        choices=["real", "sim"],
        default="sim",
        help="Select backend: 'real' for a real quantum computer or 'sim' for a simulator (default: sim)"
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=3,
        help="Number of clients (default: 3)"
    )

    parser.add_argument(
        "--num-qubits",
        type=int,
        default=8,
        help="Number of qubits (default: 8)"
    )

    args = parser.parse_args()
    
    
    

