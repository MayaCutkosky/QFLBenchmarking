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


import torch
class Model(torch.nn.Module):
    def __init__(self, dev, num_classes=1):
        '''
        

        Parameters
        ----------
        dev : qml device object
        x : x input
        params : model params

        Returns
        -------
        predicted y output

        '''
        super().__init__()
#        @qml.qnode(dev, interface = "pytorch", diff_method = 'spsa' )
        def circuit(x0, p):
            qml.AmplitudeEmbedding(x0, wires = range(8), normalize = True)
            create_model_circuit(p)
            return qml.expval(qml.PauliZ(0))
        
        qnodes = [qml.QNode(circuit, dev, interface="torch") for _ in range(num_classes)]

        
        self.circuits = qnodes
        self.num_classes = num_classes
        self.params = torch.nn.Parameter(torch.randn(num_classes, 64))

    @staticmethod
    def init_params(num_classes=1):
        return {'params' : torch.randn(num_classes, 64)}
    
    def forward(self, x):
        
        output = []
        for x0 in x:
            col = []
            for p in self.params:
                col.append(self.circuit(x0, p))
            output.append(torch.concat(col))
        return torch.stack(output)

    @staticmethod
    def loss_fun(y_true, y_pred, num_classes=1):
        '''
        Parameters
        ----------
        y_true : 
        y_pred : 
        num_classes :
    
        Returns
        -------
        loss
    
        '''
        y_true = torch.nn.functional.one_hot(y_true, num_classes).double()
        return torch.nn.MSELoss()(y_pred, y_true)
#Jax version
# import jax
# import optax
# def run(dev, x, params):
#     '''
    

#     Parameters
#     ----------
#     dev : qml device object
#     x : x input
#     params : model params

#     Returns
#     -------
#     predicted y output

#     '''
#     @qml.qnode(dev, interface = "jax")
#     def circuit(x0, p):
#         qml.AmplitudeEmbedding(x0, wires = range(8), normalize = True)
#         create_model_circuit(p)
#         return qml.expval(qml.PauliZ(0))
#     return jax.lax.map(lambda x_i : jax.lax.map(lambda p_c: circuit(x_i, p_c), params ), x)

# def loss_fun(y_true, y_pred, num_classes=1):
#     '''
    

#     Parameters
#     ----------
#     y_true : 
#     y_pred : 
#     num_classes :

#     Returns
#     -------
#     loss

#     '''
#     y_true = jax.nn.one_hot(y_true, num_classes)
#     return optax.losses.squared_error(y_pred, y_true).mean()

# def init_params(rng, num_classes=1):
#     return jax.random.uniform(rng, [num_classes, 64])