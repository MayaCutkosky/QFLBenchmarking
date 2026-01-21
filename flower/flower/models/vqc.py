#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 17:33:13 2026

@author: maya
"""

import pennylane as qml
import numpy as np

import torch
class Model(torch.nn.Module):
    def __init__(self, dev, num_classes=1, num_qubits = 8):
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
        


        
        
        def layer(W):
            """
            Apply a layer of rotations and CNOT gates.
            """
            for i in range(num_qubits):
                qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
            for j in range(num_qubits - 1):
                qml.CNOT(wires=[j, j + 1])
            if num_qubits >= 2:
                # Apply additional CNOT to entangle the last with the first qubit
                qml.CNOT(wires=[num_qubits - 1, 0])
        
        # @qml.qnode(device=dev, interface="torch")
        def circuit(weights, feat=None):
            """
            Quantum circuit using amplitude embedding.
            """
            qml.AmplitudeEmbedding(feat, range(num_qubits), pad_with=0.0, normalize=True)
            for W in weights:
                layer(W)
            return qml.expval(qml.PauliZ(0))
        
        qnodes = [qml.QNode(circuit, dev, interface="torch") for _ in range(num_classes)]

        
        self.circuits = qnodes
        self.num_classes = num_classes
        self.num_layers = 6
        self.num_qubits = num_qubits
        for key, val in self.init_params(num_classes, self.num_layers, num_qubits).items():
            setattr(self,key, torch.nn.Parameter(torch.Tensor(val)))

    @staticmethod
    def init_params(num_classes=1, num_layers = 6, num_qubits = 8):
        params = {
                'weights' : 0.1 * np.random.randn(num_classes, num_layers, num_qubits, 3),
                'biases' : 0.1 * np.ones(num_classes)
            }
        return params
    
    def forward(self, x):
        
        output = []
        for x0 in x:
            col = []
            for c, w, b in zip(range(self.num_classes), self.weights, self.biases): 
                col.append(self.circuits[c](w, x0)+ b )
            output.append(torch.stack(col))
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
        
        return torch.nn.MultiMarginLoss(margin = 0.15)(y_pred, y_true)