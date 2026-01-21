#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 07:32:28 2026

@author: maya
"""

from flower.task import get_data, get_device, train, load_model
from matplotlib import pyplot as plt

config = {
    "lr" : 0.01,
    "num_qubits" : 6,
    "features" : 64,
    }

dev = get_device("sim", config['num_qubits'])
model = load_model('vqc')(dev, 10, num_qubits = config['num_qubits'])

data = get_data('mnist', 0, 1, config['features'])

optim = None
hist, optim = train(model, data.iter(16), lr = config['lr'], optim = optim)
plt.plot(hist)
plt.show()
