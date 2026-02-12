#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 07:32:28 2026

@author: maya
"""
from flwr.common.record import RecordDict, ArrayRecord, MetricRecord
from flwr.common.message import Message
from quickstart_pennylane.task import QuantumNet, load_data,  train
import torch
import tomllib 
with open('pyproject.toml', 'rb') as f:
    run_config = tomllib.load(f)['tool']['flwr']['app']['config']

n_qubits = run_config.get("n-qubits", 4)
n_layers = run_config.get("n-layers", 3)

lr = 0.001

# Load the model and initialize it with the received weights
model = QuantumNet(num_classes=10, n_qubits=n_qubits, n_layers=n_layers)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the data
partition_id = 0
num_partitions = 1
batch_size = run_config["batch-size"]
trainloader, valloader = load_data(partition_id, num_partitions, batch_size)

# uncomment to print the training and validation data size
# print(f"Client {partition_id}/{num_partitions} starting training...")
# print(f"Training data size: {len(trainloader.dataset)}")
# print(f"Validation data size: {len(valloader.dataset)}")
from flwr.serverapp.strategy import FedAvg
strategy = FedAvg()
# Call the training function
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(columns = ['train_loss', 'val_loss', 'val_accuracy'])
state_dict = model.state_dict()
for epoch in range(run_config['num-server-rounds']):
    messages = []
    for partition_id in range(num_partitions):
        model.load_state_dict(state_dict)
        trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
        results = train(
            model,
            trainloader,
            valloader,
            20,
        #    run_config["local-epochs"],
            lr,
            device,
        )
        metrics = {
            "train_loss": results["train_loss"],
            "val_loss": results["val_loss"],
            "val_accuracy": results["val_accuracy"],
            "num-examples": len(trainloader.dataset),
        }
        metrics= MetricRecord(metrics)
        messages.append( Message(RecordDict({'arrays':ArrayRecord(model.state_dict()), 'metrics': metrics}), partition_id, 'train') )
    out = strategy.aggregate_train(epoch, messages)
    state_dict =  out[0].to_torch_state_dict()
    df.loc[len(df)] = [out[1]['train_loss'], out[1]['val_loss'], out[1]['val_accuracy']]

df[['train_loss', 'val_loss']].plot()
plt.show()
df['val_accuracy'].plot()
plt.show()
    # if input("stop?"):
    #     break
    
