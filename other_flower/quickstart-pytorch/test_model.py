#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 07:32:28 2026

@author: maya
"""
from flwr.common.record import RecordDict, ArrayRecord, MetricRecord
from flwr.common.message import Message
from pytorchexample.task import Net, load_data,  train, test
import torch
import tomllib 
with open('pyproject.toml', 'rb') as f:
    run_config = tomllib.load(f)['tool']['flwr']['app']['config']

n_qubits = run_config.get("n-qubits", 4)
n_layers = run_config.get("n-layers", 3)


# Load the model and initialize it with the received weights
model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the data
num_partitions = 10
batch_size = run_config["batch-size"]

from flwr.serverapp.strategy import FedAvg
strategy = FedAvg()
# Call the training function
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(columns = ['train_loss', 'val_loss', 'val_accuracy'])
lr = run_config['learning-rate']
state_dict = model.state_dict()
for epoch in range(run_config['num-server-rounds']):
    messages = []
    for partition_id in range(num_partitions):
        model.load_state_dict(state_dict)
        trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
        loss = train(
            model,
            trainloader,
            run_config["local-epochs"],
            lr,
            device,
        )
        lr = lr
        eval_loss, eval_acc = test(
            model,
            valloader,
            device,
        )
        metrics = MetricRecord({'train_loss': loss, 
                                'eval_loss' : eval_loss, 
                                'eval_acc' : eval_acc,
                                'num-examples' : len(trainloader.dataset)})
        messages.append( Message(RecordDict({'arrays':ArrayRecord(model.state_dict()), 'metrics': metrics}), partition_id, 'train') )
    out = strategy.aggregate_train(epoch, messages)
    state_dict =  out[0].to_torch_state_dict()
    df.loc[len(df)] = [out[1]['train_loss'], out[1]['eval_loss'], out[1]['eval_acc']]
        
df[['train_loss', 'val_loss']].plot()
plt.show()
df['val_accuracy'].plot()
plt.show()