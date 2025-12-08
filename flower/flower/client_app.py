from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import log
from logging import INFO
from flower.task import get_device, load_model, get_data
import numpy as np
# Flower ClientApp
app = ClientApp()

from torch.optim import Adam
import torch

config = {'num=server-rounds':3,'backend' : 'sim', 'num_qubits' : 8, 'model' : "QFL_QD", 'dataset' : 'mnist', 'num_classes' : 10, 'lr' : 0.1}
@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    # params = jnp.array(msg.content["arrays"])
    
    #load device
    dev = get_device(
        config['backend'], 
        config['num_qubits'], 
        context.node_id
    )
    
    
    # model, loss_fun, init_params = load_model(config['model'])
    # def forward_fun(params, X, y_true):
    #     y_pred = model(dev, X, params)
    #     return loss_fun(y_true, y_pred, num_classes = config['num_classes'])
        
    # train_fun = jax.jit(jax.value_and_grad( forward_fun))    
    
    # #train
    # optim = optax.adam(msg.context['config']['lr'])
    # opt_state = optim.init(params)
    # for x,y in get_data(config['dataset'], partition_id, num_partitions):
    #     loss, grads = train_fun(params, x, y)
    #     updates, opt_state = optim.update(grads, opt_state)
    #     params = optax.apply_updates(params, updates)
    
    
    model = load_model(config['model'])(dev, config['num_classes'])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    optim = Adam(model.parameters(), msg.content['config']['lr'])
    dset = get_data(config['dataset'], partition_id, num_partitions)
    ave_loss = 0
    for i, sample in enumerate(dset):
        x = sample['x']
        y_true = sample['y']
        optim.zero_grad()
        y_pred = model(x)
        loss = model.loss_fun(y_true, y_pred, config['num_classes'])
        loss.backward()
        optim.step()
        ave_loss  += loss.item()
    len_dset = i
    ave_loss = ave_loss / len_dset
    # Construct and return reply Message
    metrics = {
        "train_loss": ave_loss,
        'num-examples' : len_dset,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    # params = jnp.array(msg.content["arrays"])
    
    #load device
    dev = get_device(
        config['backend'], 
        config['num_qubits'], 
        context.node_id
    )
    
    
    # model, loss_fun, init_params = load_model(config['model'])
    
    # @jax.jit
    # def calc_accuracy(x,y, params):
    #     return jnp.mean(jnp.argmax( model(dev, x, params) , -1) == y).item()
    
    # for x,y in get_data(config['dataset'], partition_id, num_partitions):
    #     acc = calc_accuracy(x,y,params)
        
    model = load_model(config['model'])(dev, config['num_classes'])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    dset = get_data(config['dataset'], partition_id, num_partitions)
    acc = 0
    for i, sample in enumerate(dset):
        x = sample['x']
        y = sample['y']
        acc += (torch.argmax(model(x), -1)== y).float().mean().item()
    len_dset = i
    acc = acc / len_dset
    # Construct and return reply Message
    metrics = {
        "accuracy": acc,
        'num-examples' : len_dset,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
