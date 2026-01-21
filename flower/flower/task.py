
# import jax
# import jax.numpy as jnp

# key = jax.random.PRNGKey(0)


from functools import lru_cache
import pennylane as qml

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

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset
#fds = None  # Cache FederatedDataset


def get_data(dataset, partition_id, num_partitions, num_features = 256):
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(dataset=dataset, partitioners={"train": partitioner})
    dset = fds.load_partition(partition_id).with_format('torch')
    
    image_transform = Pipeline( [('scaler', StandardScaler()), ('pca', PCA(num_features) ) ] )
    image_transform.fit(dset['image'].reshape(len(dset), -1))
    def preprocess_data(sample):
        x = sample['image'].reshape(1, -1)
        y = sample['label']
        x = image_transform.transform(x)
        sample['x'] = x.reshape(-1)
        sample['y'] = y
        return sample
    dset = dset.map(preprocess_data, remove_columns=['image', 'label'])
#    return dset
    return dset


from importlib import import_module
def load_model(model_name):
    model_lib = import_module('flower.models.'+ model_name)
#    return model_lib.run, model_lib.loss_fun, model_lib.init_params
    return model_lib.Model

import torch
def train(model, data, lr=0.01, optim = None):
    if optim is None:
        optim = torch.optim.Adam(model.parameters(), lr)
        
    ave_loss = 0
    loss_history = []
    try:
        for i, sample in enumerate(data):
            x = sample['x']
            y_true = sample['y']
            optim.zero_grad()
            y_pred = model(x)
            loss = model.loss_fun(y_true, y_pred, 10)
            loss.backward()
            optim.step()
            ave_loss  += loss.item()
            loss_history.append(loss.item())
            if i % 2 == 0:
                print('.', end = '')
    finally:
        return loss_history, optim
