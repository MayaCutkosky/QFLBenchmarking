

import numpy as np
from flwr.app import ArrayRecord, Context, ConfigRecord
from flwr.serverapp import Grid, ServerApp


from flwr.common import log
from logging import INFO

from flwr.serverapp.strategy import FedAvg
from flwr.common import Message
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords
class CustomFedAvg(FedAvg):

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ):
        """Configure the next round of federated training and maybe do LR decay."""
        # Decrease learning rate by a factor of 0.5 every 5 rounds
        if server_round % 5 == 0 and server_round > 0:
            config["lr"] *= 0.9
#            print("LR decreased to:", config["lr"])
        # Pass the updated config and the rest of arguments to the parent class
        return super().configure_train(server_round, arrays, config, grid)

from flower.task import load_model
# from jax.random import PRNGKey



# Create ServerApp
app = ServerApp()
config = {'num-server-rounds':3,'backend' : 'sim', 'num_qubits' : 8, 'model' : "QFL_QD", 'dataset' : 'mnist', 'num_classes' : 10, 'lr' : 0.1}

import numpy as np
@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read from config
    num_rounds = config["num-server-rounds"]
    lr: float = config["lr"]

    # Load global model
    # rng = PRNGKey(0)
    # model, loss_fun, init_params = load_model(config['model'])

    # params = init_params(rng, config['num_classes'])
    # arrays = ArrayRecord([np.array(params, copy = False)])
    
    params= load_model(config['model']).init_params(config['num_classes'])
    arrays = ArrayRecord(params)

    # Initialize FedAvg strategy
    strategy = CustomFedAvg()
    
    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config = ConfigRecord({'lr' :lr}),
        num_rounds=num_rounds,
        timeout = 10
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    ndarrays = result.arrays.to_numpy_ndarrays()
    np.savez("final_model.npz", *ndarrays)
