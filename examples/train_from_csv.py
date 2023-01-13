
# To run "DRL-from-Trajectory$ python -m examples.instanciatedata_module_fromTrajectories"

from Data.Trajectory.Trajectory_CSV import Trajectory_CSV
from Data.Trajectory.Trajectories import Trajectories
from Data.DataModule_fromTrajectories import DataModule_fromTrajectories

import pytorch_lightning as pl
from Networks.MLP import MLP
import torch


trainTrajectories = Trajectories([
    Trajectory_CSV(
        "/home/nicolas/temp/ros_recording_sim3_10hz.csv", 
        state_indices=[0, 1, 2, 3, 4, 5], 
        action_indices=[6, 7, 8],
        line_begin=1,
        line_end=9_000,
    )
])

validTrajectories = Trajectories([
    Trajectory_CSV(
        "/home/nicolas/temp/ros_recording_sim3_10hz.csv", 
        state_indices=[0, 1, 2, 3, 4, 5], 
        action_indices=[6, 7, 8],
        line_begin=9_001,
        line_end=9_900,
    )
])

testTrajectories = Trajectories([
    Trajectory_CSV(
        "/home/nicolas/temp/ros_recording_sim3_10hz.csv", 
        state_indices=[0, 1, 2, 3, 4, 5], 
        action_indices=[6, 7, 8],
        line_begin=9_901,
        line_end=10_000,
    )
])

data_module = DataModule_fromTrajectories(trainTrajectories, validTrajectories, testTrajectories)
data_module.setup()
print(data_module)




class MLP_Module(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.network = MLP(
            n_inputs=9,
            n_outputs=6,
            n_hidden=3,
            shape=5,
            activations="sigmoid",
            last_activation=None
        )
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = torch.nn.MSELoss()


    def forward(self, x):
        return self.network(x)


    def training_step(self, batch, batch_idx):
        x, y = batch # x := concatenation(state, action) ; y := next_state
        x = x.type(torch.float32) 
        y = y.type(torch.float32) 
        y_hat = self.network(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch # x := concatenation(state, action) ; y := next_state
        x = x.type(torch.float32) 
        y = y.type(torch.float32) 
        y_hat = self.network(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)


    def test_step(self, batch, batch_idx):
        x, y = batch # x := concatenation(state, action) ; y := next_state
        x = x.type(torch.float32)
        y = y.type(torch.float32)
        y_hat = self.network(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


    def __str__(self):
        return str(self.network)


mlp_module = MLP_Module()
mlp_module.batch_size = 256 # Field required for auto tuning later
mlp_module.learning_rate = 1e-1 # Field required for auto tuning later
print(mlp_module)
trainer = pl.Trainer(
    max_epochs=100_000, 
    auto_scale_batch_size=True, 
    auto_lr_find=True,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="val_loss", mode="min"),
        pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
    ],
    accumulate_grad_batches={0: 1, 100: 1},
)


trainer.tune(
    mlp_module,
    datamodule=data_module
)
data_module.batch_size = mlp_module.batch_size # Update dataloaders batch size to match the one just found


trainer.fit(
    model=mlp_module, 
    datamodule=data_module
)
