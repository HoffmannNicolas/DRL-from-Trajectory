
# To run "DRL-from-Trajectory$ python -m examples.instanciateDataModule_fromTrajectories"

from Data.Trajectory.Trajectory_Random import Trajectory_Random
from Data.Trajectory.Trajectories import Trajectories
from Data.DataModule_fromTrajectories import DataModule_fromTrajectories

trainTrajectories = Trajectories([
    Trajectory_Random()
])

validTrajectories = Trajectories([
    Trajectory_Random()
])

testTrajectories = Trajectories([
    Trajectory_Random()
])

dataModule = DataModule_fromTrajectories(trainTrajectories, validTrajectories, testTrajectories)
dataModule.setup()
print(dataModule)


print("Train DataLoader")
for i, batch in enumerate(dataModule.train_dataloader()) :
    print(f"batch[{i}]={batch}")

print("\n\nValidation DataLoader")
for i, batch in enumerate(dataModule.val_dataloader()) :
    print(f"batch[{i}]={batch}")

print("\n\nTest DataLoader")
for i, batch in enumerate(dataModule.test_dataloader()) :
    print(f"batch[{i}]={batch}")
