
# To run "DRL-from-Trajectory$ python -m examples.instanciateDataModule_fromTrajectories"

from Data.Trajectory.Trajectory_CSV import Trajectory_CSV
from Data.Trajectory.Trajectories import Trajectories
from Data.DataModule_fromTrajectories import DataModule_fromTrajectories

trainTrajectories = Trajectories([
    Trajectory_CSV(
        "/home/nicolas/temp/ros_recording_sim3_10hz.csv", 
        state_indices=[0, 1, 2, 3, 4, 5], 
        action_indices=[6, 7, 8],
        line_begin=1,
        line_end=10,
    )
])

validTrajectories = Trajectories([
    Trajectory_CSV(
        "/home/nicolas/temp/ros_recording_sim3_10hz.csv", 
        state_indices=[0, 1, 2, 3, 4, 5], 
        action_indices=[6, 7, 8],
        line_begin=11,
        line_end=20,
    )
])

testTrajectories = Trajectories([
    Trajectory_CSV(
        "/home/nicolas/temp/ros_recording_sim3_10hz.csv", 
        state_indices=[0, 1, 2, 3, 4, 5], 
        action_indices=[6, 7, 8],
        line_begin=21,
        line_end=30,
    )
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
