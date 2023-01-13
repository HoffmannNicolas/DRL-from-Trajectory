
# To run "DRL-from-Trajectory$ python -m examples.read_trajectory_from_csv"

from Data.Trajectory.Trajectory_CSV import Trajectory_CSV

train_traj = Trajectory_CSV(
	"/home/nicolas/temp/ros_recording_sim3_10hz.csv", 
	state_indices=[0, 1, 2, 3, 4, 5], 
	action_indices=[6, 7, 8],
	line_begin=1,
	line_end=10,
)

valid_traj = Trajectory_CSV(
	"/home/nicolas/temp/ros_recording_sim3_10hz.csv", 
	state_indices=[0, 1, 2, 3, 4, 5], 
	action_indices=[6, 7, 8],
	line_begin=11,
	line_end=20,
)

test_traj = Trajectory_CSV(
	"/home/nicolas/temp/ros_recording_sim3_10hz.csv", 
	state_indices=[0, 1, 2, 3, 4, 5], 
	action_indices=[6, 7, 8],
	line_begin=21,
	line_end=30,
)

test_traj.print()
