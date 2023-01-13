
import random
import numpy as np
from Data.Trajectory.Trajectory import Trajectory
import csv
from numpy import genfromtxt


class Trajectory_CSV(Trajectory):

    """ Load a trajectory from a .csv """

    def __init__(
        self, 
        csv_path, 
        state_indices=[2, 3], # Which elements of a line correspond to the state
        action_indices=[0, 1], # Which elements of a line correspond to the action
        line_begin=1, # Which line to start. Default is 1 (not 0) to remove the csv header
        line_end=-1 # Which line to stop. Useful with <line_begin> to split a csv into train/valid/test.
    ):

        self.state_indices = state_indices
        self.action_indices = action_indices

        self.data = genfromtxt(csv_path, delimiter=',')
        self.data = self.data[line_begin:line_end, :]
        self.length = self.data.shape[0]


    def getLength(self):
        return self.length


    def getState(self, index):
        return self.data[index, self.state_indices]


    def getAction(self, index):
        return self.data[index, self.action_indices]

