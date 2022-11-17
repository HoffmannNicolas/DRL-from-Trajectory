
import random
import numpy as np
from Data.Trajectory.Trajectory import Trajectory

class Trajectory_Random(Trajectory) :

    """ Random Trajectory for easy testing """

    def __init__(self, length=None) :
        if (length is None) :
            length = random.randint(16, 128)
        self.length = length



    def getLength(self):
        return self.length



    def getState(self, index):
        return np.array(random.random())



    def getAction(self, index):
        return np.array(random.random())



if __name__ == "__main__" :
    print("test Trajectory Random")
    trajectory = Trajectory_Random()
    trajectory.print()

    print(trajectory(0))
