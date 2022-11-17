
import random

class Trajectories():

    """ Deals with a set of trajectories """

    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.trajectoryLengths = [trajectory.getLength() for trajectory in self.trajectories]



    def iterate(self, randomized=False, includeNextAction=False) :
        """ Iterate through all trajectories """

        if (randomized) :
            pass # TODO : Use length of all trajectory and their total to sample random from them, in proportion of their amount of transitions.

        for trajectory in self.trajectories:
            for transition in trajectory.iterate(includeNextAction=includeNextAction):
                yield transition



    def getAllTransitions(self, includeNextAction=False) :
        """ Useful for eager-loading a dataset """
        return list(self.iterate(includeNextAction=includeNextAction))



    def sampleTrajectories(self, k, uniform=False) :
        """ Return a k-sized list of trajectory samples with replacement.
        - <k> : The number of samples.
        - <weighted> Specifies the distribution :
            - If <True> the trajectory is sampled weighted by its length.
            - If <False> all trajectory have the same probability to be sampled.
        """
        if (uniform) :
            return random.choices(self.trajectories, k=k)
        else :
            return random.choices(self.trajectories, weights=self.trajectoryLengths, k=k)

    def sampleTrajectory(self, uniform=False) :
        """ Sugar coating """
        return self.sampleTrajectories(1, uniform=uniform)



    def sampleTransitions(self, k, includeNextAction=False, uniform=False) :
        """ Return a k-sized list of transition samples with replacement :
        - <k> : The number of samples.
        - <includeNextAction> specifies if the transition should include the next action.
        - <uniform> specifies the sampling distribution :
            - If <True> the transition is sampled uniformely amongst all transitions in all trajectories, implying the trajectory is sampled according to its length.
            - If <False> a trajectory is uniformely sampled, first, then a transition is uniformely sampled from that trajectory.

        This method is useful for lazy-loading a dataset.
        """
        trajectories = self.sampleTrajectories(k, uniform=not(uniform)) # /!\ Warning /!\ A uniform sampling of a transition requires non-uniform sampling of the trajectory.
        return [trajectory.sampleTransition(includeNextAction=includeNextAction) for trajectory in trajectories]

    def sampleTransition(self, includeNextAction=False, uniform=False) :
        """ Sugar coating """
        return self.sampleTransitions(1, includeNextAction=includeNextAction, uniform=uniform)



    def print(self) :
        for index, (state, action, nextState) in enumerate(self.iterate(randomized=False)) :
            print(f"Trajectories({index}) : State={state}\t\tAction={action}\t\tNextState={nextState}")



if __name__ == "__main__" :
    # Test Trajectories class

    from Trajectory_Random import Trajectory_Random
    trajectory1 = Trajectory_Random()
    trajectory2 = Trajectory_Random()
    trajectory3 = Trajectory_Random()

    trajectories = Trajectories([
        trajectory1,
        trajectory2,
        trajectory3
    ])

    trajectories.print()


    batchSize = 16
    batch = trajectories.sampleTransitions(batchSize)

    print(batch)

    for i, elt in enumerate(batch) :
        print(f"Batch[{i}] = {batch[i]}")
