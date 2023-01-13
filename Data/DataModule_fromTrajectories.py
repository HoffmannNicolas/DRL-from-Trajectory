
from torch.utils.data import random_split, DataLoader, Dataset

import pytorch_lightning as pl
from typing import Optional
import numpy as np


class TransitionDataset(Dataset):

    """ PyTorch Dataset for trajectory transitions """

    def __init__(self, transitions):
        self.transitions = transitions


    def __len__(self):
        return len(self.transitions)



    def __getitem__(self, idx):
        transition = self.transitions[idx]
        state = transition[0]
        action = transition[1]
        afterState = transition[2]
        _input = np.concatenate([state, action])
        return _input, afterState



class DataModule_fromTrajectories(pl.LightningDataModule):

    """ PyTorch Lightning Data Module that inputs its data from Trajectories """

    def __init__(self, trainTrajectories, validTrajectories, testTrajectories, batch_size=1, shuffle=True):
        super().__init__()
        self.trainTrajectories = trainTrajectories
        self.validTrajectories = validTrajectories
        self.testTrajectories = testTrajectories

        self.batch_size = batch_size
        self.shuffle = shuffle


    def setup(self, stage: Optional[str] = None):
        """ Get all transitions from trajectories and format tham as Datasets """
        trainTransitions = self.trainTrajectories.getAllTransitions()
        validTransitions = self.validTrajectories.getAllTransitions()
        testTransitions = self.testTrajectories.getAllTransitions()

        self.trainDataset = TransitionDataset(trainTransitions)
        self.validDataset = TransitionDataset(validTransitions)
        self.testDataset = TransitionDataset(testTransitions)



    def train_dataloader(self):
        return DataLoader(
            self.trainDataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )



    def val_dataloader(self):
        return DataLoader(
            self.validDataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )



    def test_dataloader(self):
        return DataLoader(
            self.testDataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )
