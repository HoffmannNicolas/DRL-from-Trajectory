

import torch

class MLP(torch.nn.Module):

    """ Basic configurable MLP """

    def __init__(
        self,
        n_inputs : int = 32, 
        n_outputs : int = 64, 
        n_hidden : int = 3, 
        shape : float = 1.0, 
        dropoutRate : float = 0.75,
        activations : str = "relu", # "relu" or "sigmoid"
        last_activation = torch.nn.ReLU() 
    ):

        """
        <n_inputs> : Number of input features.
        <n_outputs> : Number of output features.
        <n_hidden> : Number of hidden self.layers, output layer not included.
        <shape> : Width of the middle layer, as a proportion of (<n_inputs> + <n_outputs>) : 0.5 looks like an autoencoder, 2 expands then retracts.

        Example of networks :
                            n_inputs = 12            n_inputs = 12           n_inputs = 4
                            n_outputs = 20           n_outputs = 8           n_outputs = 12
                            n_hidden = 3             n_hidden = 3            n_hidden = 3
                            Shape = 1/8              Shape = 2               Shape = 1
        Input Layer    :     ############            ############                ####
        Hidden Layer 1 :       ########            ################             ######
        Hidden Layer 2 :         ####            ####################          ########
        Hidden Layer 3 :     ############           ##############            ##########
        Output Layer   : ####################          ########              ############
        """

        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.shape = shape

        self.layers = []
        previousLayerWidth = n_inputs

        n_middle = shape * (n_inputs + n_outputs) / 2
        for i in range(n_hidden) :
            networkProportion = (1 + i) / (n_hidden + 1) # 0 for the input layer, 1 for output layer, and in [0, 1] for other hidden self.layers
            middleDistance = abs(0.5 - networkProportion) # Absolute distance to the middle of the network
            if (networkProportion < 0.5) : # First half of the network : Linear interpolation between <n_inputs> and <n_middle>
                layerWidth = (n_inputs * middleDistance + n_middle * networkProportion) * 2
            elif (networkProportion == 0.5) :
                layerWidth = n_middle
            else : # Second half of the network : Linear interpolation between <n_middle> and <n_outputs>
                layerWidth = (n_outputs * middleDistance + n_middle * abs(1 - networkProportion)) * 2
            layerWidth = int(layerWidth)
            layerWidth = max(layerWidth, 1)
            self.layers.append(torch.nn.Linear(previousLayerWidth, layerWidth))
            self.layers.append(torch.nn.Dropout(p=dropoutRate))
            if (activations == "relu"):
                self.layers.append(torch.nn.ReLU())
            if (activations == "sigmoid"):
                self.layers.append(torch.nn.Sigmoid())
            previousLayerWidth = layerWidth
        self.layers.append(torch.nn.Linear(previousLayerWidth, n_outputs))
        self.layers.append(torch.nn.Dropout(p=dropoutRate))
        if (last_activation is not None):
            self.layers.append(last_activation)

        # Make all layers visible to optimizer with model.parameters()
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, inputBatch):
        result = inputBatch
        for layer in self.layers:
            result = layer(result)
        return result


    def generateFakeBatch(self, batchSize=8) :
        return torch.rand(batchSize, self.n_inputs)

    def __str__(self, maxNetworkSize=50) :

        def _displayLayer(_layerWidth, _maxWidth) :
            _toPrint = ""
            beforeSpace = round((_maxWidth - _layerWidth) / 2)
            beforeSpace = round(maxNetworkSize * (_maxWidth - _layerWidth) / (2 * _maxWidth))
            _toPrint += beforeSpace * " "
            rescaledLayerSize = round(_layerWidth * maxNetworkSize / _maxWidth)
            _toPrint += "#" * rescaledLayerSize
            afterSpace = maxNetworkSize - beforeSpace - rescaledLayerSize
            _toPrint += " " * afterSpace
            _toPrint += " | "
            return _toPrint


        toPrint = f"\n[MultiLayerPerceptron] :: n_inputs={self.n_inputs} ; n_outputs={self.n_outputs} ; n_hidden={self.n_hidden} ; shape={self.shape}\n"
        layersSizes = []
        for layer in self.layers :
            if not(isinstance(layer, torch.nn.modules.linear.Linear)) : continue
            layersSizes.append(layer.in_features)
        layersSizes.append(self.n_outputs)
        maxSize = max(layersSizes)
        fcFound = 0
        for layer in self.layers :
            if isinstance(layer, torch.nn.modules.linear.Linear) :
                layerSize = layersSizes[fcFound]
                fcFound += 1
                toPrint += _displayLayer(layerSize, maxSize)
            else :
                toPrint += " " * maxNetworkSize + " | "
            toPrint += str(layer) + "\n"
        toPrint += _displayLayer(self.n_outputs, maxSize)
        toPrint += "Output"
        return toPrint


    @classmethod
    def sample(cls, trial, n_inputs=32, n_outputs=32) :
        return cls(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_hidden=trial.suggest_int("MLP_n_hidden", 1, 16, log=True),
            shape=trial.suggest_float("MLP_shape", 0.1, 10, log=True),
            dropoutRate=trial.suggest_float("MLP_dropoutRate", 0.1, 1, log=True)
        )
