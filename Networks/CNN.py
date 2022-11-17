

import torch
import math
import random
from Networks.MLP import MLP


class ConvolutionnalBlock(torch.nn.Module):

    """ A CNN block containing <n_convolutions> convolution and appropriate pooling """

    def __init__(self, prior_n_channels=3, posterior_n_channels=6, n_convolutions=2, poolingKernel=(2, 2)) :

        super().__init__()

        self.prior_n_channels = prior_n_channels
        self.posterior_n_channels = posterior_n_channels
        self.n_convolutions = n_convolutions
        self.poolingKernel = poolingKernel
        self.layers = []

        for i_conv in range(n_convolutions) :
            self.layers.append(torch.nn.Conv2d(
                in_channels=prior_n_channels if (i_conv == 0) else posterior_n_channels,
                out_channels=posterior_n_channels,
                kernel_size=(3, 3),
                padding='same'
            ))
        self.layers.append(torch.nn.BatchNorm2d(posterior_n_channels))
        self.layers.append(torch.nn.MaxPool2d(
            kernel_size=poolingKernel,
            stride=poolingKernel,
            ceil_mode=True
        ))

        # Make all layers visible to optimizer with model.parameters()
        self.layers = torch.nn.ModuleList(self.layers)


    def forward(self, x) :
        for layer in self.layers :
            x = layer(x)
        return x


    def generateRandomBatch(self, batchSize=8) :
        return torch.rand(
            batchSize,
            self.prior_n_channels,
            int(self.poolingKernel[0] * (1 + 10 * random.random())),
            int(self.poolingKernel[1] * (1 + 10 * random.random()))
        )


    def __str__(self, n_tab=0) :
        toPrint = "\t" * n_tab + "== Conv Block ==\n"
        for i, layer in enumerate(self.layers) :
            if (i > 0) : toPrint += "\n"
            toPrint += "\t" * (n_tab + 1) + str(layer)
        return toPrint


class ReshapeCNN(torch.nn.Module):

    """ Module to reshape the output of a CNN feature extractor """

    def __init__(self, width=128, height=128, n_channels=3, n_outputs=256):

        super().__init__()

        self.width = width
        self.height = height
        self.n_channels = n_channels
        self.n_outputs = n_outputs

        self.layers = []
        self.layers.append(torch.nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_outputs,
            kernel_size=(width, height),
            stride=(width, height),
        ))
        self.layers.append(torch.nn.Flatten(start_dim=1, end_dim=-1))

    def forward(self, x) :
        for layer in self.layers :
            x = layer(x)

        return x


    def generateRandomBatch(self, batchSize=8) :
        return torch.rand(
            batchSize,
            self.n_channels,
            self.width,
            self.height
        )


    def __str__(self, n_tab=0) :
        toPrint = "\t" * n_tab + "== Reshape Module ==\n"
        for i, layer in enumerate(self.layers) :
            if (i > 0) : toPrint += "\n"
            toPrint += "\t" * (n_tab + 1) + str(layer)
        return toPrint


class FeatureExtractor(torch.nn.Module) :

    """ CNN feature extractor composed of convolutionnal blocks and resizing """

    def __init__(self, width=128, height=128, n_channels=3, n_convPerBlock=2, n_convBlocks=3, n_convFeatures=128) :

        super().__init__()

        self.width = width
        self.height = height
        self.n_channels = n_channels
        self.n_convPerBlock = n_convPerBlock
        self.n_convBlocks = n_convBlocks
        self.n_convFeatures = n_convFeatures

        self.submodules = []

        # Add CNN blocks to modules
        running_width = self.width
        running_height = self.height
        prior_n_channels = self.n_channels
        for i_block in range(self.n_convBlocks) :
            width_pooling = 2 if (running_width >= 2) else 1
            height_pooling = 2 if (running_height >= 2) else 1
            posterior_n_channels = int(prior_n_channels * math.sqrt(width_pooling * height_pooling))
            self.submodules.append(ConvolutionnalBlock(
                prior_n_channels=prior_n_channels, posterior_n_channels=posterior_n_channels, n_convolutions=n_convPerBlock, poolingKernel=(width_pooling, height_pooling)
            ))
            running_width = math.ceil(running_width / width_pooling)
            running_height = math.ceil(running_height / height_pooling)
            prior_n_channels = posterior_n_channels

        # Add reshaping module to get the required <n_convFeatures>
        self.submodules.append(ReshapeCNN(
            width=running_width,
            height=running_height,
            n_channels=posterior_n_channels,
            n_outputs=n_convFeatures
        ))

    def forward(self, x) :
        for submodule in self.submodules :
            x = submodule(x)
        return x

    def __str__(self, n_tab=0) :
        toPrint = "== Feature Extractor ==\n"
        for i, submodule in enumerate(self.submodules) :
            if (i > 0) : toPrint += "\n"
            toPrint += submodule.__str__(n_tab=n_tab + 1)
        return toPrint


    @classmethod
    def sample(cls, trial, width=128, height=128, n_channels=3, n_convFeatures=128) :
        return cls(
            width=width,
            height=height,
            n_channels=n_channels,
            n_convPerBlock=trial.suggest_int("CNN_n_convPerBlock", 1, 5),
            n_convBlocks=trial.suggest_int("CNN_n_convBlocks", 1, 8),
            n_convFeatures=n_convFeatures
        )


class CNN(torch.nn.Module):

    """ Basic configurable CNN """

    def __init__(self,
        width : int = 128,
        height : int = 128,
        n_channels : int = 3,
        n_convPerBlock : int = 2,
        n_convBlocks : int = 3,
        n_convFeatures : int = 128,
        isFullyConvolutionnal : bool = False,
        isResidual : bool = False,
        n_outputs : int = 64,
        n_hidden : int = 3,
        shape : float = 1.0,
        dropoutRate : float = 0.75
    ):

        """
        Feature extractor parameters :
            <width> : Width of input map.
            <height> : Height of input map.
            <n_channels> : Number of channels of input map.
            <n_convPerBlock> : Number of convolutions in a block (before a maxPool)
            <n_convBlocks> : Number of convolutionnal blocks
            <n_convFeatures> : Number of features extracted by the CNN, before the MLP.
            <isFullyConvolutionnal> : Weather CNN contains FCs or not.
            <isResidual> : Weather CNN is residual or not.
        Mlp parameters :
            <n_outputs> : Number of outputs of the CNN.
            <n_hidden> : Number of hidden self.layers, output layer not included.
            <shape> : With of the middle layer, as a proportion of (<n_inputs> + <n_outputs>) : 0.5 looks like an autoencoder, 2 expands then retracts.

        Example of networks :

          n_convPerBlock = 4
          n_convBlocks = 3
               ||||
               ||||
     width     |||| ||||
       x       |||| |||| ||||  n_convFeatures   shape     n_hidden     n_outputs
     height    |||| |||| |||| ################ ####### #### ######## #############
       x       |||| ||||
   n_channels  ||||
               ||||

               |===== Feature Extractor =====| |=============== MLP ==============|
        """

        super().__init__()
        self.width = width
        self.height = height
        self.n_channels = n_channels
        self.n_convPerBlock = n_convPerBlock
        self.n_convBlocks = n_convBlocks
        self.n_convFeatures = n_convFeatures
        self.isFullyConvolutionnal = isFullyConvolutionnal
        self.isResidual = isResidual
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.shape = shape

        self.featureExtractor = FeatureExtractor(
            width=width,
            height=height,
            n_channels=n_channels,
            n_convPerBlock=n_convPerBlock,
            n_convBlocks=n_convBlocks,
            n_convFeatures=n_convFeatures
        )
        self.mlp = MLP(n_convFeatures, n_outputs, n_hidden, shape)


    def forward(self, inputBatch):
        features = self.featureExtractor(inputBatch)
        predictions = self.mlp(features)
        return predictions


    def generateRandomBatch(self, batchSize=8) :
        return torch.rand(batchSize, self.n_channels, self.width, self.height)


    def __str__(self, n_tab=0) :
        toPrint = "\t" * n_tab + "== CNN ==\n"
        toPrint += "\t" * (n_tab + 1) + self.featureExtractor.__str__(n_tab=n_tab + 1)
        toPrint += self.mlp.__str__()
        return toPrint

    @classmethod
    def sample(cls, trial, width=128, height=128, n_channels=3, n_outputs=64) :
        return cls(
            width=width,
            height=height,
            n_channels=n_channels,
            n_convPerBlock=trial.suggest_int("CNN_n_convPerBlock", 1, 5),
            n_convBlocks=trial.suggest_int("CNN_n_convBlocks", 1, 8),
            n_convFeatures=trial.suggest_int("CNN_n_convFeatures", 16, 1024, log=True),
            # isFullyConvolutionnal=trial.suggest_bool("CNN_isFullyConvolutionnal"),
            # isResidual=trial.suggest_bool("CNN_isResidual"),
            n_outputs=n_outputs,
            n_hidden=trial.suggest_int("CNN_n_hidden", 1, 16, log=True),
            shape=trial.suggest_float("CNN_shape", 0.1, 10, log=True),
            dropoutRate=trial.suggest_float("CNN_dropoutRate", 0.1, 1, log=True)
        )
