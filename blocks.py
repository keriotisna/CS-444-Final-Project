from torch import nn


class CNN(nn.Module):
    """
    A simple CNN for classifying images in the AnimalDataset.
    """

    def __init__(self):

        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),

            nn.Flatten(),
            
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Softmax(dim=0)
        )
        

    def forward(self, x):
        
        # for layer in self.fc:
        #     x = layer(x)
        #     print(type(x).__class__.__name__)
        #     print(x.size())
        
        x = self.fc(x)
        
        return x


# TODO: Create modified residual block that acts as a highway block with several concurrent residual connections in the forward pass
#   This should be generalizable, so I can pass a 4 to the input and there will be 4 residual blocks with residual connections that link all the way through
#   This should also easily connect with existing block classes, so I may need to modify each of those to be able to access the gradients I need
# TODO: This should maybe downweight contributions as residual activations are passed forward.
# TODO: Should residuals be double skipped so the very first residual gets propagated cleanly to the output? Probably not since that would start to contaiminate the 
#   feature representations made through the block.
class HighwayBlock(nn.Module):
    
    
    """
    A network block with longer residual connections that allows long stretches of residual connections.
    """
    
    def __init__(self, in_channels:int, highwaySequence:list=None, highwayLength:int=3, *args, **kwargs) -> None:
        
        """
        Initialize a long highway block with continuous residual connections for long parts of the network
        """
        
        super().__init__(*args, **kwargs)
        
        if highwaySequence:
            self.highwaySequence = highwaySequence
        else:
            highwayElement = nn.Sequential(
                BottleneckBlock4(in_channels=in_channels, encode_factor=4),
            )
            self.highwaySequence = [highwayElement for _ in range(highwayLength)]
        
        
        
    def forward(self, x):
        
        """
        In the forward pass, we have a list of nn.Sequentials self.highwaySequence and we need to do a standard forward pass while ensuring the
        residuals are passed all the way through to create longer continuous residual connections

        Returns:
            _type_: _description_
        """
        
        firstPass = True
        output = None
        
        for layer in self.highwaySequence:
            
            y = None
            
            if firstPass:
                y = layer(x)
                firstPass = False
            else:
                y = layer(output)
            
            residual = layer.residual
            output = y + residual

        return output
    



# TODO: Define bottleneck blocks or other residual connections
class ResidualBlock(nn.Module):
        
    printOutsize = False
        
    def __init__(self, channelCount, activation:nn.Module=nn.ReLU(), kernel_size:int=3, stride:int=1, padding:int=1):
        super().__init__()
        
        self.activation = activation
        
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=channelCount, out_channels=channelCount, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=channelCount),
            self.activation,
        )
        
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=channelCount, out_channels=channelCount, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=channelCount),
            self.activation,
        )

    def forward(self, x):
        
        if self.printOutsize:
            print(f'x.size(): {x.size()}')
            
        y1 = self.c1(x)
        
        if self.printOutsize:
            print(f'y1.size(): {y1.size()}')
            
        y = self.c2(y1)
        
        if self.printOutsize:
            print(f'y.size(): {y.size()}\n')
            
        y = y + x
        self.outsize = y.size()
        self.residual = x
        
        return self.activation(y)



# Normalize residual before adding
class ResidualBlock2(nn.Module):
        
    printOutsize = False
        
    def __init__(self, channelCount, activation:nn.Module=nn.ReLU(), kernel_size:int=3, stride:int=1, padding:int=1):
        super().__init__()
        
        self.activation = activation
        
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=channelCount, out_channels=channelCount, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=channelCount),
            self.activation,
        )
        
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=channelCount, out_channels=channelCount, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=channelCount),
            self.activation,
        )
        
        self.residualNormalization = nn.BatchNorm2d(num_features=channelCount)
        

    def forward(self, x):
        
        if self.printOutsize:
            print(f'x.size(): {x.size()}')
            
        y1 = self.c1(x)
        
        if self.printOutsize:
            print(f'y1.size(): {y1.size()}')
            
        y = self.c2(y1)
        
        if self.printOutsize:
            print(f'y.size(): {y.size()}\n')
            
        normalizedResidual = self.residualNormalization(x)
            
        y = y + normalizedResidual
        self.outsize = y.size()
        self.residual = normalizedResidual
        
        return self.activation(y)




# NOTE: This should theoretically be more expressive and faster, but we are limited by the encoding channel size.
#   If this is too small, then we may be forcing the model to over-compress information leading to worse performance.
class BottleneckBlock(nn.Module):
    
    """
    Bottleneck blocks are slightly more efficient than standard ResidualBlocks in that they condense C channels into 1 via 1x1 convolution
        before performing the more expensive convolution operations. The condensed channels are filtered via convolution, then
        passed through another 1x1 convolution with C channels as the output.
        
        This effectively allows for more transforms with less cost if using 3x3 convolutions
    """
        
    printOutsize = False
        
    def __init__(self, in_channels:int, encode_channels:int=1, kernel_size:int=3, stride:int=1, padding:int=1, activation:nn.Module=nn.ReLU(), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.activation = activation
        self.residual = None

        
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=encode_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=encode_channels),
            self.activation
        )
        
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=encode_channels, out_channels=encode_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(num_features=encode_channels),
            self.activation
        )
        
        self.decode1 = nn.Sequential(
            nn.Conv2d(in_channels=encode_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=in_channels),
            self.activation
        )
    
    
    
    def forward(self, x):
        
        if self.printOutsize:
            print(f'x.size(): {x.size()}')
        
        encoded = self.encode1(x)
        
        if self.printOutsize:
            print(f'encoded.size(): {encoded.size()}')
            
        convolved = self.convolution(encoded)
        
        if self.printOutsize:
            print(f'convolved.size(): {convolved.size()}')
        
        decoded = self.decode1(convolved)
        # TODO: Normalize the decoded values with BatchNorm here?
        y = x + decoded
        self.residual = x
        
        # TODO: BatchNorm+ReLU here?
        
        if self.printOutsize:
            print(f'y.size(): {y.size()}')

        return y



class BottleneckBlock2(BottleneckBlock):
    
    """
    BottleneckBlock2 changes the order of layers from Conv -> BN -> Activ to BN -> Activ -> Conv
    """
        
    def __init__(self, in_channels:int, encode_channels:int=1, kernel_size:int=3, stride:int=1, padding:int=1, activation:nn.Module=nn.ReLU()) -> None:
        super().__init__(in_channels=in_channels, encode_channels=encode_channels, kernel_size=kernel_size, stride=stride, padding=padding, activation=activation)

        self.activation = activation
        self.residual = None
        
        self.encode1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            self.activation,
            nn.Conv2d(in_channels=in_channels, out_channels=encode_channels, kernel_size=1, stride=1, padding=0),
        )
        
        self.convolution = nn.Sequential(
            nn.BatchNorm2d(num_features=encode_channels),
            self.activation,
            nn.Conv2d(in_channels=encode_channels, out_channels=encode_channels, kernel_size=kernel_size, stride=1, padding=1),
        )
        
        self.decode1 = nn.Sequential(
            nn.BatchNorm2d(num_features=encode_channels),
            self.activation,
            nn.Conv2d(in_channels=encode_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
        )
        
    def forward(self, x):
        
        return super().forward(x)



class BottleneckBlock3(BottleneckBlock):
    
    """
    BottleneckBlock3 removes internal normalization and just does the raw operations. Normalization should be done externally when using this block
    """
        
    def __init__(self, in_channels:int, encode_channels:int=1, kernel_size:int=3, stride:int=1, padding:int=1, activation:nn.Module=nn.ReLU()) -> None:
        super().__init__(in_channels=in_channels, encode_channels=encode_channels, kernel_size=kernel_size, stride=stride, padding=padding, activation=activation)

        self.activation = activation
        self.residual = None
        
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=encode_channels, kernel_size=1, stride=1, padding=0),
            self.activation,
        )
        
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=encode_channels, out_channels=encode_channels, kernel_size=kernel_size, stride=1, padding=1),
            self.activation,
        )
        
        self.decode1 = nn.Sequential(
            nn.Conv2d(in_channels=encode_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            self.activation,
        )
        
    def forward(self, x):
        
        return super().forward(x)


class BottleneckBlock4(BottleneckBlock):
    
    """
    BottleneckBlock4 adds back internal batch norms and reduces encoding channels by a factor instead of just to 1.
    """
        
    def __init__(self, in_channels:int, encode_factor:int=4, kernel_size:int=3, stride:int=1, padding:int=1, activation:nn.Module=nn.ReLU()) -> None:
        
        encode_channels = in_channels//encode_factor
        
        super().__init__(in_channels=in_channels, encode_channels=encode_channels, kernel_size=kernel_size, stride=stride, padding=padding, activation=activation)

        self.activation = activation
        self.residual = None
        
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=encode_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=encode_channels),
            self.activation,
        )
        
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=encode_channels, out_channels=encode_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(num_features=encode_channels),
            self.activation,
        )
        
        self.decode1 = nn.Sequential(
            nn.Conv2d(in_channels=encode_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=in_channels),
            self.activation,
        )
        
    def forward(self, x):
        
        return super().forward(x)



class BottleneckBlock5(BottleneckBlock):
    
    """
    BottleneckBlock5 changes the forward function to BatchNorm+ReLU the final output to ensure activations stay consistently normalized through the network
    """
        
    def __init__(self, in_channels:int, encode_factor:int=4, kernel_size:int=3, stride:int=1, padding:int=1, activation:nn.Module=nn.ReLU()) -> None:
        
        encode_channels = in_channels//encode_factor
        
        super().__init__(in_channels=in_channels, encode_channels=encode_channels, kernel_size=kernel_size, stride=stride, padding=padding, activation=activation)

        self.activation = activation
        self.residual = None
        
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=encode_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=encode_channels),
            self.activation,
        )
        
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=encode_channels, out_channels=encode_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(num_features=encode_channels),
            self.activation,
        )
        
        self.decode1 = nn.Sequential(
            nn.Conv2d(in_channels=encode_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=in_channels),
            self.activation,
        )
        
    def forward(self, x):
        
        return super().forward(x)


class ResidualCNN(nn.Module):
    def __init__(self, network:nn.Sequential, printOutsize=False):

        super().__init__()
        
        ResidualBlock.printOutsize = printOutsize
        ResidualBlock2.printOutsize = printOutsize
        BottleneckBlock.printOutsize = printOutsize

        self.network = network
        

    def forward(self, x):

        y = self.network(x)
        
        return y

