from torch import nn
import torch

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
# TODO: Allow easy setting of default highway elements
class HighwayBlock(nn.Module):
    
    
    printOutsize = False
    
    """
    A network block with longer residual connections that allows long stretches of residual connections.
    """
    
    def __init__(self, in_channels:int, highwaySequence:nn.Sequential=None, highwayLength:int=3, *args, **kwargs) -> None:
        
        """
        Initialize a long highway block with continuous residual connections for long parts of the network
        """
        
        super().__init__(*args, **kwargs)
                
        if highwaySequence:
            self.highwaySequence = highwaySequence
        else:
            highwayElement = BottleneckBlock4(in_channels=in_channels, encode_factor=4)
            self.highwaySequence = nn.Sequential(*[highwayElement for _ in range(highwayLength)])
        
        
        
    def forward(self, x):
        
        """
        Performs a forward pass on all highwaySequence elements. Also creates a long skip connection which propagates
            the input all the way to the output.
        """
        
        firstPass = True
        output = torch.empty_like(x)
        
        for layer in self.highwaySequence:
                        
            y = torch.empty_like(x)
            
            if firstPass:
                y = layer(x)
                firstPass = False
            else:
                y = layer(output)
            
            # TODO: Add original activations to each layer?
            output = y

        # Add the original residual creating a long skip from the start of the highway to the end
        return output + x
    



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
        
        self.residualNormalization = nn.Sequential(
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
            
        normalizedResidual = self.residualNormalization(x)
            
        y = y + normalizedResidual
        self.outsize = y.size()
        
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
    BottleneckBlock5 has residual normalization while previous versions do not
    """
        
    def __init__(self, in_channels:int, encode_factor:int=4, kernel_size:int=3, stride:int=1, padding:int=1, activation:nn.Module=nn.ReLU()) -> None:
        
        encode_channels = in_channels//encode_factor
        
        super().__init__(in_channels=in_channels, encode_channels=encode_channels, kernel_size=kernel_size, stride=stride, padding=padding, activation=activation)

        self.activation = activation
        
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
        
        self.residualNormalization = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            self.activation,
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
        
        y = decoded + self.residualNormalization(x)
                
        if self.printOutsize:
            print(f'y.size(): {y.size()}')

        return y



class DoubleEncodeBottleneckBlock(nn.Module):
    
    
    """
    The DoubleEncodeBottleneckBlock condenses the bottleneck channels a second time before using the convolution to try and save more computation time.
    This may have significant model performance impacts, but should in theory be even more efficient
    """
        
    def __init__(self, in_channels:int, encode_factor1:int=4, encode_factor2:int=4, kernel_size:int=3, stride:int=1, padding:int=1, activation:nn.Module=nn.ReLU()) -> None:
        
        super().__init__()
        
        encode_channels1 = in_channels//encode_factor1
        encode_channels2 = encode_channels1//encode_factor2

        self.activation = activation
        
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=encode_channels1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=encode_channels1),
            self.activation,
        )
        
        self.encode2 = nn.Sequential(
            nn.Conv2d(in_channels=encode_channels1, out_channels=encode_channels2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=encode_channels2),
            self.activation,
        )
        
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=encode_channels2, out_channels=encode_channels2, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(num_features=encode_channels2),
            self.activation,
        )
        
        self.decode2 = nn.Sequential(
            nn.Conv2d(in_channels=encode_channels2, out_channels=encode_channels1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=encode_channels1),
            self.activation,
        )
        
        self.decode1 = nn.Sequential(
            nn.Conv2d(in_channels=encode_channels1, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=in_channels),
            self.activation,
        )
        
        self.residualNormalization = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            self.activation,
        )
        
        
    def forward(self, x):
        
        encoded1 = self.encode1(x)
        encoded2 = self.encode2(encoded1)
            
        convolved = self.convolution(encoded2)
        
        decoded2 = self.decode2(convolved)
        decoded1 = self.decode1(decoded2)

        y = decoded1 + self.residualNormalization(x)
                
        return y


class DoubleEncodeBottleneckBlock2(nn.Module):
    
    
    """
    The DoubleEncodeBottleneckBlock condenses the bottleneck channels a second time before using the convolution to try and save more computation time.
    This may have significant model performance impacts, but should in theory be even more efficient
    """
        
    def __init__(self, in_channels:int, encode_factor1:int=4, encode_factor2:int=4, kernel_size:int=3, stride:int=1, padding:int=1, activation:nn.Module=nn.ReLU()) -> None:
        
        super().__init__()
        
        encode_channels1 = in_channels//encode_factor1
        encode_channels2 = encode_channels1//encode_factor2

        self.activation = activation
        
        self.inputNorm = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            self.activation
        )
        
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=encode_channels1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=encode_channels1),
            self.activation,
        )
        
        self.encode2 = nn.Sequential(
            nn.Conv2d(in_channels=encode_channels1, out_channels=encode_channels2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=encode_channels2),
            self.activation,
        )
        
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=encode_channels2, out_channels=encode_channels2, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(num_features=encode_channels2),
            self.activation,
        )
        
        self.decode2 = nn.Sequential(
            nn.Conv2d(in_channels=encode_channels2, out_channels=encode_channels1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=encode_channels1),
            self.activation,
        )
        
        self.decode1 = nn.Sequential(
            nn.Conv2d(in_channels=encode_channels1, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=in_channels),
            self.activation,
        )
        
    def forward(self, x):
        
        normInput = self.inputNorm(x)
        
        encoded1 = self.encode1(normInput)
        encoded2 = self.encode2(encoded1)
            
        convolved = self.convolution(encoded2)
        
        decoded2 = self.decode2(convolved)
        decoded1 = self.decode1(decoded2)

        y = decoded1 + x
                
        return y



class BranchBlock(nn.Module):
    
    """
    The BranchBlock splits one feature map into several parallel classifiers before concatenating their outputs together again.
    This also creates a residual connection between the inputs and the concatenated branches. 
    """
        
    def __init__(self, in_channels:int, branches:list, activation=nn.ReLU(), averageChannels=False) -> None:
        
        super().__init__()
        
        self.activation = activation
        
        self.inputNorm = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            self.activation
        )
        
        self.out_channels = in_channels * len(branches)
        # We need to use a ModuleList to ensure that the .to(device) operation registers these as submodules
        self.branches = nn.ModuleList(branches)
        
        self.averageChannels = averageChannels
    
    
    def forward(self, x:torch.Tensor):
        
        normInput = self.inputNorm(x)
                
        outputs = [branch(normInput) + x for branch in self.branches]
        if self.averageChannels:
            y = torch.mean(torch.stack(outputs), dim=0)
        else:
            y = torch.cat(outputs, dim=1)
                
        return y

class BranchBlockNorm(nn.Module):
    
    """
    The BranchBlockNorm splits one feature map into several parallel classifiers before concatenating their outputs together again.
    This also creates a residual connection between the inputs and the concatenated branches. 
    The Norm version also normalizes activations via a batch normalization following all the branches
    """
        
    def __init__(self, in_channels:int, branches:list, activation=nn.ReLU(), averageChannels=False) -> None:
        
        super().__init__()
        
        self.activation = activation
        
        self.inputNorm = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            self.activation
        )
        
        self.out_channels = in_channels * len(branches)
        # We need to use a ModuleList to ensure that the .to(device) operation registers these as submodules
        self.branches = nn.ModuleList(branches)
        
        self.averageChannels = averageChannels
    
    
    def forward(self, x:torch.Tensor):
        
        normInput = self.inputNorm(x)
                
        rawOutputs = [branch(normInput) for branch in self.branches]
        normOutputs = [self.inputNorm(raw) + x for raw in rawOutputs]
        if self.averageChannels:
            y = torch.mean(torch.stack(normOutputs), dim=0)
        else:
            y = torch.cat(normOutputs, dim=1)
                
        return y


# This is the main class used to run a network. It's pretty simple, and the only real difference is in setting values for debugging
class ResidualCNN(nn.Module):
    def __init__(self, network:nn.Sequential, printOutsize=False):

        super().__init__()
        
        ResidualBlock.printOutsize = printOutsize
        ResidualBlock2.printOutsize = printOutsize
        BottleneckBlock.printOutsize = printOutsize
        HighwayBlock.printOutsize = printOutsize

        self.network = network
        

    def forward(self, x):

        y = self.network(x)
        
        return y

