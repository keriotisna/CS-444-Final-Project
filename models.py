from torch import nn
from blocks import *

"""
highwaynetv2 is just highwaynetv1 but the highwaySegments are now Bottleneck4 blocks instead of Bottleneck5
        which means there is no normalization between residual additions
"""

testModel = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=7),
            nn.BatchNorm2d(num_features=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=7),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(in_features=196, out_features=10),
            nn.Softmax(dim=0)
        )


baseline130k = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.Flatten(),
    
    nn.Linear(in_features=512, out_features=64),
    nn.ReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

baseline130kN = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=16),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=32),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.Flatten(),
    
    nn.Linear(in_features=512, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

baseline430k = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.Flatten(),
    
    nn.Linear(in_features=512, out_features=64),
    nn.ReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

baseline430kN = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=16),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=32),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.Flatten(),
    
    nn.Linear(in_features=512, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

fc0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(in_features=144, out_features=10),
            nn.Softmax(dim=0)
        )


fc1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(in_features=144, out_features=10)
        )


fc1Leaky = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            ResidualBlock(channelCount=16, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            ResidualBlock(channelCount=16, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            ResidualBlock(channelCount=16, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            ResidualBlock(channelCount=16, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            ResidualBlock(channelCount=16, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            ResidualBlock(channelCount=16, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            ResidualBlock(channelCount=16, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            
            nn.Flatten(),
            nn.Linear(in_features=144, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.LeakyReLU(),
            
            nn.Linear(in_features=64, out_features=10),
            nn.LeakyReLU(),
        )

fc2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            ResidualBlock2(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            ResidualBlock2(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            ResidualBlock2(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            ResidualBlock2(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            ResidualBlock2(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            ResidualBlock2(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            ResidualBlock2(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(in_features=144, out_features=10)
    )


bottleneck1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            BottleneckBlock(in_channels=32, encode_channels=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock(in_channels=32, encode_channels=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock(in_channels=32, encode_channels=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            BottleneckBlock(in_channels=32, encode_channels=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            BottleneckBlock(in_channels=32, encode_channels=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            BottleneckBlock(in_channels=32, encode_channels=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            BottleneckBlock(in_channels=32, encode_channels=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            BottleneckBlock(in_channels=32, encode_channels=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(in_features=288, out_features=10)
        )

bottleneck2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            BottleneckBlock(in_channels=32, encode_channels=1, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock(in_channels=32, encode_channels=1, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock(in_channels=32, encode_channels=1, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock(in_channels=32, encode_channels=1, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock(in_channels=32, encode_channels=1, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock(in_channels=32, encode_channels=1, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock(in_channels=32, encode_channels=1, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock(in_channels=32, encode_channels=1, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            nn.Flatten(),
            nn.Linear(in_features=288, out_features=10)
        )

bottleneck2R = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            BottleneckBlock2(in_channels=32, encode_channels=1, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock2(in_channels=32, encode_channels=1, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock2(in_channels=32, encode_channels=1, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock2(in_channels=32, encode_channels=1, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock2(in_channels=32, encode_channels=1, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock2(in_channels=32, encode_channels=1, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock2(in_channels=32, encode_channels=1, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock2(in_channels=32, encode_channels=1, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(in_features=32, out_features=144),
            nn.LayerNorm(normalized_shape=144),
            nn.LeakyReLU(),

            nn.Linear(in_features=144, out_features=10),
            nn.LayerNorm(normalized_shape=10),
            nn.LeakyReLU(),
        )


bottleneck3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            BottleneckBlock3(in_channels=32, encode_channels=3, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock3(in_channels=32, encode_channels=3, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock3(in_channels=32, encode_channels=3, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock3(in_channels=32, encode_channels=3, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock3(in_channels=32, encode_channels=3, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock3(in_channels=32, encode_channels=3, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock3(in_channels=32, encode_channels=3, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock3(in_channels=32, encode_channels=3, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(in_features=288, out_features=144),
            nn.LayerNorm(normalized_shape=144),
            nn.LeakyReLU(),

            nn.Linear(in_features=144, out_features=10),
            nn.LayerNorm(normalized_shape=10),
            nn.LeakyReLU(),
        )


bottleneck4 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            ResidualBlock(channelCount=16),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            BottleneckBlock4(in_channels=32, encode_factor=4, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock4(in_channels=32, encode_factor=4, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock4(in_channels=32, encode_factor=4, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock4(in_channels=32, encode_factor=4, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock4(in_channels=32, encode_factor=4, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock4(in_channels=32, encode_factor=4, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock4(in_channels=32, encode_factor=4, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            BottleneckBlock4(in_channels=32, encode_factor=4, activation=nn.LeakyReLU()),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(in_features=288, out_features=144),
            nn.LayerNorm(normalized_shape=144),
            nn.LeakyReLU(),

            nn.Linear(in_features=144, out_features=10),
            nn.LayerNorm(normalized_shape=10),
            nn.LeakyReLU(),
        )


bottleneck4v2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            BottleneckBlock4(in_channels=32, encode_factor=4),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            BottleneckBlock4(in_channels=64, encode_factor=4),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            BottleneckBlock4(in_channels=128, encode_factor=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            BottleneckBlock4(in_channels=128, encode_factor=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            
            BottleneckBlock4(in_channels=128, encode_factor=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            BottleneckBlock4(in_channels=128, encode_factor=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            BottleneckBlock4(in_channels=128, encode_factor=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            BottleneckBlock4(in_channels=128, encode_factor=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            BottleneckBlock4(in_channels=128, encode_factor=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            
            nn.Linear(in_features=128, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.ReLU(),
            
            nn.Linear(in_features=64, out_features=10),
            # Usually, we just send out raw logits to the loss function with no layer norms or activation functions
        )


# Remove redundant norms and activations
bottleneck4v3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            BottleneckBlock4(in_channels=32, encode_factor=4),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            BottleneckBlock4(in_channels=64, encode_factor=4),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            BottleneckBlock4(in_channels=128, encode_factor=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            BottleneckBlock4(in_channels=128, encode_factor=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            BottleneckBlock4(in_channels=128, encode_factor=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            BottleneckBlock4(in_channels=128, encode_factor=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock4(in_channels=128, encode_factor=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            BottleneckBlock4(in_channels=128, encode_factor=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            BottleneckBlock4(in_channels=128, encode_factor=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            
            nn.Linear(in_features=128, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.ReLU(),
            
            nn.Linear(in_features=64, out_features=10),
            # Usually, we just send out raw logits to the loss function with no layer norms or activation functions
        )


# Reduce complexity
bottleneck4v4 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            BottleneckBlock4(in_channels=32, encode_factor=4),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            BottleneckBlock4(in_channels=64, encode_factor=4),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            BottleneckBlock4(in_channels=96, encode_factor=4),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock4(in_channels=96, encode_factor=4),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock4(in_channels=96, encode_factor=4),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            
            nn.Linear(in_features=384, out_features=128),
            nn.LayerNorm(normalized_shape=128),
            nn.ReLU(),
            
            nn.Linear(in_features=128, out_features=10),
            # Usually, we just send out raw logits to the loss function with no layer norms or activation functions
        )


# Aggressive downsampling before bottleblocks
bottleneck4v5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock4(in_channels=64, encode_factor=4),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock4(in_channels=64, encode_factor=4),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            BottleneckBlock4(in_channels=64, encode_factor=4),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Flatten(),
            
            nn.Linear(in_features=256, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.ReLU(),
            
            nn.Linear(in_features=64, out_features=10),
            # Usually, we just send out raw logits to the loss function with no layer norms or activation functions
    )



# THIS IS STILL USING BOTTLENECK4
bottleneck5v1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock4(in_channels=64, encode_factor=4),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock4(in_channels=64, encode_factor=4),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            BottleneckBlock4(in_channels=64, encode_factor=4),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Flatten(),
            
            nn.Linear(in_features=256, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.ReLU(),
            
            nn.Linear(in_features=64, out_features=10),
            # Usually, we just send out raw logits to the loss function with no layer norms or activation functions
    )


bottleneck5v2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock5(in_channels=64, encode_factor=4),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BottleneckBlock5(in_channels=64, encode_factor=4),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            BottleneckBlock5(in_channels=64, encode_factor=4),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Flatten(),
            
            nn.Linear(in_features=256, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.ReLU(),
            
            nn.Linear(in_features=64, out_features=10),
            # Usually, we just send out raw logits to the loss function with no layer norms or activation functions
    )



highwaynetv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            HighwayBlock(in_channels=64, highwayLength=3),

            nn.Flatten(),

            nn.Linear(in_features=256, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=10),
    )


highwaynetv3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
                *[BottleneckBlock4(in_channels=64, encode_factor=4) for _ in range(6)]
                )),

            nn.Flatten(),

            nn.Linear(in_features=256, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=10),
    )


highwaynetv4 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
                *[BottleneckBlock5(in_channels=64, encode_factor=4) for _ in range(6)]
                )),

            nn.Flatten(),

            nn.Linear(in_features=256, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=10),
    )

highwaynetv5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
                *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4) for _ in range(6)]
                )),

            nn.Flatten(),

            nn.Linear(in_features=512, out_features=256),
            nn.LayerNorm(normalized_shape=256),
            nn.ReLU(),
            
            nn.Linear(in_features=256, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.ReLU(),
            
            nn.Linear(in_features=64, out_features=10),
    )


highwaynetv6 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
                *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(6)]
            )),

            HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
                *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(6)]
            )),
            
            nn.Flatten(),

            nn.Linear(in_features=512, out_features=256),
            nn.LayerNorm(normalized_shape=256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            
            nn.Linear(in_features=256, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            
            nn.Linear(in_features=64, out_features=10),
    )

highwaynetv7 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
                *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4) for _ in range(6)]
                )),

            HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
                *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4) for _ in range(6)]
                )),

            nn.Flatten(),

            nn.Linear(in_features=512, out_features=256),
            nn.LayerNorm(normalized_shape=256),
            nn.ReLU(),
            
            nn.Linear(in_features=256, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.ReLU(),
            
            nn.Linear(in_features=64, out_features=10),
    )


highwaynetv8 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
                *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4) for _ in range(6)]
                )),

            HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
                *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4) for _ in range(6)]
                )),

            nn.Flatten(),

            nn.Linear(in_features=512, out_features=256),
            nn.LayerNorm(normalized_shape=256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            
            nn.Linear(in_features=256, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.ReLU(),
            
            nn.Linear(in_features=64, out_features=10),
    )

highwaynetv8L = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
                *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4, activation=nn.LeakyReLU()) for _ in range(6)]
                )),

            HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
                *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4, activation=nn.LeakyReLU()) for _ in range(6)]
                )),

            nn.Flatten(),

            nn.Linear(in_features=512, out_features=256),
            nn.LayerNorm(normalized_shape=256),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            
            nn.Linear(in_features=256, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.LeakyReLU(),
            
            nn.Linear(in_features=64, out_features=10),
    )


doubleBottlev1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            DoubleEncodeBottleneckBlock(in_channels=64, encode_factor1=4, encode_factor2=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            nn.Flatten(),
            
            nn.Linear(in_features=512, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            
            nn.Linear(in_features=64, out_features=10),
    )



doubleBottlev2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            DoubleEncodeBottleneckBlock(in_channels=64, encode_factor1=4, encode_factor2=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),

            nn.Flatten(),
            
            nn.Linear(in_features=512, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            
            nn.Linear(in_features=64, out_features=10),
    )


doubleBottlev3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4) for _ in range(12)],
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(in_features=512, out_features=256),
            nn.LayerNorm(normalized_shape=256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            
            nn.Linear(in_features=256, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            
            nn.Linear(in_features=64, out_features=10),
    )

doubleBottlev4 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(12)],
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(in_features=512, out_features=256),
            nn.LayerNorm(normalized_shape=256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            
            nn.Linear(in_features=256, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            
            nn.Linear(in_features=64, out_features=10),
    )

doubleBottlev5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(12)],
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(in_features=512, out_features=256),
            nn.LayerNorm(normalized_shape=256),
            nn.ReLU(),
            
            nn.Linear(in_features=256, out_features=64),
            nn.LayerNorm(normalized_shape=64),
            nn.ReLU(),
            
            nn.Linear(in_features=64, out_features=10),
    )

# TODO: Try models closer to baseline like baseline430kN but add highway segments between