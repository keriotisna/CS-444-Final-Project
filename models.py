from torch import nn
from blocks import *
import torchvision as tv

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

baseline13MN = nn.Sequential(
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
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),

    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=512),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=512),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=512),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=512),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=512),
    nn.ReLU(),
    
    nn.Flatten(),
    
    nn.Linear(in_features=2048, out_features=256),
    nn.LayerNorm(normalized_shape=256),
    nn.ReLU(),
    
    nn.Linear(in_features=256, out_features=10)
)

baseline36MN = nn.Sequential(
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
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),

    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=512),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=1024),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=1024),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=1024),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=1024),
    nn.ReLU(),
    
    nn.AvgPool2d(kernel_size=2),
    
    nn.Flatten(),
    
    nn.Linear(in_features=1024, out_features=256),
    nn.LayerNorm(normalized_shape=256),
    nn.ReLU(),
    
    nn.Linear(in_features=256, out_features=10)
)

baseline108MN = nn.Sequential(
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
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=512),
    nn.ReLU(),

    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=512),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=512),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=512),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=1024),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=2048),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=2048),
    nn.ReLU(),
    
    nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=2048),
    nn.ReLU(),
    
    nn.AvgPool2d(kernel_size=2),
    
    nn.Flatten(),
    
    nn.Linear(in_features=2048, out_features=256),
    nn.LayerNorm(normalized_shape=256),
    nn.ReLU(),
    
    nn.Linear(in_features=256, out_features=10)
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



highwaynetv9 = nn.Sequential(
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
            
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(3)]
        )),

    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(3)]
        )),

    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(3)]
        )),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(3)]
        )),

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

highwaynetv10 = nn.Sequential(
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
            
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4) for _ in range(3)]
        )),

    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4) for _ in range(3)]
        )),

    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4) for _ in range(3)]
        )),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4) for _ in range(3)]
        )),

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

highwaynetv11 = nn.Sequential(
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
            
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(12)]
        )),

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





bigModel1 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),

    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),

    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(8)]
    )),

    nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),

    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=2, encode_factor2=2) for _ in range(8)]
    )),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),

    nn.Flatten(),

    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),

    nn.Linear(in_features=64, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),

    nn.Linear(in_features=64, out_features=10),
    )

bigmodel2 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=16),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    HighwayBlock(in_channels=16, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=16, encode_factor1=2, encode_factor2=1) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=32),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=32, encode_factor1=2, encode_factor2=1) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=64, encode_factor1=2, encode_factor2=2) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=4) for _ in range(3)]
    )),
    
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

bigmodel3 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),

    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),

    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=256, encode_factor1=2, encode_factor2=2) for _ in range(8)]
    )),

    nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),

    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=2, encode_factor2=2) for _ in range(8)]
    )),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),

    nn.Flatten(),

    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),

    nn.Linear(in_features=64, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),

    nn.Linear(in_features=64, out_features=10),
    )

bigmodel4 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=16),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    HighwayBlock(in_channels=16, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=16, encode_factor1=2, encode_factor2=1) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=32),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=32, encode_factor1=2, encode_factor2=1) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=64, encode_factor1=2, encode_factor2=2) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock(in_channels=128, encode_factor1=2, encode_factor2=2) for _ in range(6)]
    )),
    
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


bigModel1_DBN2 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=16),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),

    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),

    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(8)]
    )),

    nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),

    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=2) for _ in range(8)]
    )),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),

    nn.Flatten(),

    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),

    nn.Linear(in_features=64, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),

    nn.Linear(in_features=64, out_features=10),
    )

bigmodel2_DBN2 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=16),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    HighwayBlock(in_channels=16, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=16, encode_factor1=2, encode_factor2=1) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=32),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=2, encode_factor2=2) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=4) for _ in range(3)]
    )),
    
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

bigmodel3_DBN2 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=32),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=2, encode_factor2=1) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=2) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    
    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=2) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    
    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.AvgPool2d(kernel_size=2, stride=1, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

bigmodel4_DBN2 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=32),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.LeakyReLU(),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.LeakyReLU()) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.LeakyReLU(),
    
    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=2, encode_factor2=1, activation=nn.LeakyReLU()) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.LeakyReLU(),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=2, activation=nn.LeakyReLU()) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.LeakyReLU(),
    
    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=2, activation=nn.LeakyReLU()) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.LeakyReLU(),
    
    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4, activation=nn.LeakyReLU()) for _ in range(3)]
    )),
    
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.LeakyReLU(),
    
    nn.AvgPool2d(kernel_size=2, stride=1, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.LeakyReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

bigmodel5_DBN2 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=32),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.LeakyReLU(),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.LeakyReLU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.LeakyReLU(),
    
    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=2, encode_factor2=1, activation=nn.LeakyReLU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.LeakyReLU(),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=2, activation=nn.LeakyReLU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.LeakyReLU(),
    
    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=2, activation=nn.LeakyReLU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.LeakyReLU(),
    
    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4, activation=nn.LeakyReLU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.LeakyReLU(),
    
    nn.AvgPool2d(kernel_size=2, stride=1, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.LeakyReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

bigmodel6_DBN2 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=32),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=2, encode_factor2=1) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=2) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    
    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=2) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),
    
    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.ReLU(),
    
    nn.AvgPool2d(kernel_size=2, stride=1, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)


####################################################################################################################################
# ALLEN MODELS
####################################################################################################################################


allenModelv1_standard = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    

    *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1) for _ in range(6)],
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1) for _ in range(6)],
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(6)],
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),

    *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(30)],

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

allenModelv2_highway = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(6)]
    )),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(6)]
    )),
    
    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(6)]
    )),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(6)]
    )),
    
    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(6)]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)


allenModelv3_convFinal = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1) for _ in range(6)],
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1) for _ in range(6)],
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(6)],
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),

    *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(30)],

    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=1, padding=0),
    nn.ReLU(),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)


allenModelv1Lite_standard = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    

    *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1) for _ in range(6)],
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1) for _ in range(6)],
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(6)],
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),

    # *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(30)],

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

allenModelv2Lite_highway = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(6)]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

allenModelv3Lite_convFinal = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1) for _ in range(6)],
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1) for _ in range(6)],
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(6)],
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),

    *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(6)],

    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=1, padding=0),
    nn.ReLU(),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

allenModelv4Lite_highway_avgPool = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(6)]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.LayerNorm(256),    
    nn.Linear(in_features=256, out_features=10)
)

allenModelv5Lite_highway_funnel = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1) for _ in range(4)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(8)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(12)]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

allenModelv5Lite_highway_Deep = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1) for _ in range(12)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1) for _ in range(12)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(12)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ReLU(),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(12)]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.LayerNorm(256),    
    nn.Linear(in_features=256, out_features=10)
)

allenModelv6Lite_highway_instanceNorm = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.InstanceNorm2d(num_features=32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.InstanceNorm2d(num_features=64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.InstanceNorm2d(num_features=128),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.InstanceNorm2d(num_features=256),
    nn.ReLU(),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4) for _ in range(6)]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

###########################################################################################################
# Wilson models
###########################################################################################################

wilsonNetv1_ELU = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.ELU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.ELU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ELU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1, activation=nn.ELU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ELU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2, activation=nn.ELU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ELU(),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4, activation=nn.ELU()) for _ in range(6)]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ELU(),
    
    nn.Linear(in_features=64, out_features=10)
)

wilsonNetv2_ELU_frontDeep = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.ELU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.ELU()) for _ in range(12)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ELU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1, activation=nn.ELU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ELU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2, activation=nn.ELU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ELU(),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4, activation=nn.ELU()) for _ in range(6)]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ELU(),
    
    nn.Linear(in_features=64, out_features=10)
)

wilsonNetv3_ELU_rearDeep = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.ELU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.ELU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ELU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1, activation=nn.ELU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ELU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2, activation=nn.ELU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ELU(),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4, activation=nn.ELU()) for _ in range(12)]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ELU(),
    
    nn.Linear(in_features=64, out_features=10)
)

wilsonNetv4_ELU_rearDoubleDeep = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.ELU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.ELU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.ELU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1, activation=nn.ELU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.ELU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2, activation=nn.ELU()) for _ in range(12)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.ELU(),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4, activation=nn.ELU()) for _ in range(12)]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.ELU(),
    
    nn.Linear(in_features=64, out_features=10)
)


wilsonNetv5_PReLU = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2, activation=nn.PReLU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.PReLU(),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4, activation=nn.PReLU()) for _ in range(6)]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)


wilsonNetv6_PReLU_rearDeep = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=64),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=128),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2, activation=nn.PReLU()) for _ in range(6)]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(num_features=256),
    nn.PReLU(),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=4, activation=nn.PReLU()) for _ in range(12)]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)


###########################################################################################################
# Jesse models
###########################################################################################################

# Introduction of branch blocks
jesseNetv1 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=8),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlock(in_channels=8, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=8, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=8, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=8, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=8, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ]),

    BranchBlock(in_channels=32, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=2, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=4, encode_factor2=2, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=4, encode_factor2=2, activation=nn.PReLU()),
    ]),
    nn.MaxPool2d(kernel_size=2, stride=2),

    BranchBlock(in_channels=128, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=2, activation=nn.PReLU()),
    ]),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=512, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=4, encode_factor2=4, activation=nn.PReLU()) for _ in range(6)]
    )),
    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=512, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

# Less encode 2 when compared to JesseNetv1
jesseNetv2 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=8),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlock(in_channels=8, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=8, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=8, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=8, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=8, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ]),

    BranchBlock(in_channels=32, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ]),
    nn.MaxPool2d(kernel_size=2, stride=2),

    BranchBlock(in_channels=128, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ]),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=512, highwaySequence=nn.Sequential(
        *[DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=2, activation=nn.PReLU()) for _ in range(6)]
    )),
    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=512, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

# Consistent encoding factors
jesseNetv3 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=8),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlock(in_channels=8, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=8, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=8, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=8, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=8, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ]),

    BranchBlock(in_channels=32, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ]),
    nn.MaxPool2d(kernel_size=2, stride=2),

    BranchBlock(in_channels=128, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ]),
    nn.MaxPool2d(kernel_size=2, stride=2),

    BranchBlock(in_channels=512, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ]),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=2048, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

# Averaging once at final branch to save memory
jesseNetv4 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlock(in_channels=32, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ]),

    BranchBlock(in_channels=128, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ]),
    nn.MaxPool2d(kernel_size=2, stride=2),

    BranchBlock(in_channels=512, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ]),
    nn.MaxPool2d(kernel_size=2, stride=2),

    BranchBlock(in_channels=2048, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=2048, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=2048, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=2048, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=2048, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ], averageChannels=True),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=2048, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

# Full averaging between branches and branch highway
jesseNetv5 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlock(in_channels=32, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ], averageChannels=True),

    nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=128),
    nn.PReLU(),

    BranchBlock(in_channels=128, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ], averageChannels=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlock(in_channels=128, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ], averageChannels=True),
    
    nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=512),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=512, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=512, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(6)
        ]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=512, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

# JesseNetv5 with reversed encode factors
jesseNetv5_2_reverseEncode = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlock(in_channels=32, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=2, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=4, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=8, activation=nn.PReLU()),
    ], averageChannels=True),

    nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=128),
    nn.PReLU(),

    BranchBlock(in_channels=128, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=2, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=4, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=8, activation=nn.PReLU()),
    ], averageChannels=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlock(in_channels=128, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=2, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=4, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=8, activation=nn.PReLU()),
    ], averageChannels=True),
    
    nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=512),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=512, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=512, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=8, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(6)
        ]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=512, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)


# Adds more branches with heavy encode values
jesseNetv5_3_wideBranchesLinearEncode = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlock(in_channels=32, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=6, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=10, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=12, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=14, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=16, encode_factor2=1, activation=nn.PReLU()),
    ], averageChannels=True),

    nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=128),
    nn.PReLU(),

    BranchBlock(in_channels=128, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=6, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=10, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=12, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=14, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=16, encode_factor2=1, activation=nn.PReLU()),
    ], averageChannels=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlock(in_channels=128, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=6, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=10, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=12, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=14, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=16, encode_factor2=1, activation=nn.PReLU()),
    ], averageChannels=True),
    
    nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=512),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=512, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=512, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=6, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=10, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=12, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=14, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=16, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(6)
        ]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=512, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

# Double identical branch counts compared to 5
jesseNetv5_4_doubleWideBranches = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlock(in_channels=32, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ], averageChannels=True),

    nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=128),
    nn.PReLU(),

    BranchBlock(in_channels=128, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ], averageChannels=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlock(in_channels=128, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ], averageChannels=True),
    
    nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=512),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=512, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=512, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(6)
        ]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=512, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

# Removes highway segment and replaces without long residual
jesseNetv5_5_noHighway = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlock(in_channels=32, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ], averageChannels=True),

    nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=128),
    nn.PReLU(),

    BranchBlock(in_channels=128, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ], averageChannels=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlock(in_channels=128, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ], averageChannels=True),
    
    nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=512),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    *[
        BranchBlock(in_channels=512, branches=[
            DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
            DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
            DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
            DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
        ], averageChannels=True) for _ in range(6)
    ],

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=512, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

# Longer branch highway when compared to JesseNetv5
jesseNetv6 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlock(in_channels=32, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ], averageChannels=True),

    nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=128),
    nn.PReLU(),

    BranchBlock(in_channels=128, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ], averageChannels=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    BranchBlock(in_channels=128, branches=[
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
        DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
    ], averageChannels=True),
    
    nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=512),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=512, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=512, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(12)
        ]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=512, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)



# Full highway for all branch blocks
jesseNetv7_multiHighway = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=32, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(3)
        ]
    )),

    nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=128),
    nn.PReLU(),

    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=128, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(3)
        ]
    )),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=128, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(3)
        ]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=512),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=512, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=512, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(3)
        ]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=512, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

# Add more single encode bottlenecks to try and reduce varaince in sublayers
jesseNetv7_2_multiHighway_duplicateBottle = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=32, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(2)
        ]
    )),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=64),
    nn.PReLU(),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=64, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(2)
        ]
    )),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=128),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=128, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(2)
        ]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=512),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=512, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=512, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(2)
        ]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=512, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)


jesseNetv7_2_multiHighway_duplicateBottleRevEncode = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=32, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=8, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(2)
        ]
    )),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=64),
    nn.PReLU(),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=64, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=8, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(2)
        ]
    )),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=128),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=128, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=8, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(2)
        ]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=512),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=512, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=512, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=8, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(2)
        ]
    )),
)


jesseNetv7_2_multiHighway_duplicateBottleRevEncodex2 = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=32, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=8, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(4)
        ]
    )),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=64),
    nn.PReLU(),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=64, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=8, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(4)
        ]
    )),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=128),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=128, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=8, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(4)
        ]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=512),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=512, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=512, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=1, encode_factor2=8, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(4)
        ]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=512, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

jesseNetv7_2_multiHighway_duplicateBottleRevEncodex2Compact = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=32, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=8, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(4)
        ]
    )),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=64),
    nn.PReLU(),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=64, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=2, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=2, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=2, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=2, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=2, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=2, encode_factor2=8, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=2, encode_factor2=16, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(4)
        ]
    )),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=128),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=128, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=8, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(4)
        ]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=512),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=512, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=512, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=2, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=4, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=512, encode_factor1=2, encode_factor2=8, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(4)
        ]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=512, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)


jesseNetv7_3_multiHighway_mini = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=32, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(3)
        ]
    )),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=64),
    nn.PReLU(),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=64, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(3)
        ]
    )),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=128),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=128, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(3)
        ]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=256),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=256, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=1, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(3)
        ]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)

# Remove uncompressed layers to reduce computation
jesseNetv7_4_multiHighway_micro = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.BatchNorm2d(num_features=32),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=32, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=32, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=32, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(3)
        ]
    )),

    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=64),
    nn.PReLU(),

    HighwayBlock(in_channels=64, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=64, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=64, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(3)
        ]
    )),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=128),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    HighwayBlock(in_channels=128, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=128, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=128, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(3)
        ]
    )),
    
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(num_features=256),
    nn.PReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),

    HighwayBlock(in_channels=256, highwaySequence=nn.Sequential(
        *[
            BranchBlock(in_channels=256, branches=[
                DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=2, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=4, encode_factor2=1, activation=nn.PReLU()),
                DoubleEncodeBottleneckBlock2(in_channels=256, encode_factor1=8, encode_factor2=1, activation=nn.PReLU()),
            ], averageChannels=True) for _ in range(3)
        ]
    )),

    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
    
    nn.Flatten(),
    
    nn.Linear(in_features=256, out_features=64),
    nn.LayerNorm(normalized_shape=64),
    nn.PReLU(),
    
    nn.Linear(in_features=64, out_features=10)
)


def _getResNet18(fineTuneLayers:nn.Sequential, freezePreTrained=True) -> nn.Module:

    sequentialResnet = list(tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT).children())
    # Insert flatten layer since it doesn't exist for some reason
    sequentialResnet.insert(len(sequentialResnet)-1, nn.Flatten())
    model = nn.Sequential(*sequentialResnet)

    # Freeze pre-trained weights
    if freezePreTrained:
        for param in model.parameters():
            param.requires_grad_(False)

    model = nn.Sequential(model, *fineTuneLayers)
    
    return model
    
resNet18Test = _getResNet18(fineTuneLayers=nn.Sequential(
    nn.Linear(in_features=1000, out_features=256),
    nn.LayerNorm(normalized_shape=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=10),
))