import os
from pandas import read_csv
import torch
import torchvision as tv
from torch.utils.data import Dataset, DataLoader, random_split
from icecream import ic
from PIL import Image
import torchvision.transforms.v2 as v2
from torchvision.io import read_image
import matplotlib.pyplot as plt
import random
import numpy as np



def displayImageGrid(images: list, H: int, W: int=0, shuffle=False, figsize=None):
    """
    Display list of images in a grid (H, W) without boundaries. The images MUST be the same size or this will probably look weird.

    Parameters:
    images: List of numpy arrays representing the images. The images should be the same size
    H: Number of rows.
    W: Number of columns.
    """
    
    numImages = len(images)
    
    # Shuffle images before so we can get a good sampling
    if shuffle:
        random.shuffle(images)
    
    # If no width is defined, we assume a single row of images
    if W == 0:
        W = numImages
    
    if numImages < H * W:
        raise ValueError(f"Number of images ({len(images)}) is smaller than given grid size!")
    
    # Shrink figure size if plotting lots of images
    if figsize is None:
        fig = plt.figure(figsize=(W/5, H/5))
    else:
        fig = plt.figure(figsize=figsize)

    for i in range(H * W):
        img = images[i]
        ax = fig.add_subplot(H, W, i+1)
        ax.imshow(img)

        # Remove axis details
        ax.axis('off')
        
        # Adjust the position of the axis for each image
        ax.set_position([i%W/W, 1-(i//W+1)/H, 1/W, 1/H])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


class CIFAR10Dataset(Dataset):
    
    defaultTransform = tv.transforms.Compose([
        # Ensure everything is in the right size and format before ending our transforms
        v2.Resize(size=(32, 32), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    def __init__(self, rootDirectory, csvFilename, dataFolder, transform=None):
        
        
        csvPath = os.path.join(rootDirectory, csvFilename)
        annotations = read_csv(csvPath)
        
        dataPath = os.path.join(rootDirectory, dataFolder)
        
        # print(annotations.dtypes)
        
        # Convert labels to string https://stackoverflow.com/questions/22231592/pandas-change-data-type-of-series-to-string
        # annotations["label"] = annotations["label"].astype('string')
        # print(annotations.dtypes)

        # Replace string labels with integer labels 0-9
        unique = annotations['label'].unique()#.astype(str)
        stringNumberLabelEnumerations = enumerate(sorted(list(unique)))
        
        # Create dictionaries for easily mapping to and from strings or integers as picture labels
        stringNumberMappings = {string: number for number, string in stringNumberLabelEnumerations}
        numberStringMappings = {number: string for string, number in stringNumberMappings.items()}
        
        
        # .mask acts similarly to boolean indexing. If the condition is true, the current value is replaced with num+1
        for string, num in stringNumberMappings.items():
            annotations['label'].mask(annotations['label'] == string, num, inplace=True)        
        
        self.stringNumberMappings = stringNumberMappings
        self.numberStringMappings = numberStringMappings
        
        self.annotations = annotations
        self.dataPath = dataPath
        
        if transform is not None:
            # Splice in the given transform before setting everything in the right format for the network
            transform = tv.transforms.Compose([
                transform,
                self.defaultTransform
            ])
        else:
            transform = self.defaultTransform
        
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        
        imageId = self.annotations.iloc[index, 0]
        imageName = os.path.join(self.dataPath, f"{imageId}.png")
        image = Image.open(imageName)
        
        label = self.annotations.iloc[index, 1]
        
        if self.transform:
            image = self.transform(image)

        return image, label












def main():


    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        # Add any other transformations you need
    ])

    # Create dataset instances
    fullDataset = CIFAR10Dataset(rootDirectory='cifar-10', csvFilename='trainLabels.csv', dataFolder='train', transform=transform)
    # test_dataset = CIFAR10LocalDataset(csv_file='path/to/testLabels.csv', root_dir='path/to/test', transform=transform)

    generator = torch.Generator().manual_seed(42)
    TRAIN_DATASET, VALIDATION_DATSET, TEST_DATSAET = random_split(fullDataset, [0.8, 0.1, 0.1], generator=generator)

    # Create DataLoader instances
    BATCH_SIZE = 256
    trainLoader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True)

    train_features, train_labels = next(iter(trainLoader))
    
    trainFeaturesArray = train_features.numpy().transpose(2, 3, 1, 0)
    trainLabelsArray = train_labels.numpy()
    
    displayImageGrid([trainFeaturesArray[..., idx] for idx in range(BATCH_SIZE)], H=8, W=32)
    
    ic(train_features.size())
    ic(train_labels.size())


if __name__ == "__main__":
    main()