
import torchvision as tv
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Dataset
from icecream import ic
import matplotlib.pyplot as plt
import random
from dataLoading import CIFAR10Dataset


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

def showTransform(imageName:str, transform:tv.transforms.Compose=None):
    
    """
    Shows an image transformed by the given transform for data augmentation
    """
    
    if transform is None:
        transform = tv.transforms.Compose([
        v2.Identity()
    ])
    
    baseTransform = CIFAR10Dataset.defaultTransform
    
    image = tv.io.read_image(imageName)

    # image = baseTransform(image)

    image = transform(image)
    image = baseTransform(image)

    image = image.numpy().transpose(1, 2, 0)

    plt.figure(figsize=(3, 3))
    plt.imshow(image)
    plt.title(f'{transform}\n{image.shape}')
    plt.show()
    
    
def showDatasetSamples(dataloader:DataLoader, datasetClass:Dataset):
    
    features, labels = next(iter(dataloader))

    trainFeaturesArray = features.numpy().transpose(2, 3, 1, 0)
    trainLabelsArray = labels.numpy()

    displayImageGrid([trainFeaturesArray[..., idx] for idx in range(256)], H=8, W=32, figsize=(20, 5))    

    [print(datasetClass.numberStringMappings[val], end='\t') for val in trainLabelsArray[:32]]
    print()
    [print(val, end='\t') for val in trainLabelsArray[:32]]

    ic(features.size())
    ic(labels.size())