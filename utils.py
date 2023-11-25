
import torchvision as tv
import torchvision.transforms.v2 as v2
import torch
from torchsummary import summary
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from icecream import ic
import matplotlib.pyplot as plt
import random
from dataLoading import CIFAR10Dataset


class TransformableSubset(Dataset):
    
    """
    A wrapper class for applying transforms to a subset of a dataset since a subset doesn't have a .transform attribute
    Mainly used for applying training and val/testing transforms to different portions of the data.
    """
    
    def __init__(self, subset:Subset, fullDataset:Dataset, transform:v2.Compose=None):
        
        """
        Arguments:
            subset: A subset of an original dataset as retrieved via random_split
            fullDataset: The instance of the full dataset loaded, should have a class variable self.defaultTransform which notes
                what the first transform should be to ensure all subsequent transforms work as intended
            transform: A custom transform that will be applied to the given subset of the data.
        """
        
        self.subset = subset
        
        if transform:
            self.transform = v2.Compose(
                fullDataset.defaultTransform.transforms +
                transform.transforms
            )
        else:
            self.transform = fullDataset.defaultTransform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)



def validateModelIO(model:nn.Module, printSummary=True):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")

    model = model.to(device)

    if printSummary:
        print(model)
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters.")
        print(summary(model, input_size=(3, 32, 32), device=device))

    dummy_input = torch.randn(1, 3, 32, 32, device=device, dtype=torch.float)
    output = model(dummy_input)
    assert output.size() == (1, 10), f"Expected output size (1, 10), got {output.size()}!"
    print("Test passed!")




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


def showTransform(imageName:str, transform:v2.Compose=None):
    
    """
    Shows an image transformed by the given transform for data augmentation
    """
    
    if transform is None:
        transform = v2.Compose([
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

    print(f'MAX VALUE: {torch.max(features)}')
    print(f'MIN VALUE: {torch.min(features)}')

    trainFeaturesArray = features.numpy().transpose(2, 3, 1, 0)
    trainLabelsArray = labels.numpy()

    displayImageGrid([trainFeaturesArray[..., idx] for idx in range(256)], H=8, W=32, figsize=(20, 5))    

    [print(datasetClass.numberStringMappings[val], end='\t') for val in trainLabelsArray[:32]]
    print()
    [print(val, end='\t') for val in trainLabelsArray[:32]]

    ic(features.size())
    ic(labels.size())
    
    
    
def getDatasetNormalization(trainDataset:Dataset) -> tuple[float, float]:
    
    """
    Get the mean and standard deviation for a dataset with an augmentation transform already applied.

    Returns:
        (mean, std):
            The mean and standard deviation of the given dataset which can be used for normalization before training
    """
    
    BATCH_SIZE = 1024
    trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False, prefetch_factor=4)

    mean = 0.
    std = 0.
    for images, _ in trainLoader:
        
        batchSamples = images.size(0)
        # Get an image view of shape (batchSamples, C, H*W) which is faster than a transpose as we don't shift any data
        images = images.view(batchSamples, images.size(1), -1)
        # Calculate total mean and total std over dim=2
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    # Divide means and stdevs by number of samples
    mean /= len(trainLoader.dataset)
    std /= len(trainLoader.dataset)
    print(mean)
    print(std)
    
    return mean, std

@DeprecationWarning
def getNormalizedTransform(fullDataset:Dataset, customTransforms:v2.Compose, showSamples=False):
    
    """
    Return a modified customTransforms which adds normalization by the augmentation
    
    Arguments:
        customTransforms: The base data augmentation transform
        showSamples: Whether or not to show the samples resulting from the customTransform (without normalization)

    Returns:
        finalTransform: customTransforms with an appended normlization transform for training
    """
    
    trainSubset, validationSubset, testSubset = random_split(fullDataset, [0.8, 0.1, 0.1])

    trainDataset = TransformableSubset(trainSubset, fullDataset, transform=customTransforms)

    mean, std = getDatasetNormalization(trainDataset)

    trainLoader = DataLoader(trainDataset, batch_size=256, shuffle=True)
    
    if showSamples:
        showDatasetSamples(trainLoader, fullDataset)


    finalTransform = v2.Compose(
        customTransforms.transforms +
        # Normalize before creating the real dataset
        [v2.Normalize(mean=mean, std=std)]
    )
    
    return finalTransform



def getNormalizedTransforms(fullDataset:Dataset, trainTransform:v2.Compose, valTestTransform:v2.Compose, showSamples=False):
    
    """
    Return a modified customTransforms which adds normalization by the augmentation
    
    Arguments:
        trainTransform: The base data augmentation transform used for training data
        valTestTransform: The base data transform to be used for validation and testing. Can be Identity() if needed
        showSamples: Whether or not to show the samples resulting from the customTransform (without normalization)

    Returns:
        (normalizedTrainTransform, normalizedValTestTransform): training and validation transforms with added normalization
    """
    
    trainSubset, validationSubset, testSubset = random_split(fullDataset, [0.8, 0.1, 0.1])

    trainDataset = TransformableSubset(trainSubset, fullDataset, transform=trainTransform)

    mean, std = getDatasetNormalization(trainDataset)

    trainLoader = DataLoader(trainDataset, batch_size=256, shuffle=True)
    
    if showSamples:
        showDatasetSamples(trainLoader, fullDataset)


    # We want to normalize both the transforms by the same augmented statistics to ensure the distributions are similar
    normalizedTrainTransform = v2.Compose(
        trainTransform.transforms +
        [v2.Normalize(mean=mean, std=std)]
    )
    
    normalizedValTestTransform = v2.Compose(
        valTestTransform.transforms +
        [v2.Normalize(mean=mean, std=std)]
    )
    
    return normalizedTrainTransform, normalizedValTestTransform