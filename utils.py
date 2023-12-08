import torchvision as tv
import torchvision.transforms.v2 as v2
import torch
import torchinfo
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
# from icecream import ic
import matplotlib.pyplot as plt
import random
import numpy as np
from dataLoading import CIFAR10Dataset
from models import *
from transforms import *
from torch.profiler import profile, record_function, ProfilerActivity
from tabulate import tabulate


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



def validateModelIO(model:nn.Module, printSummary=True, batchSize=2) -> torchinfo.ModelStatistics:
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")

    model = model.to(device)

    dummy_input = torch.randn(batchSize, 3, 32, 32, device=device, dtype=torch.float)
    output = model(dummy_input)
    assert output.size() == (batchSize, 10), f"Expected output size ({batchSize}, 10), got {output.size()}!"

    summaryObject = torchinfo.summary(model=model, input_size=(batchSize, 3, 32, 32), device=device, mode='train', depth=20, verbose=0)

    if printSummary:
        print(model)
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters.")
        print(summaryObject)

    print("Test passed!")
    
    return summaryObject

def getModelSummary(model:nn.Module, batch_size:int, **kwargs) -> torchinfo.ModelStatistics:
    
    """
    Generates a model summary given a model and a batch size using the torchinfo.summary function
    
    Arguments:
        model: The model that the summary should be obtained for
        batch_size: The batch size of the model to be trained
        
    Returns:
        modelStatistics: A ModelStatistics object. 
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return torchinfo.summary(model=model, input_size=(batch_size, 3, 32, 32), device=device, mode='train', depth=20, verbose=0, **kwargs)

def getEstimatedModelSize(summary:torchinfo.ModelStatistics) -> float:
    
    """
    Gets an estimated VRAM usage for a model in MB
    
    Arguments:
        summary: A ModelStatistics object which holds all information about a model
        
    Returns:
        estimatedMemorySize: An estimate of model size in MB. This is NOT a guaranteed size, so models may use more or less memory than this function returns.
    """
    
    inputBytes = summary.total_input
    outputBytes = summary.total_output_bytes
    paramBytes = summary.total_param_bytes
    
    return round((inputBytes + outputBytes + paramBytes)/(1024**2), 3)

def getModel(modelName:str, printResult=True) -> nn.Sequential:
    
    """
    Get a model architecture based on its name. New models MUST have new names which means I
        can track all changes made to models to associate them with a performance.
        
    Arguments:
        modelName: The string name of the model, can be anything, but should be the same as the 
            variable name of the model itself
            
    Returns:
        model: Returns an nn.Sequential which is the model architecture to be trained
    """

    globalVars = globals()
    # Split actual model variable name from suffix
    actualName = modelName.rsplit('_', maxsplit=1)[0]
    if printResult:
        print(f'Got model: {actualName}')
    retrievedModel = globalVars[actualName]
    return retrievedModel


def getTransform(transformName:str, printResult=True) -> v2.Compose:
    
    """
    Get a transformation based on its name. New transforms MUST have new names which means I
        can track all changes made to transforms to associate them with a performance.
        
    Arguments:
        transformName: The string name of the transform, can be anything, but should be the same as the 
            variable name of the model itself
            
    Returns:
        retrievedTransform: Returns a transform which is the model architecture to be trained
    """

    globalVars = globals()

    retrievedTransform = globalVars.get(transformName, None)
    if printResult:
        print(f'Got transform: {transformName}')
        print(retrievedTransform)
    return retrievedTransform


def getBinPackingResults(values:list, MAX_CAP:float) -> tuple[list, list]:

    """
    Finds a near-optimal solution to the bin packing problem by minimizing the total number of bins a set of values will fit into 
    assuming each bin has an identical max capacity.
    
    This can be used to optimally schedule batches of network training to minimize the number of individual batches while maximizing parallelism.
    
    Arguments:
        values: A list of values which represents the estimated memory in MB each model will use during training
        MAX_CAP: A float representing the total MB capacity of the GPU
        
    Returns:
        (valueBins, indexBins)
        valueBins: A list of lists where each sublist represents the values of items that sum to less than MAX_CAP
        indexBins: A list of lists where each sublist holds the indices of the original values which should be grouped to minimize bin counts.
    """

    values = np.array(values, dtype=np.float32)
    
    sortedIndices = np.argsort(values, kind='stable')[::-1]
    values = values[sortedIndices]
    
    # Initialize empty bins and indices since we know at most, there can be N bins for N values
    bins = [[] for i in range(len(values))]
    binIndices = [[] for i in range(len(values))]
    binCapacities = [int(MAX_CAP - np.sum(bin)) for bin in bins]
    remainingCapacities = (np.ones((len(bins))) * MAX_CAP)
    
    for idx in range(len(values)):
        
        currentValue = values[idx]
        currentValueIdx = sortedIndices[idx]

        for bCapIdx, binCapacity in enumerate(binCapacities):
            
            remainingCapacity = binCapacity - currentValue
            remainingCapacities[bCapIdx] = remainingCapacity

        # If a value is out of the maximum range, add it to its own list
        if currentValue > MAX_CAP:
            bins.append([currentValue])
            binIndices.append([currentValueIdx])
            continue

        bestBinIdx = np.where(remainingCapacities >= 0)[0][np.argmin(remainingCapacities[remainingCapacities >= 0])]
        
        bins[bestBinIdx].append(currentValue)
        binIndices[bestBinIdx].append(currentValueIdx)
        binCapacities[bestBinIdx] = np.sum(binCapacities[bestBinIdx]) - currentValue

    return [bin for bin in bins if len(bin)>0], [idx for idx in binIndices if len(idx)>0]

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

    # ic(features.size())
    # ic(labels.size())
    
    
    
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
        break

    # Divide means and stdevs by number of samples
    # mean /= len(trainLoader.dataset)
    # std /= len(trainLoader.dataset)
    mean /= BATCH_SIZE
    std /= BATCH_SIZE
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



def getNormalizedTransforms(fullDataset:Dataset, trainTransform:v2.Compose, valTestTransform:v2.Compose, showSamples=False,
        customNormalization=None):
    
    """
    Return a modified customTransforms which adds normalization by the augmentation
    
    Arguments:
        trainTransform: The base data augmentation transform used for training data
        valTestTransform: The base data transform to be used for validation and testing. Can be Identity() if needed
        showSamples: Whether or not to show the samples resulting from the customTransform (without normalization)
        customNormalization: If not none, a custom normalization is used as defined in transforms.py which is useful for pre-trained models with their own normalization

    Returns:
        (normalizedTrainTransform, normalizedValTestTransform): training and validation transforms with added normalization
    """
    
    trainSubset, validationSubset, testSubset = random_split(fullDataset, [0.8, 0.1, 0.1])

    trainDataset = TransformableSubset(trainSubset, fullDataset, transform=trainTransform)

    # Get the normalization values on the augmented training dataset
    mean, std = getDatasetNormalization(trainDataset)

    trainLoader = DataLoader(trainDataset, batch_size=256, shuffle=True)
    
    if showSamples:
        showDatasetSamples(trainLoader, fullDataset)


    if customNormalization is not None:
        selectedNormalization = getTransform(customNormalization)
    else:
        selectedNormalization = v2.Compose([v2.Normalize(mean=mean, std=std)])

    # We want to normalize both the transforms by the same augmented statistics to ensure the distributions are similar
    normalizedTrainTransform = v2.Compose(
        trainTransform.transforms +
        selectedNormalization.transforms
    )
    
    normalizedValTestTransform = v2.Compose(
        valTestTransform.transforms +
        selectedNormalization.transforms
    )
    
    return normalizedTrainTransform, normalizedValTestTransform


def determinsticSplitFullDataset(fullDataset:Dataset, trainValTestSplit:list) -> tuple[Subset, Subset, Subset]:
    
    """
    Deterministically splits a full dataset into a training, validation and test dataset based on the provided split information

    Arguments:
        fullDataset: A dataset object which represents the raw dataset
        trainValTestSplit: A list that sums to 1 which represents what fraction of data should be in training, validation, or test splits

    Returns:
        (trainDataset, validataionDataset, testDataset)
        A tuple of 3 datasets of varying sizes determined by the trainValTestSplit
    """
    
    assert sum(trainValTestSplit) == 1
    
    dataIndices = np.arange(stop=50000)
    
    # Calculate the split sizes
    totalSize = len(fullDataset)
    trainSize = int(totalSize * trainValTestSplit[0])
    valSize = int(totalSize * trainValTestSplit[1])
    testSize = totalSize - trainSize - valSize

    # Deterministically shuffle dataset so we are consistent across runs and can train saved models without giving them access to test data
    currentRandomState = torch.random.get_rng_state()
    torch.random.manual_seed(11)
    shuffledIndices = torch.randperm(dataIndices.shape[0]) # 2441, 31547, 48866, ...
    torch.random.set_rng_state(currentRandomState)


    # Split the indices
    trainIndices = shuffledIndices[:trainSize].tolist() # These need to be lists instead of tensors for some reason
    valIndices = shuffledIndices[trainSize:trainSize + valSize].tolist()
    testIndices = shuffledIndices[trainSize + valSize:].tolist()

    # Create data subsets
    trainDataset = Subset(fullDataset, trainIndices)
    validationDataset = Subset(fullDataset, valIndices)
    testDataset = Subset(fullDataset, testIndices)
    
    return trainDataset, validationDataset, testDataset


# TODO: Write to allow conditional profiling depth, so we can say if we want to profile each individual branch in a BranchBlock
def profileModel(model:nn.Sequential, input_size:tuple, printOriginalTable=False):
    
    """
    Prints and plots relevant model information to get a sense of model size and expected performance.
    
    Arguments:
        model: A Sequential representation of a model
        input_size: The shape of the expected input in the form (B, C, W, H)
        printoriginalTable: Whether or not to print the original tables from the profiling library
        
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    randomInput = torch.randn(input_size, device=device, dtype=torch.float)
    model = model.to(device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True, with_modules=True, record_shapes=True, profile_memory=True) as profilerContext:
        
        output = randomInput
        for i, layer in enumerate(model.children()):
            # Get the class name for each layer and add an "_" so we can filter it
            with record_function(f"_{type(layer).__name__} {i}"):
                output = layer(output)

    originalEvents = profilerContext.key_averages()

    # Filter events with a custom name
    events = [event for event in originalEvents if event.key.startswith('_')]
    attributesList = []

    assert len(list(model.children())) == len(events)

    # Manually extract parameters from each event
    for event, layer in zip(events, model.children()):
        cpu_time = event.cpu_time_total
        cuda_time = event.cuda_time_total
        cuda_memory_usage = round(event.cuda_memory_usage / (1024**2), 3)
        param_count = sum([p.numel() for p in layer.parameters()])
        key = event.key
        attributesList.append({
            'key': key,
            'cpu_time': cpu_time,
            'cuda_time': cuda_time,
            'cuda_memory_usage': cuda_memory_usage,
            'param_count': param_count
        })
        
    # Creates a 2D list in the same shape as a table
    # table = [[val for _, val in attrs.items() if val != 'key'] for attrs in attributesList]
    table = [[val for _, val in attrs.items()] for attrs in attributesList]
    
    nRows = len(table)
    nCols = len(table[0])
    
    columnTotals = [None, ]
    # Start from 1 since we want to skip the names of the layers
    for cIdx in range(1, nCols):
        currentTotal = 0
        for rIdx in range(nRows):
            currentTotal += table[rIdx][cIdx]
        columnTotals.append(currentTotal)
    
    table.append(columnTotals)
    
    print(tabulate(table, headers=['CPU Time', 'CUDA Time (ms)', 'CUDA Memory Usage (MB)', 'Parameter Count'], tablefmt='outline'))

    cudaTimes = [attr['cuda_time']/1000 for attr in attributesList]    
    cudaMemories = [attr['cuda_memory_usage'] for attr in attributesList]
    param_counts = [attr['param_count'] for attr in attributesList]

    layerCount = len(attributesList)

    plt.bar(range(layerCount), cudaTimes), plt.title('CUDA times'), plt.xlabel('Layer number'), plt.xticks(range(layerCount)), plt.ylabel('Layer time (ms)'), plt.show()
    plt.bar(range(layerCount), cudaMemories), plt.title('CUDA Memory Usage (MB)'), plt.xlabel('Layer number'), plt.xticks(range(layerCount)), plt.ylabel('Memory usage (MB)'), plt.show()
    plt.bar(range(layerCount), param_counts), plt.title('Parameter Counts'), plt.xlabel('Layer number'), plt.xticks(range(layerCount)), plt.ylabel('Parameters'), plt.show()

    if printOriginalTable:
        outputTable = originalEvents.table()
        print(outputTable)