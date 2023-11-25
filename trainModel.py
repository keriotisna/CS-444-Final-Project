import argparse
from blocks import *
import torch
import torchvision as tv
import torchvision.transforms.v2 as v2
from utils import validateModelIO, getNormalizedTransforms

from trainableModel import TrainingParameters, TrainableModel

from dataLoading import CIFAR10Dataset

# Import model and transform variables
from models import *
from transforms import *

import numpy as np

import multiprocessing



def getModel(modelName:str) -> nn.Sequential:
    
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
    actualName = modelName.split('_')[0]
    retrievedModel = globalVars[actualName]
    return retrievedModel

    

    
    
def getTrainTransform(trainTransformID:str) -> v2.Compose:
    
    """
    Gets the training transform to be used based on the command line argument trainTransformID
    
    Arguments:
        trainTransformID: The name of the training transform to be used. 

    Returns:
        transform: Returns the training transform to be used for the training dataset.
    """
    
    globalVars = globals()
    return globalVars[trainTransformID]
        
    

def getValTestTransform(valTestTransformID):
    
    """
    Gets the validation and test transform to be used based on the command line argument valTestTransformID
    
    Arguments:
        valTestTransformID: The name of the training transform to be used. 

    Returns:
        transform: Returns the training transform to be used for the validation and test dataset.
    """
    
    globalVars = globals()
    return globalVars[valTestTransformID]



def main():

    parser = argparse.ArgumentParser(description='Train a specified model')

    parser.add_argument('--modelName', type=str, help='Name of the model for file saving')
    # parser.add_argument('--modelID', type=int, help='ID to select correct model in trainModel.py')
    parser.add_argument('--trainTransformID', type=str, help='ID of the transform to use on training set. Will be normalized in trainModel.py')
    parser.add_argument('--valTestTransformID', type=str, help='ID of the transform to use on validation and test sets')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train model')
    parser.add_argument('--warmupEpochs', type=int, help='Number of warmup epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size for training loop')
    parser.add_argument('--lr', type=float, help='Initial learning rate for model')
    parser.add_argument('--momentum', type=float, help='Momentum of model SGD optimizer')
    parser.add_argument('--weight_decay', type=float, help='Weight decay of SGD optimizer')
    parser.add_argument('--nesterov', type=bool, help='Whether or not to use the Nesterov accelerated SGD')
    parser.add_argument('--plateuPatience', type=int, help='How many epochs without improvement before lr is decayed by plateuFactor')
    parser.add_argument('--plateuFactor', type=float, help='How much lr is decayed after no loss improvements')
    parser.add_argument('--saveResults', type=int, help='Whether or not to write Tensorboard events or save models')


    # Initialize a process ID to identify which subprocesses correspond to which output
    # Yes, PIDs may not be unique. No, I don't care
    PID = str(np.random.randint(0, 100000)).zfill(5)

    args = parser.parse_args()

    modelName = args.modelName
    trainTransformID = args.trainTransformID
    valTestTransformID = args.valTestTransformID
    epochs = args.epochs
    warmupEpochs = args.warmupEpochs
    batch_size = args.batch_size
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    nesterov = args.nesterov
    plateuPatience = args.plateuPatience
    plateuFactor = args.plateuFactor
    SAVE_RESULTS = (1 == args.saveResults)
    
    print(f'{PID} {args}', flush=True)
    print(f'SAVE_RESULTS: {SAVE_RESULTS}')

    # Default arguments for simple testing
    # modelName = 'highwaynetv5'
    # trainTransformID = 'default'
    # valTestTransformID = 'NONE'
    # epochs = 50
    # warmupEpochs = 5
    # batch_size = 2048
    # lr = 1e-2
    # momentum = 0.8
    # weight_decay = 0.01
    # nesterov = True
    # plateuPatience = 3
    # plateuFactor = 0.5
    # SAVE_RESULTS = False


    model = getModel(modelName)
    trainTransform = getTrainTransform(trainTransformID)
    valTestTransform = getValTestTransform(valTestTransformID)
    


    try:
        validateModelIO(ResidualCNN(network=model, printOutsize=False), printSummary=False)
    except Exception as e:
        print(f'PID {PID}: {modelName} failed validation!')
        raise e
    
    # Create dataset instances
    fullDataset = CIFAR10Dataset(rootDirectory='cifar-10', csvFilename='trainLabels.csv', dataFolder='train', transform=None)

    print(f'{PID} Normalizing...', flush=True)
    normalizedTrainTransform, normalizedValTestTransform = getNormalizedTransforms(fullDataset=fullDataset, trainTransform=trainTransform, valTestTransform=valTestTransform, showSamples=False)
    print(f'{PID} Done normalizing!', flush=True)

    print(f'{PID} normalizedTrainTransform: {normalizedTrainTransform}')
    print(f'{PID} normalizedValTestTransform: {normalizedValTestTransform}')

    # Create dataset instances
    fullDataset = CIFAR10Dataset(rootDirectory='cifar-10', csvFilename='trainLabels.csv', dataFolder='train', transform=None)

    modelParams = TrainingParameters(fullDataset=fullDataset, trainTransform=normalizedTrainTransform, valTestTransform=normalizedValTestTransform, 
                                    trainValTestSplit=[0.8, 0.1, 0.1], epochs=epochs, warmupEpochs=warmupEpochs, batch_size=batch_size,
                                    lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, plateuPatience=plateuPatience, plateuFactor=plateuFactor)

    print(f'{PID} Starting training...', flush=True)
    trainableModel = TrainableModel(modelName=modelName, model=model, trainingParameters=modelParams)
    trainableModel.train(PID=PID, SAVE_RESULTS=SAVE_RESULTS)
    print(f'{PID} Training complete!', flush=True)



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn') # This tells python to treat this whole thing as an independent script for multiprocessing
    main()
