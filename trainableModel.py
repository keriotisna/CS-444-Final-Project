import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from blocks import *
import torchvision
import numpy as np

import os

from utils import TransformableSubset, validateModelIO

from tensorboardX import SummaryWriter
from tqdm import tqdm # Progress bar for training


class TrainingParameters():
    
    def __init__(self, fullDataset:Dataset, trainTransform:torchvision.transforms.Compose, valTestTransform:torchvision.transforms.Compose,
                 trainValTestSplit:list, epochs:int, warmupEpochs:int,
                 batch_size:int, lr:float, momentum:float, weight_decay=0.01, nesterov=True, plateuPatience=3, plateuFactor=0.5) -> None:
        
        assert len(trainValTestSplit) == 3
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
        
        self.trainDataset = TransformableSubset(trainDataset, fullDataset, transform=trainTransform)
        self.validationDataset = TransformableSubset(validationDataset, fullDataset, transform=valTestTransform)
        self.testDataset = TransformableSubset(testDataset, fullDataset, transform=valTestTransform)

        self.epochs = epochs
        self.warmupEpochs = warmupEpochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.plateuPatience = plateuPatience
        self.plateuFactor = plateuFactor
        



class TrainableModel():
    
    def __init__(self, modelName:str, model:nn.Module, 
                 trainingParameters:TrainingParameters, dataLoaderParameters=None) -> None:
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        assert self.device == 'cuda', f'Device is not cuda: {self.device}'
        
        self.modelName = modelName
        self.model = model
        
        self.trainingParameters = trainingParameters
        
        if dataLoaderParameters is None:
            dataLoaderParameters = {
                'num_workers': 2,
                'pin_memory': True,
                'prefetch_factor': 4
            }
        else:
            dataLoaderParameters = dataLoaderParameters
        
        self.trainLoader = DataLoader(trainingParameters.trainDataset, batch_size=trainingParameters.batch_size, shuffle=True, **dataLoaderParameters)
        self.validationLoader = DataLoader(trainingParameters.validationDataset, batch_size=trainingParameters.batch_size, shuffle=True, **dataLoaderParameters)
        self.testLoader = DataLoader(trainingParameters.testDataset, batch_size=trainingParameters.batch_size, shuffle=True, **dataLoaderParameters)

    
    def getSaveFileName(self, duplicateID=''):
        
        epochs = self.trainingParameters.epochs
        batch_size = self.trainingParameters.batch_size
        lr = self.trainingParameters.lr
        momentum = self.trainingParameters.momentum
        
        modelName = self.modelName
        
        if duplicateID != '':
            duplicateID = '-' + str(duplicateID)
        
        filename = f'{modelName}{duplicateID}_Epoch{epochs}_Batch{batch_size}_LR{lr}_Momentum{momentum}'
        
        return filename
    
    
    def trainEpoch(self, dataloader:DataLoader, optimizer:torch.optim.Optimizer, freezeModel=False) -> tuple[float, float]:
    
        """
        Trains self.model for a single epoch when given a dataloader and optimizer.
        
        Arguments:
            dataloader: A dataloader for either a training, validation, or test dataset
            optimizer: The current optimizer being used during training
            freezeModel: Whether or not to freeze the model for the current epoch. We want to freeze the model when running 
                inference on validation or test datasets.

        Returns:
            (modelLoss, modelAccuracy):
                modelLoss: The average model loss across the given dataset
                modelAccuracy: The average model accuracy
        """

        
        lossFunction = nn.CrossEntropyLoss()
        
        totalLoss = 0
        
        N = 0
        correct = 0
        
        model = self.model
        device = self.device
        
        for features, labels in dataloader:
            x, y = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if not freezeModel:
                optimizer.zero_grad()
                
            forwardPass = model.forward(x)
            
            # This adds the current accuracy to correct which is averaged over all iterations of the epoch.
            correct += (forwardPass.argmax(dim=1) == y).float().mean().item()

            loss = lossFunction(forwardPass, y)
            totalLoss += loss.item()
            
            if not freezeModel:
                # Do gradient clipping to ensure nothing crazy is happening # TODO: Analyze gradients with and without for report
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=5.0) # Clip gradients after calculating loss
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
            
            N += 1
        
        return totalLoss / N, correct / N
    
    
    def train(self, PID='', startingEpochs=0, SAVE_RESULTS=True):
    
        # torch.cuda.empty_cache()

        RUNS_DIR = 'runs'
        MODELS_DIR = 'models'

        epochs = self.trainingParameters.epochs
        warmupEpochs = self.trainingParameters.warmupEpochs

        lr = self.trainingParameters.lr
        momentum = self.trainingParameters.momentum
        weight_decay = self.trainingParameters.weight_decay
        nesterov = self.trainingParameters.nesterov
        factor = self.trainingParameters.plateuFactor
        patience = self.trainingParameters.plateuPatience
        
        model = ResidualCNN(network=self.model).to(self.device)
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-9, end_factor=1, total_iters=warmupEpochs)
        plateuScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)

        trainLoader = self.trainLoader
        validationLoader = self.validationLoader

        MODEL_PARAM_COUNT = sum(p.numel() for p in model.parameters() if p.requires_grad)


        writerPath = f'{RUNS_DIR}/{self.getSaveFileName()}'
        modelPath = f'{MODELS_DIR}/{self.getSaveFileName()}'
        
        
        duplicateID = 0
        while os.path.exists(writerPath):
            duplicateID += 1
            writerPath = f'{RUNS_DIR}/{self.getSaveFileName(duplicateID=duplicateID)}'
            modelPath = f'{MODELS_DIR}/{self.getSaveFileName(duplicateID=duplicateID)}'


        if SAVE_RESULTS:
            writer = SummaryWriter(writerPath, flush_secs=10)

        best_val_acc = 0

        pbar = tqdm(range(startingEpochs, epochs+startingEpochs))
        for epoch in pbar:
            trainLoss, trainAccuracy = self.trainEpoch(dataloader=trainLoader, optimizer=optimizer)
            validationLoss, validationAccuracy = self.trainEpoch(dataloader=validationLoader, optimizer=optimizer, freezeModel=True)

            if epoch < warmupEpochs:
                currentLr = warmup.get_last_lr()[0]
            else:
                currentLr = plateuScheduler.optimizer.param_groups[0]['lr']  # Directly get LR from optimizer used in plateuScheduler

            if validationAccuracy > best_val_acc and SAVE_RESULTS:
                best_val_acc = validationAccuracy
                bestEpoch = epoch
                torch.save(self.model.state_dict(), modelPath)

            # Write values to progress bar and save for tensorboardX
            pbar.set_description("lr: {:.6f}, trainLoss: {:.4f}, trainAccuracy: {:.4f}, validationLoss: {:.4f}, validationAccuracy: {:.4f}".format(currentLr, trainLoss, trainAccuracy, validationLoss, validationAccuracy), refresh=False)
            print(f'{PID} {pbar}', flush=True)
            
            
            if SAVE_RESULTS:
                writer.add_scalar('trainLoss', trainLoss, epoch)
                writer.add_scalar('trainAccuracy', trainAccuracy, epoch)
                
                writer.add_scalar('validationLoss', validationLoss, epoch)
                writer.add_scalar('validationAccuracy', validationAccuracy, epoch)
                
                writer.add_scalar('lr', currentLr, epoch)

            if epoch < warmupEpochs:
                warmup.step()
            else:
                plateuScheduler.step(validationLoss)
                
            if currentLr < 1e-7 and epoch >= warmupEpochs:
                print(f'Learning rate collapsed, ending training at epoch {epoch}')
                break
            
