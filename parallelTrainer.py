import subprocess
import threading
import multiprocessing
import time
import queue
import os
import numpy as np
from utils import getModel, getModelSummary, getEstimatedModelSize
from models import *
from itertools import chain
from trainingParameters import *


def getLoggingFile(logDir="logs"):
    # Ensure the logs directory exists
    os.makedirs(logDir, exist_ok=True)

    # Find the next available log file number
    logNumber = 0
    while os.path.exists(os.path.join(logDir, f"log_{logNumber}.txt")):
        logNumber += 1

    # Return the log file path
    return os.path.join(logDir, f"log_{logNumber}.txt")

def logOutput(logPath, string):
    # Append a string to the log file
    with open(logPath, "a") as file:
        file.write(string)


def dictToArgs(dict):
    return ' '.join([f'--{key} {value}' for key, value in dict.items()])

def enqueueOutput(out, queue):
    try:
        for line in iter(out.readline, ''):
            queue.put(line)
    finally:
        out.close()

def handleOutput(process, logFile):
    outQueue = queue.Queue()
    outThread = threading.Thread(target=enqueueOutput, args=(process.stdout, outQueue))
    errThread = threading.Thread(target=enqueueOutput, args=(process.stderr, outQueue))

    outThread.daemon = True
    outThread.start()

    errThread.daemon = True
    errThread.start()

    # Use the provided log file path instead of getting a new one
    while True:
        try:
            line = outQueue.get(timeout=.1)
        except queue.Empty:
            if process.poll() is not None: # If process has not terminated, break
                break
        else:
            print(line, end='')
            logOutput(logFile, line)


def getTotalVRAM():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'], capture_output=True, text=True)
        totalVRAM = int(result.stdout.strip())
        return totalVRAM
    except Exception as e:
        print(f"Error retrieving total VRAM: {e}")
        return None


def isVRAMAvailable(neededMemory:float, modelArguments:dict) -> bool:
    global totalAllocatedVRAM
    global TOTAL_VRAM
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], capture_output=True, text=True)
        availableMemory = int(result.stdout.strip().split('\n')[0]) # This returns MB, not B
        
        neededMemoryMB = round(neededMemory, 3)
        totalNeededMemoryMB = neededMemoryMB + totalAllocatedVRAM

        vramAvailable = availableMemory >= totalNeededMemoryMB
        
        if not vramAvailable:
            # The available VRAM calculation is wrong, but it still works
            print(f'Current {modelArguments["modelName"]} in queue requires {neededMemoryMB} MB but only {TOTAL_VRAM - round(totalAllocatedVRAM, 3)} MB is available, sleeping...')
        else:
            print(f'Current model {modelArguments["modelName"]} only needs {neededMemoryMB} MB of the available {availableMemory - round(totalAllocatedVRAM, 3)} MB and will be trained')
            totalAllocatedVRAM += neededMemoryMB
        
        return vramAvailable
    except Exception as e:
        print(f"Error checking VRAM: {e}")
        return False


def canInsertModel(memoryParameterList:list) -> bool:
    global totalAllocatedVRAM
    global TOTAL_VRAM
    try:
        availableMemory = getAvailableVRAM()
        
        neededMemoryMBArray = np.array([p[0] for p in memoryParameterList])
        totalNeededMemoryMB = neededMemoryMBArray + totalAllocatedVRAM

        vramAvailable = availableMemory >= totalNeededMemoryMB
        
        if not vramAvailable:
            # The available VRAM calculation is wrong, but it still works
            print(f'Not enough room in queue, sleeping...')
        else:
            print(f'Found model to train!')
        
        return vramAvailable
    except Exception as e:
        print(f"Error checking VRAM: {e}")
        return False


def monitorProcess(process, modelArguments, realMemoryUsage):
    global totalAllocatedVRAM
    # Wait for the process to complete
    process.wait()
    
    # Once the process completes, update totalAllocatedVRAM
    modelName = modelArguments['modelName']
    vramReleased = realMemoryUsage
    totalAllocatedVRAM -= vramReleased
    print(f"Model {modelName} completed, released {vramReleased} MB of VRAM.")

def getAvailableVRAM() -> float:
    
    """
    Returns the available VRAM that can be used for training in MB
    """
    
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], capture_output=True, text=True)
    availableMemory = int(result.stdout.strip().split('\n')[0]) # This returns MB, not B
    return availableMemory

def getNextModelParameters(currentMemoryParameterList:list) -> tuple(float, dict):
    global totalAllocatedVRAM
    global TOTAL_VRAM
    
    availableMemory = getAvailableVRAM()
    
    memoryReqs = np.array([p[0] for p in currentMemoryParameterList])
    totalNeededMemoryMBArray = memoryReqs + totalAllocatedVRAM

    # Get the remaining memory we would have if we inserted each model
    remainingMemoryCapacity = availableMemory - totalNeededMemoryMBArray
    
    # validModelIndices are all indices that we could use without going over the memory cap
    validModelIndices = remainingMemoryCapacity >= 0
    # This gets the best index which represents the model that leaves the least memory capacity if used. 
    bestModelIdx = np.where(validModelIndices)[0][np.argmin(remainingMemoryCapacity[validModelIndices])]

    memoryReq, modelParams = currentMemoryParameterList[bestModelIdx]
    
    totalAllocatedVRAM += memoryReq

    return memoryReq, modelParams


# Global variable to track allocated VRAM
totalAllocatedVRAM = 0
TOTAL_VRAM = getTotalVRAM()
modelsInTraining = 0
tryToTrainNextModel = False


def main():
    global totalAllocatedVRAM

    MODEL_BATCHES = [
        # BIG_MODEL_BATCH_1_EASYAUGMENT,
        # BIG_MODEL_BATCH_2_EASYAUGMENT
        # BIG_MODEL_BATCH_7_EASYAUGMENT,
        # BIG_MODEL_BATCH_8_EASYAUGMENT
        # BIG_MODEL_BATCH_9_EASYAUGMENT,
        # BIG_MODEL_BATCH_10_EASYAUGMENT,
        # BIG_MODEL_BATCH_11_EASYAUGMENT,
        # BIG_MODEL_BATCH_12_EASYAUGMENT,
        # BIG_MODEL_BATCH_13_EASYAUGMENT,
        # BIG_MODEL_BATCH_14_EASYAUGMENT,
        # ALLEN_NET_BATCH_1_EASYAUGMENT, # Too deep, doesn't train well
        # ALLEN_NET_BATCH_2_EASYAUGMENT, # Too deep, doesn't train well
        # ALLEN_NET_BATCH_3_EASYAUGMENT, # Too deep, doesn't train well
        # ALLEN_NET_LITE_BATCH_1_EASYAUGMENT,
        # ALLEN_NET_LITE_BATCH_2_EASYAUGMENT,
        # ALLEN_NET_LITE_BATCH_3_EASYAUGMENT
        # ALLEN_NET_LITE_BATCH_4_EASYAUGMENT
        PARAMETER_SWEEP_TEST_BATCH_1,
        PARAMETER_SWEEP_TEST_BATCH_2
        ]

    # 0 for False, 1 for True
    SAVE_RESULTS = 0
    
    logFile = getLoggingFile()  # Get the log file path once at the start
    
    
    fullParameterList = list(chain.from_iterable(MODEL_BATCHES))
    
    modelMemoryRequirements = []
    
    for modelParam in fullParameterList:
        
        modelName = modelParam['modelName']
        batch_size = modelParam['batch_size']
        
        retrievedModel = getModel(modelName)
        modelSummary = getModelSummary(retrievedModel, batch_size=batch_size)
        # It actually seems faster to train more models slower than a few models quickly. 
        totalMemoryRequirement = getEstimatedModelSize(modelSummary)*0.75 # Assume models will take up 50% more memory so we guarantee that everything will be done on the GPU
        
        modelMemoryRequirements.append(totalMemoryRequirement)
     
    # This is a list of format [(memRequired, modelParams)] sorted by memRequired in descending order
    memoryParameterList = [(y, x) for (y, x) in sorted(zip(modelMemoryRequirements, fullParameterList), reverse=True)]
    
    START_TIME = time.time()
    processes = []
    baseCommand = r'C:\Users\Nicholas\anaconda3\envs\CS444Env\python.exe trainModel.py'
    
    
    # Get the starting model to begin the process
    memoryReq, arguments = getNextModelParameters(memoryParameterList)
    
    for iter in range(len(memoryParameterList)):
        
        command = f"{baseCommand} {dictToArgs(arguments)} --saveResults {SAVE_RESULTS}"
        
        modelName = arguments['modelName']
        batch_size = arguments['batch_size']
        
        # Check if enough VRAM is available before starting training
        while not canInsertModel(memoryParameterList):
            time.sleep(10)  # Wait for N seconds before checking again
    
        memoryReq, arguments = getNextModelParameters(memoryParameterList)
        print(f'Found model:\n{arguments}\n Required memory: {memoryReq} MB')

        # Start a trainModel instance
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
        processes.append(process)

        # Pass the same log file path to handle_output
        outputThread = threading.Thread(target=handleOutput, args=(process, logFile))
        outputThread.start()

        # Create a new thread to monitor the process and free allocated memory
        monitorThread = threading.Thread(target=monitorProcess, args=(process, arguments, totalMemoryRequirement))
        monitorThread.start()

        time.sleep(5) # Wait for memory values to update
        
        pass
    
    
    
    
    
    
    
    #############################################################################################################################################################################
    
    for argumentsList in MODEL_BATCHES:

        START_TIME = time.time()
        
        processes = []

        baseCommand = r'C:\Users\Nicholas\anaconda3\envs\CS444Env\python.exe trainModel.py'

        for arguments in argumentsList:
            command = f"{baseCommand} {dictToArgs(arguments)} --saveResults {SAVE_RESULTS}"
            
            modelName = arguments['modelName']
            batch_size = arguments['batch_size']
            
            retrievedModel = getModel(modelName)
            modelSummary = getModelSummary(retrievedModel, batch_size=batch_size)
            # It actually seems faster to train more models slower than a few models quickly. 
            totalMemoryRequirement = getEstimatedModelSize(modelSummary)*0.75 # Assume models will take up 50% more memory so we guarantee that everything will be done on the GPU
            
            # Check if enough VRAM is available before starting training
            while not isVRAMAvailable(totalMemoryRequirement, arguments):
                time.sleep(10)  # Wait for N seconds before checking again
            
            # Start a trainModel instance
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
            processes.append(process)

            # Pass the same log file path to handle_output
            outputThread = threading.Thread(target=handleOutput, args=(process, logFile))
            outputThread.start()

            # Create a new thread to monitor the process and free allocated memory
            monitorThread = threading.Thread(target=monitorProcess, args=(process, arguments, totalMemoryRequirement))
            monitorThread.start()

            time.sleep(5) # Wait for memory values to update


        # Wait for all processes to complete
        for p in processes:
            p.wait()
            
        print(f'Total Runtime: {time.time() - START_TIME}')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn') # This tells python to treat this whole thing as an independent script for multiprocessing
    main()


