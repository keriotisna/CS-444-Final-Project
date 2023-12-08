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


def getLoggingFile(logDir="logs") -> str:
    
    """
    Gets the next numerically available logging file name from the logs directory
    
    Arguments:
        logDir: The directory to look for the logging file
    """
    
    # Ensure the logs directory exists
    os.makedirs(logDir, exist_ok=True)

    # Find the next available log file number
    logNumber = 0
    while os.path.exists(os.path.join(logDir, f"log_{logNumber}.txt")):
        logNumber += 1

    # Return the log file path
    return os.path.join(logDir, f"log_{logNumber}.txt")

def logOutput(logPath:str, string:str):
    
    """
    Writes a string to the given logfile
    
    Arguments:
        logPath: The path the new string should be added to
        string: The string to be written
    """
    
    # Append a string to the log file
    with open(logPath, "a") as file:
        file.write(string)


def dictToArgs(dictionary:dict) -> str:
    
    """
    Converts a dictionary to a command line argument based on keys and values. {'batch_size': 20} becomes --batch_size 20
    
    Arguments:
        dictionary: The dictionary to convert to command line args
        
    Returns:
        arguments: The command line arguments of the dictionary.
    """
    
    return ' '.join([f'--{key} {value}' for key, value in dictionary.items()])

def enqueueOutput(out, queue: queue.Queue):
    
    """
    Places output from a process into the logging queue.
    
    Arguments:
        out: The output from a subprocess
        queue: The queue the output should be placed
    """
    
    try:
        for line in iter(out.readline, ''):
            queue.put(line)
    finally:
        out.close()

def handleOutput(process, logFile:str):
    
    """
    A subprocess function that handles outputs from a subprocess and enqueues them in a logging queue to be logged
    
    Arguments:
        process: The subprocess the output should be captured from
        logFile: A log file path where the output will be written.
    """
    
    global loggingQueue
    outThread = threading.Thread(target=enqueueOutput, args=(process.stdout, loggingQueue))
    errThread = threading.Thread(target=enqueueOutput, args=(process.stderr, loggingQueue))

    # Setting the thread as a daemon means it will die with the parent process, so no resources will be wasted.
    outThread.daemon = True
    outThread.start()

    errThread.daemon = True
    errThread.start()

    # Use the provided log file path instead of getting a new one
    while True:
        try:
            line = loggingQueue.get(timeout=.1)
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


def canInsertModel(memoryParameterList:list) -> bool:
    global totalAllocatedVRAM, TOTAL_VRAM, modelsInTraining, MAX_CONCURRENT_MODELS

    try:
        availableMemory = TOTAL_VRAM - totalAllocatedVRAM
        
        neededMemoryMBArray = np.array([p[0] for p in memoryParameterList])

        vramAvailable = availableMemory >= neededMemoryMBArray
        
        if modelsInTraining >= MAX_CONCURRENT_MODELS:
            printq(f'There are currently {modelsInTraining} training which is equal to or more than the allowed {MAX_CONCURRENT_MODELS}, sleeping...')
            return False
        
        if not np.any(vramAvailable):
            printq(f'Possible model memory requirements: {neededMemoryMBArray}')
            printq(f'Not enough room in queue with {round(TOTAL_VRAM - totalAllocatedVRAM, 3)} MB available, sleeping...')
            return False
        else:
            printq(f'Found model to train!')
            return True
        
    except Exception as e:
        printq(f"Error checking VRAM: {e}")
        return False


def monitorProcess(process:subprocess.Popen, modelArguments:dict, realMemoryUsage:float):
    
    """
    A thread function to monitor the training of an individual model. Mainly used to release allocated VRAM once training completes.
    
    Arguments:
        process: An associated process this function should be tied to.
        modelArguments: The dictionary containing arguments associated with the current model process.
        realMemoryUsage: How much estimated memory this model will use during training
    """
    
    global totalAllocatedVRAM, modelsInTraining

    # Wait for the process to complete
    process.wait()
    
    # Once the process completes, update totalAllocatedVRAM
    modelName = modelArguments['modelName']
    vramReleased = realMemoryUsage
    totalAllocatedVRAM -= vramReleased
    modelsInTraining -= 1
    print(f"Model {modelName} completed, released {round(vramReleased, 3)} MB of VRAM.")


def getNextModelParameters(currentMemoryParameterList:list) -> tuple[float, dict]:
    
    """
    Given a list of all models that need to be trained, returns the model that will best use the available resources using a bin-packing descending approach
    
    Arguments:
        currentMemoryParametersList: A list of tuples in the form (memReq, params) where memReq is the memory needed for the associated parameters.
        
    Returns:
        (memReq, params)
        memReq: A float which refers to how much memory the associated model parameters will utilize during training.
        params: A dict which refers to the parameters of the model to be trained
    """
    
    global totalAllocatedVRAM
    global TOTAL_VRAM
    
    availableMemory = TOTAL_VRAM - totalAllocatedVRAM
    
    memoryReqs = np.array([p[0] for p in currentMemoryParameterList])

    validModelIndices = memoryReqs <= availableMemory
    bestModelIdx = np.where(validModelIndices)[0][np.argmax(memoryReqs[validModelIndices])]

    memoryReq, modelParams = currentMemoryParameterList.pop(bestModelIdx)
    
    totalAllocatedVRAM += memoryReq
    printq(f'Current allocation for VRAM is {round(totalAllocatedVRAM, 3)} MB')

    return memoryReq, modelParams


def printq(stringToLog:str):
    
    """
    Inputs a string to the logging queue and prints it in console.
    """

    global loggingQueue
    loggingQueue.put(stringToLog+'\n')    
    

# A queue for logging subprocess and parent process outputs
loggingQueue = queue.Queue()


# Global variables to track allocated and total VRAM
totalAllocatedVRAM = 0
TOTAL_VRAM = getTotalVRAM()
modelsInTraining = 0

# 0 for False, 1 for True
SAVE_RESULTS = 1

# How much to multiply model memory estimates by, larger values will guarantee that all model training happens in dedicated VRAM
# while low values may allow for models to use memory in shared memory.
MODEL_MEMORY_MULTIPLIER = 1.5
# Sets how many models can be trained at once to prevent CPU bottlenecking, 5 seems to approach 100% CPU utilization, but this needs to change depending on model size
MAX_CONCURRENT_MODELS = 4


def main():
    global modelsInTraining, SAVE_RESULTS, MODEL_MEMORY_MULTIPLIER

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
        # PARAMETER_SWEEP_TEST_BATCH_1,
        # PARAMETER_SWEEP_TEST_BATCH_2
        # ALLEN_NET_LITE_BATCH_5_EASYAUGMENT,
        # WILSON_NET_BATCH_1_EASYAUGMENT
        # RESNET_18_BATCH_1_EASYAUGMENT
        # JESSE_NET_BATCH_1_EASYAUGMENT,
        # JESSE_NET_BATCH_2_EASYAUGMENT,
        # WILSON_NET_FT_BATCH_1_EASYAUGMENT
        # JESSE_NET_BATCH_3_EASYAUGMENT
        # JESSE_NET_BATCH_4_EASYAUGMENT,
        # BASELINE_BATCH_2
        BASELINE_BATCH_4,
        BASELINE_BATCH_5,
        WILSON_NET_BATCH_1_HARDAUGMENT3,
        JESSE_NET_BATCH_4_HARDAUGMENT3
        ]
    
    # Get the log file path once at the start
    logFile = getLoggingFile()
    
    fullParameterList = list(chain.from_iterable(MODEL_BATCHES))
    modelMemoryRequirements = []
    
    for modelParam in fullParameterList:
        
        modelName = modelParam['modelName']
        batch_size = modelParam['batch_size']
        
        retrievedModel = getModel(modelName)
        modelSummary = getModelSummary(retrievedModel, batch_size=batch_size)
        
        # Scale true estimated model size to get memory estimates for each model
        totalMemoryRequirement = getEstimatedModelSize(modelSummary) * MODEL_MEMORY_MULTIPLIER
        modelMemoryRequirements.append(totalMemoryRequirement)
     
    pass
    # This is a list of format [(memRequired, modelParams)] sorted by memRequired in descending order for memory packing
    memoryParameterList = [(y, x) for (y, x) in sorted(zip(modelMemoryRequirements, fullParameterList), key=lambda pair: pair[0], reverse=True)]
    
    START_TIME = time.time()
    processes = []
    baseCommand = r'C:\Users\Nicholas\anaconda3\envs\CS444Env\python.exe trainModel.py'
    
    numModels = len(memoryParameterList)
    for iter in range(numModels):
        
        # Check if enough VRAM is available before starting training
        while not canInsertModel(memoryParameterList):
            time.sleep(300)  # Wait for N seconds before checking again
    
        memoryReq, arguments = getNextModelParameters(memoryParameterList)
        printq(f'Found model:\n{arguments}\n Required memory: {memoryReq} MB')

        command = f"{baseCommand} {dictToArgs(arguments)} --saveResults {SAVE_RESULTS}"

        # Start a trainModel instance
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
        processes.append(process)

        # Pass the same log file path to handle_output
        outputThread = threading.Thread(target=handleOutput, args=(process, logFile))
        outputThread.start()

        # Create a new thread to monitor the process and perform actions on the process's termination
        monitorThread = threading.Thread(target=monitorProcess, args=(process, arguments, memoryReq))
        monitorThread.start()


        modelsInTraining += 1
        printq(f'Currently, there are {modelsInTraining} models in training')
        time.sleep(5) # Wait for memory values to update
        
    # Wait for all processes to complete before getting totalRuntime
    for p in processes:
        p.wait()
        
    printq(f'Total Runtime: {time.time() - START_TIME}')    



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn') # This tells python to treat this whole thing as an independent script for multiprocessing
    main()


