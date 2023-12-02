import subprocess
import threading
import multiprocessing
import time
import queue
import os

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

def handle_output(process):
    outQueue = queue.Queue()
    outThread = threading.Thread(target=enqueueOutput, args=(process.stdout, outQueue))
    errThread = threading.Thread(target=enqueueOutput, args=(process.stderr, outQueue))

    # Daemon thread means once this process dies, the daemons will die with it so we won't have resource leask
    outThread.daemon = True
    outThread.start()

    errThread.daemon = True
    errThread.start()

    logFile = getLoggingFile()

    # Print child output to parent output
    while True:
        try:
            line = outQueue.get(timeout=.1)
        except queue.Empty:
            if process.poll() is not None: # If process has not terminated, break
                break
        else: # got line
            print(line, end='')
            logOutput(logFile, line)


def main():


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
        # ALLEN_NET_BATCH_2_EASYAUGMENT,
        # ALLEN_NET_BATCH_3_EASYAUGMENT
        ALLEN_NET_LITE_BATCH_1_EASYAUGMENT,
        ALLEN_NET_LITE_BATCH_2_EASYAUGMENT,
        ALLEN_NET_LITE_BATCH_3_EASYAUGMENT
        ]

    # 0 for False, 1 for True
    SAVE_RESULTS = 1

    for modelArgBatch in MODEL_BATCHES:

        START_TIME = time.time()

        argumentsList = modelArgBatch
        
        processes = []

        baseCommand = r'C:\Users\Nicholas\anaconda3\envs\CS444Env\python.exe trainModel.py'

        for arguments in argumentsList:
            command = f"{baseCommand} {dictToArgs(arguments)} --saveResults {SAVE_RESULTS}"
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
            processes.append(process)

            # Create a new thread to handle the output of each subprocess
            outputThread = threading.Thread(target=handle_output, args=(process,))
            outputThread.start()

        # Wait for all processes to complete
        for p in processes:
            p.wait()

        print(f'Total Runtime: {time.time() - START_TIME}')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn') # This tells python to treat this whole thing as an independent script for multiprocessing
    main()


