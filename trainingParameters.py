from models import *
import numpy as np

def _getBatchParameterList(modelNames:list, nameSuffix='', 
        parametersDict:dict={
            'trainTransformID': 'default',
            'valTestTransformID': 'NONE',
            'epochs': 200,
            'warmupEpochs': 5,
            'batch_size': 2048,
            'lr': 5e-2,
            'momentum': 0.8,
            'weight_decay': 0.01,
            'nesterov': True,
            'plateuPatience': 3,
            'plateuFactor': 0.5
        }):
    
    """
    Gets a list of identical parameters for the given model names which can be used for batch training
    
    Arguments:
        modelNames: A list of model names that match variable names in models.py which correspond to a single sequential model
        nameSuffix: An optional suffix that will be added to saved model data for easy identification. This is handled in trainModel.py
        parametersDict: The set of parameters that will be used on each model in modelNames

    Returns:
        batchParametersList: A list of identical parameters with the modelName attribute set to the given model names.
    """
    
    batchParametersList = []
    
    for currentName in modelNames:
        
        workingDict = parametersDict.copy()
        
        suffix = ''
        # if len(nameSuffix) >= 0:
        suffix = '_' + nameSuffix
        
        workingDict['modelName'] = currentName + suffix
        batchParametersList.append(workingDict)
    
    return batchParametersList


def _getBatchParametersSweep(modelName:str, sweepParamName:str, sweepParamList:list, nameSuffix:str='',
        parametersDict:dict={
            'trainTransformID': 'default',
            'valTestTransformID': 'NONE',
            'epochs': 200,
            'warmupEpochs': 5,
            'batch_size': 2048,
            'lr': 5e-2,
            'momentum': 0.8,
            'weight_decay': 0.01,
            'nesterov': True,
            'plateuPatience': 3,
            'plateuFactor': 0.5
        }):
    
    """
    Returns a trainable set of model parameters that sweeps a certain parameter over a range of provided values. 
    This function should be used for hyperparameter tuning by testing multiple values of a single hyperparameter for a single model.
    
    Arguments:
        modelName: The name of the model
        sweepParamName: The hyperparameter value the sweepParamList should be associated with.
        sweepParamList: The list of values the respective hyperparameter should be evaluated at.
        nameSuffix: An optional suffix that goes after the model name for easier identification.
        
    Returns:
        batchParametersList: A list of different hyperparameters with identical models.
    """
    
    batchParametersList = []
        

        
    for paramValue in sweepParamList:
        
        workingDict = parametersDict.copy()
        
        suffix = '_' + nameSuffix
        
        workingDict[sweepParamName] = paramValue
        workingDict['modelName'] = modelName + suffix
        
        
        batchParametersList.append(workingDict)
    
    return batchParametersList


def _getBatchParametersSweep2(modelName:str, nameSuffix:str='',
        parametersDict:dict={
            'trainTransformID': 'default',
            'valTestTransformID': 'NONE',
            'epochs': 200,
            'warmupEpochs': 5,
            'batch_size': 2048,
            'lr': 5e-2,
            'momentum': 0.8,
            'weight_decay': 0.01,
            'nesterov': True,
            'plateuPatience': 3,
            'plateuFactor': 0.5
        }):
    
    """
    Returns a trainable set of model parameters that sweeps a certain parameter over a range of provided values. 
    This function should be used for hyperparameter tuning by testing multiple values of a single hyperparameter for a single model.
    
    Arguments:
        modelName: The name of the model
        sweepParamName: The hyperparameter value the sweepParamList should be associated with.
        sweepParamList: The list of values the respective hyperparameter should be evaluated at.
        nameSuffix: An optional suffix that goes after the model name for easier identification.
        
    Returns:
        batchParametersList: A list of different hyperparameters with identical models.
    """
    
    batchParametersList = []
        
    listCount = 0
    for key, value in parametersDict.items():
        if isinstance(value, list):
            listCount += 1
            iterableValueKey = key
            sweepParamList = value
            
    assert listCount == 1
    
    
        
    for paramValue in sweepParamList:
        
        workingDict = parametersDict.copy()
        
        suffix = '_' + nameSuffix
        
        workingDict[iterableValueKey] = paramValue
        workingDict['modelName'] = modelName + suffix
        
        
        batchParametersList.append(workingDict)
    
    return batchParametersList





PARAMETER_SWEEP_TEST_BATCH_1 = _getBatchParametersSweep(modelName='baseline430kN', sweepParamName='batch_size', sweepParamList=[8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    parametersDict={
        'trainTransformID': 'NONE',
        'valTestTransformID': 'NONE',
        'epochs': 5,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

PARAMETER_SWEEP_TEST_BATCH_2 = _getBatchParametersSweep(modelName='baseline130kN', sweepParamName='batch_size', sweepParamList=[8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    parametersDict={
        'trainTransformID': 'NONE',
        'valTestTransformID': 'NONE',
        'epochs': 5,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })


BASELINE_BATCH_1 = _getBatchParameterList(modelNames=['baseline130k', 'baseline130kN', 'baseline430k', 'baseline430kN'],
    nameSuffix='vanilla',
    parametersDict={
        'trainTransformID': 'NONE',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

BASELINE_BATCH_2 = _getBatchParameterList(modelNames=['baseline13MN', 'baseline36MN', 'baseline108MN'],
    nameSuffix='vanilla',
    parametersDict={
        'trainTransformID': 'NONE',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 512,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

BASELINE_BATCH_3 = _getBatchParameterList(modelNames=['residualNetv1', 'bottleneckResidualv1', 'highwayResidualv1'],
    nameSuffix='vanilla',
    parametersDict={
        'trainTransformID': 'NONE',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 256,
        'lr': 5e-2,
        'momentum': 0.9,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })



####################################################################################
# AUTOAUGMENT
####################################################################################

BASELINE_BATCH_1_AUTOAUGMENT = _getBatchParameterList(modelNames=['baseline130k', 'baseline130kN', 'baseline430k', 'baseline430kN'],
    nameSuffix='autoaugment',
    parametersDict={
        'trainTransformID': 'autoaugment',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

HIGHWAY_BATCH_1_AUTOAUGMENT = _getBatchParameterList(modelNames=['highwaynetv3', 'highwaynetv4', 'highwaynetv5'],
    nameSuffix='autoaugment',
    parametersDict={
        'trainTransformID': 'autoaugment',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

HIGHWAY_BATCH_2_AUTOAUGMENT = _getBatchParameterList(modelNames=['highwaynetv6', 'highwaynetv7', 'highwaynetv8'],
    nameSuffix='autoaugment',
    parametersDict={
        'trainTransformID': 'autoaugment',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

HIGHWAY_BATCH_3_AUTOAUGMENT = _getBatchParameterList(modelNames=['highwaynetv9', 'highwaynetv10', 'highwaynetv11'],
    nameSuffix='autoaugment',
    parametersDict={
        'trainTransformID': 'autoaugment',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

DOUBLE_BOTTLE_BATCH_1_AUTOAUGMENT = _getBatchParameterList(modelNames=['doubleBottlev3', 'doubleBottlev4', 'doubleBottlev5'],
    nameSuffix='autoaugment',
    parametersDict={
        'trainTransformID': 'autoaugment',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })



####################################################################################
# DEFAULT
####################################################################################


HIGHWAY_BATCH_1_DEFAULTAUGMENT = _getBatchParameterList(modelNames=['highwaynetv3', 'highwaynetv4', 'highwaynetv5'],
    nameSuffix='defaultaugment',
    parametersDict={
        'trainTransformID': 'default',
        'valTestTransformID': 'default',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

HIGHWAY_BATCH_2_DEFAULTAUGMENT = _getBatchParameterList(modelNames=['highwaynetv6', 'highwaynetv7', 'highwaynetv8'],
    nameSuffix='defaultaugment',
    parametersDict={
        'trainTransformID': 'default',
        'valTestTransformID': 'default',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })


DOUBLE_BOTTLE_BATCH_1_DEFAULTAUGMENT = _getBatchParameterList(modelNames=['doubleBottlev3', 'doubleBottlev4', 'doubleBottlev5'],
    nameSuffix='defaultaugment',
    parametersDict={
        'trainTransformID': 'default',
        'valTestTransformID': 'default',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })






####################################################################################
# EASYAUGMENT
####################################################################################


BASELINE_BATCH_1_EASYAUGMENT = _getBatchParameterList(modelNames=['baseline130k', 'baseline130kN', 'baseline430k', 'baseline430kN'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })


HIGHWAY_BATCH_1_EASYAUGMENT = _getBatchParameterList(modelNames=['highwaynetv3', 'highwaynetv4', 'highwaynetv5'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

HIGHWAY_BATCH_2_EASYAUGMENT = _getBatchParameterList(modelNames=['highwaynetv6', 'highwaynetv7', 'highwaynetv8'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

HIGHWAY_BATCH_3_EASYAUGMENT = _getBatchParameterList(modelNames=['highwaynetv9', 'highwaynetv10', 'highwaynetv11'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

DOUBLE_BOTTLE_BATCH_1_EASYAUGMENT = _getBatchParameterList(modelNames=['doubleBottlev3', 'doubleBottlev4', 'doubleBottlev5'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })


ALLEN_NET_BATCH_1_EASYAUGMENT = _getBatchParameterList(modelNames=['allenModelv1_standard', 'allenModelv2_highway', 'allenModelv3_convFinal'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 256,
        'lr': 5e-2,
        'momentum': 0.99,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

ALLEN_NET_BATCH_2_EASYAUGMENT = _getBatchParameterList(modelNames=['allenModelv1_standard', 'allenModelv2_highway', 'allenModelv3_convFinal'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 256,
        'lr': 5e-2,
        'momentum': 0.99,
        'weight_decay': 0.1,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

ALLEN_NET_BATCH_3_EASYAUGMENT = _getBatchParameterList(modelNames=['allenModelv1_standard', 'allenModelv2_highway', 'allenModelv3_convFinal'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 256,
        'lr': 5e-2,
        'momentum': 0.99,
        'weight_decay': 0.2,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })


ALLEN_NET_LITE_BATCH_1_EASYAUGMENT = _getBatchParameterList(modelNames=['allenModelv1Lite_standard', 'allenModelv2Lite_highway', 'allenModelv3Lite_convFinal'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 320,
        'lr': 5e-2,
        'momentum': 0.90,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

ALLEN_NET_LITE_BATCH_2_EASYAUGMENT = _getBatchParameterList(modelNames=['allenModelv1Lite_standard', 'allenModelv2Lite_highway', 'allenModelv3Lite_convFinal'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 320,
        'lr': 5e-2,
        'momentum': 0.99,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

ALLEN_NET_LITE_BATCH_3_EASYAUGMENT = _getBatchParameterList(modelNames=['allenModelv1Lite_standard', 'allenModelv2Lite_highway', 'allenModelv3Lite_convFinal'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 320,
        'lr': 5e-2,
        'momentum': 0.99,
        'weight_decay': 0.1,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

ALLEN_NET_LITE_BATCH_4_EASYAUGMENT = _getBatchParameterList(modelNames=['allenModelv2Lite_highway', 'allenModelv2Lite_highway', 'allenModelv2Lite_highway', 'allenModelv2Lite_highway', 'allenModelv2Lite_highway', 'allenModelv2Lite_highway'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 50,
        'warmupEpochs': 5,
        'batch_size': 320,
        'lr': 5e-2,
        'momentum': 0.90,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

ALLEN_NET_LITE_BATCH_5_EASYAUGMENT = _getBatchParameterList(modelNames=['allenModelv4Lite_highway_avgPool', 'allenModelv5Lite_highway_funnel', 'allenModelv5Lite_highway_Deep', 'allenModelv6Lite_highway_instanceNorm'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 320,
        'lr': 5e-2,
        'momentum': 0.90,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

WILSON_NET_BATCH_1_EASYAUGMENT = _getBatchParameterList(modelNames=['wilsonNetv1_ELU', 'wilsonNetv2_ELU_frontDeep', 'wilsonNetv3_ELU_rearDeep', 'wilsonNetv4_ELU_rearDoubleDeep', 'wilsonNetv5_PReLU'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 320,
        'lr': 5e-2,
        'momentum': 0.90,
        'weight_decay': 0.0, # No weight decay for ELU if we want good results with PReLU according to documentation
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

RESNET_18_BATCH_1_EASYAUGMENT = _getBatchParameterList(modelNames=['resNet18Test'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 320,
        'lr': 5e-2,
        'momentum': 0.90,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5,
        'customNormalization': 'RESNET_18_NORMALIZATION'
    })

JESSE_NET_BATCH_1_EASYAUGMENT = _getBatchParameterList(modelNames=['jesseNetv1', 'jesseNetv2', 'jesseNetv3', 'jesseNetv4', 'jesseNetv5'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 128,
        'lr': 5e-2,
        'momentum': 0.90,
        'weight_decay': 0.0, # No weight decay for ELU if we want good results with PReLU according to documentation
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

JESSE_NET_BATCH_2_EASYAUGMENT = _getBatchParameterList(modelNames=['jesseNetv6'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 64,
        'lr': 5e-2,
        'momentum': 0.90,
        'weight_decay': 0.0, # No weight decay for ELU if we want good results with PReLU according to documentation
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

WILSON_NET_FT_BATCH_1_EASYAUGMENT = _getBatchParametersSweep2(modelName='wilsonNetv5_PReLU',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 320,
        'lr': 5e-2,
        'momentum': list(np.linspace(start=0.7, stop=0.99, num=10)),
        'weight_decay': 0.0,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

JESSE_NET_BATCH_3_EASYAUGMENT = _getBatchParameterList(modelNames=['jesseNetv5_2_reverseEncode', 'jesseNetv7_3_multiHighway_mini', 'jesseNetv5_3_wideBranchesLinearEncode', 'jesseNetv5_4_doubleWideBranches', 'jesseNetv5_5_noHighway', 'jesseNetv7_2_multiHighway_duplicateBottle', 'jesseNetv7_3_multiHighway_mini', 'jesseNetv7_4_multiHighway_micro'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 128,
        'lr': 5e-2,
        'momentum': 0.90,
        'weight_decay': 0.0, # No weight decay for ELU if we want good results with PReLU according to documentation
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

JESSE_NET_BATCH_4_EASYAUGMENT = _getBatchParameterList(modelNames=['jesseNetv7_2_multiHighway_duplicateBottleRevEncode', 'jesseNetv7_2_multiHighway_duplicateBottleRevEncodex2', 'jesseNetv7_2_multiHighway_duplicateBottleRevEncodex2Compact'],
    nameSuffix='easyaugment',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 128,
        'lr': 5e-2,
        'momentum': 0.90,
        'weight_decay': 0.0, # No weight decay for ELU if we want good results with PReLU according to documentation
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

BASELINE_BOTTLENECK_BATCH_1_HARDAUGMENTATION2_5 = _getBatchParameterList(modelNames=['bottleneckResidualv1', 'bottleneckResidualv2', 'doubleBottleneckResidualv1'],
    nameSuffix='hardAugmentation2-5',
    parametersDict={
        'trainTransformID': 'hardAugmentation2_5',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 128,
        'lr': 5e-2,
        'momentum': 0.90,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

BASELINE_RESIDUALS_BATCH_1_HARDAUGMENTATION2_5 = _getBatchParameterList(modelNames=['branchResidualv1', 'branchResidualv2'],
    nameSuffix='hardAugmentation2-5',
    parametersDict={
        'trainTransformID': 'hardAugmentation2_5',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 128,
        'lr': 5e-2,
        'momentum': 0.90,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

####################################################################################
# HARDAUGMENT
####################################################################################


BASELINE_BATCH_1_HARDAUGMENT = _getBatchParameterList(modelNames=['baseline130k', 'baseline130kN', 'baseline430k', 'baseline430kN'],
    nameSuffix='hardAugmentation',
    parametersDict={
        'trainTransformID': 'hardAugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })


HIGHWAY_BATCH_1_HARDAUGMENT = _getBatchParameterList(modelNames=['highwaynetv3', 'highwaynetv4', 'highwaynetv5'],
    nameSuffix='hardAugmentation',
    parametersDict={
        'trainTransformID': 'hardAugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

HIGHWAY_BATCH_2_HARDAUGMENT = _getBatchParameterList(modelNames=['highwaynetv6', 'highwaynetv7', 'highwaynetv8'],
    nameSuffix='hardAugmentation',
    parametersDict={
        'trainTransformID': 'hardAugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

HIGHWAY_BATCH_3_HARDAUGMENT = _getBatchParameterList(modelNames=['highwaynetv9', 'highwaynetv10', 'highwaynetv11'],
    nameSuffix='hardAugmentation',
    parametersDict={
        'trainTransformID': 'hardAugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

DOUBLE_BOTTLE_BATCH_1_HARDAUGMENT = _getBatchParameterList(modelNames=['doubleBottlev3', 'doubleBottlev4', 'doubleBottlev5'],
    nameSuffix='hardAugmentation',
    parametersDict={
        'trainTransformID': 'hardAugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })


BASELINE_BATCH_4 = _getBatchParameterList(modelNames=['residualNetv1', 'bottleneckResidualv1', 'highwayResidualv1', 'branchResidualv1'],
    nameSuffix='hardAugmentation3',
    parametersDict={
        'trainTransformID': 'hardAugmentation3',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 256,
        'lr': 5e-2,
        'momentum': 0.9,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

BASELINE_BATCH_5 = _getBatchParameterList(modelNames=['baseline130k', 'baseline130kN', 'baseline430k', 'baseline430kN', 'baseline13MN', 'baseline36MN', 'baseline108MN'],
    nameSuffix='hardAugmentation3',
    parametersDict={
        'trainTransformID': 'hardAugmentation3',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 512,
        'lr': 5e-2,
        'momentum': 0.9,
        'weight_decay': 0.01,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })


JESSE_NET_BATCH_4_HARDAUGMENT3 = _getBatchParameterList(modelNames=['jesseNetv7_2_multiHighway_duplicateBottleRevEncode', 'jesseNetv7_2_multiHighway_duplicateBottleRevEncodex2', 'jesseNetv7_2_multiHighway_duplicateBottleRevEncodex2Compact', 'jesseNetv5_2_reverseEncode_EF2'],
    nameSuffix='hardAugmentation3',
    parametersDict={
        'trainTransformID': 'hardAugmentation3',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 128,
        'lr': 5e-2,
        'momentum': 0.90,
        'weight_decay': 0.0, # No weight decay for ELU if we want good results with PReLU according to documentation
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

WILSON_NET_BATCH_1_HARDAUGMENT3 = _getBatchParameterList(modelNames=['wilsonNetv1_ELU', 'wilsonNetv2_ELU_frontDeep', 'wilsonNetv3_ELU_rearDeep', 'wilsonNetv4_ELU_rearDoubleDeep', 'wilsonNetv5_PReLU'],
    nameSuffix='hardAugmentation3',
    parametersDict={
        'trainTransformID': 'hardAugmentation3',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 320,
        'lr': 5e-2,
        'momentum': 0.90,
        'weight_decay': 0.0, # No weight decay for ELU if we want good results with PReLU according to documentation
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

####################################################################################
# BIG MODELS
####################################################################################

BIG_MODEL_BATCH_1_EASYAUGMENT = _getBatchParameterList(modelNames=['bigModel1', 'bigmodel2'],
    nameSuffix='easyaugmentation',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.05,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

BIG_MODEL_BATCH_2_EASYAUGMENT = _getBatchParameterList(modelNames=['bigModel1', 'bigmodel2'],
    nameSuffix='easyaugmentation',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 512,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.05,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

BIG_MODEL_BATCH_3_EASYAUGMENT = _getBatchParameterList(modelNames=['bigmodel3'],
    nameSuffix='easyaugmentation',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.05,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

BIG_MODEL_BATCH_4_EASYAUGMENT = _getBatchParameterList(modelNames=['bigmodel3'],
    nameSuffix='easyaugmentation',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 4096,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.05,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

BIG_MODEL_BATCH_5_EASYAUGMENT = _getBatchParameterList(modelNames=['bigmodel4'],
    nameSuffix='easyaugmentation',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.05,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

BIG_MODEL_BATCH_6_EASYAUGMENT = _getBatchParameterList(modelNames=['bigmodel4'],
    nameSuffix='easyaugmentation',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 4096,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.05,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })



BIG_MODEL_BATCH_7_EASYAUGMENT = _getBatchParameterList(modelNames=['bigModel1_DBN2', 'bigmodel2_DBN2'],
    nameSuffix='easyaugmentation',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.05,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

BIG_MODEL_BATCH_8_EASYAUGMENT = _getBatchParameterList(modelNames=['bigModel1_DBN2', 'bigmodel2_DBN2'],
    nameSuffix='easyaugmentation',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.10,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })



BIG_MODEL_BATCH_9_EASYAUGMENT = _getBatchParameterList(modelNames=['bigmodel3_DBN2'],
    nameSuffix='easyaugmentation',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.10,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

BIG_MODEL_BATCH_10_EASYAUGMENT = _getBatchParameterList(modelNames=['bigmodel4_DBN2'],
    nameSuffix='easyaugmentation',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.10,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

BIG_MODEL_BATCH_11_EASYAUGMENT = _getBatchParameterList(modelNames=['bigmodel5_DBN2'],
    nameSuffix='easyaugmentation',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.10,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })
BIG_MODEL_BATCH_12_EASYAUGMENT = _getBatchParameterList(modelNames=['bigmodel6_DBN2'],
    nameSuffix='easyaugmentation',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.8,
        'weight_decay': 0.10,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

# NOTE: MOMENTUM IS VERY VERY IMPORTANT
BIG_MODEL_BATCH_13_EASYAUGMENT = _getBatchParameterList(modelNames=['bigmodel3_DBN2'],
    nameSuffix='easyaugmentation',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.5,
        'weight_decay': 0.2,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })

BIG_MODEL_BATCH_14_EASYAUGMENT = _getBatchParameterList(modelNames=['bigmodel4_DBN2'],
    nameSuffix='easyaugmentation',
    parametersDict={
        'trainTransformID': 'easyaugmentation',
        'valTestTransformID': 'NONE',
        'epochs': 200,
        'warmupEpochs': 5,
        'batch_size': 2048,
        'lr': 5e-2,
        'momentum': 0.5,
        'weight_decay': 0.2,
        'nesterov': True,
        'plateuPatience': 3,
        'plateuFactor': 0.5
    })






####################################################################################
# INDIVIDUAL
####################################################################################

baseline130k = {
    'modelName': 'baseline130k',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 200,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

baseline130kN = {
    'modelName': 'baseline130kN',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 200,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

baseline430k = {
    'modelName': 'baseline430k',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 200,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

baseline430kN = {
    'modelName': 'baseline430kN',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 200,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}




highwaynetv3 = {
    'modelName': 'highwaynetv3',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 200,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

highwaynetv4 = {
    'modelName': 'highwaynetv4',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 200,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

highwaynetv5 = {
    'modelName': 'highwaynetv5',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 200,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

highwaynetv6 = {
    'modelName': 'highwaynetv6',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 200,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

highwaynetv7 = {
    'modelName': 'highwaynetv7',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 200,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

highwaynetv8 = {
    'modelName': 'highwaynetv8',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 200,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

highwaynetv8L = {
    'modelName': 'highwaynetv8L',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 200,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

doubleBottlev1 = {
    'modelName': 'doubleBottlev1',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 200,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

doubleBottlev2 = {
    'modelName': 'doubleBottlev2',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 200,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

doubleBottlev3 = {
    'modelName': 'doubleBottlev3',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 200,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

doubleBottlev4 = {
    'modelName': 'doubleBottlev4',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 200,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

doubleBottlev5 = {
    'modelName': 'doubleBottlev5',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 200,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}







testModel1 = {
    'modelName': 'testModel',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 2,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

testModel2 = {
    'modelName': 'testModel',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 3,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

testModel3 = {
    'modelName': 'testModel',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 4,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}

testModel4 = {
    'modelName': 'testModel',
    'trainTransformID': 'default',
    'valTestTransformID': 'default',
    'epochs': 5,
    'warmupEpochs': 5,
    'batch_size': 2048,
    'lr': 5e-2,
    'momentum': 0.8,
    'weight_decay': 0.01,
    'nesterov': True,
    'plateuPatience': 3,
    'plateuFactor': 0.5
}