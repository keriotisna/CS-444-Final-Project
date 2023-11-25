

def _getBatchParameterList(modelNames:list, nameSuffix='', 
        parametersDict:dict={
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
        if len(nameSuffix) > 0:
            suffix = '_' + nameSuffix
        
        workingDict['modelName'] = currentName + suffix
        batchParametersList.append(workingDict)
    
    return batchParametersList





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