

def _getBatchParameterList(modelNames:list, 
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

    Returns:
        batchParametersList: A list of identical parameters with the modelName attribute set to the given model names.
    """
    
    batchParametersList = []
    
    for currentName in modelNames:
        
        workingDict = parametersDict.copy()
        
        workingDict['modelName'] = currentName
        batchParametersList.append(workingDict)
    
    return batchParametersList





BASELINE_BATCH_1 = _getBatchParameterList(modelNames=['baseline130k_vanilla', 'baseline130kN_vanilla', 'baseline430k_vanilla', 'baseline430kN_vanilla'],
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

BASELINE_BATCH_1_AUTOAUGMENT = _getBatchParameterList(modelNames=['baseline130k_autoaugment', 'baseline130kN_autoaugment', 'baseline430k_autoaugment', 'baseline430kN_autoaugment'],
    parametersDict={
        'trainTransformID': 'autoaugment',
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

HIGHWAY_BATCH_1_AUTOAUGMENT = _getBatchParameterList(modelNames=['highwaynetv3_autoaugment', 'highwaynetv4_autoaugment', 'highwaynetv5_autoaugment'],
    parametersDict={
        'trainTransformID': 'autoaugment',
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

HIGHWAY_BATCH_2_AUTOAUGMENT = _getBatchParameterList(modelNames=['highwaynetv6_autoaugment', 'highwaynetv7_autoaugment', 'highwaynetv8_autoaugment'],
    parametersDict={
        'trainTransformID': 'autoaugment',
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

DOUBLE_BOTTLE_BATCH_1_AUTOAUGMENT = _getBatchParameterList(modelNames=['doubleBottlev3_autoaugment', 'doubleBottlev4_autoaugment', 'doubleBottlev5_autoaugment'],
    parametersDict={
        'trainTransformID': 'autoaugment',
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