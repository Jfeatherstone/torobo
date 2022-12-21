import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mse(output, target):
    return np.sum(0.5 * (target - output)**2)

def normalize(data, normMin=.15, normMax=.85, returnFunc=False):
    
    normalizedData = data - np.min(data)
    normalizedDataMax = np.max(normalizedData)
    normalizedData = normalizedData / normalizedDataMax * (normMax - normMin) + normMin
    
    def reverse(x):
        return (x - normMin) / (normMax - normMin) * normalizedDataMax + np.min(data)
    
    return (normalizedData, reverse) if returnFunc else normalizedData

def closedLoop(model, x0, nSteps):
    outputArr = np.zeros((nSteps+1, x0.shape[-1]))
    outputArr[0] = x0
    
    for i in range(nSteps):
        outputArr[i+1] = model(outputArr[i:i+1])
        
    return outputArr
