import numpy as np
import numba
import numba.types as nt

@numba.njit()
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@numba.njit()
def mse(output, target):
    return np.sum(0.5 * (target - output)**2)

spec = [
    ('inputDim', nt.int32),
    ('pbDim', nt.int32),
    ('contextDim', nt.int32),
    ('outputDim', nt.int32),
    ('useBias', nt.boolean),
    ('learningRate', nt.float64),
    ('optimizer', nt.unicode_type),
    ('beta1', nt.float64),
    ('beta2', nt.float64),
    ('t', nt.int32),
    ('Wux', nt.float64[:,:]),
    ('Wup', nt.float64[:,:]),
    ('Wuc', nt.float64[:,:]),
    ('Wvc', nt.float64[:,:]),
    ('bu', nt.float64[:]),
    ('bv', nt.float64[:]),
    ('dEdWux', nt.float64[:,:]),
    ('dEdWup', nt.float64[:,:]),
    ('dEdWuc', nt.float64[:,:]),
    ('dEdWvc', nt.float64[:,:]),
    ('dEdbu', nt.float64[:]),
    ('dEdbv', nt.float64[:]),
    ('m_dEdWux', nt.float64[:,:]),
    ('m_dEdWup', nt.float64[:,:]),
    ('m_dEdWuc', nt.float64[:,:]),
    ('m_dEdWvc', nt.float64[:,:]),
    ('m_dEdbu', nt.float64[:]),
    ('m_dEdbv', nt.float64[:]),
    ('v_dEdWux', nt.float64[:,:]),
    ('v_dEdWup', nt.float64[:,:]),
    ('v_dEdWuc', nt.float64[:,:]),
    ('v_dEdWvc', nt.float64[:,:]),
    ('v_dEdbu', nt.float64[:]),
    ('v_dEdbv', nt.float64[:]),
    ('mostRecentContextState', nt.float64[:]),
    ('carryForwardContext', nt.boolean),
    ('predictionSteps', nt.int32),
]

STATIC_PB_VALUE = 0

@numba.experimental.jitclass(spec)
class ElmanPBRNN():
    
    def __init__(self, inputDim=None, pbDim=None, contextDim=None, outputDim=None, learningRate=1e-3, optimizer='', beta1=.9, beta2=.999):
      
        self.inputDim = inputDim
        self.pbDim = pbDim
        self.contextDim = contextDim # Aka hidden layer dim
        self.outputDim = outputDim
        self.learningRate = learningRate
        # Adam parameters
        self.beta1 = beta1
        self.beta2 = beta2
                    
        if optimizer == 'adam':
            self.optimizer = 'adam'
        else:
            self.optimizer = ''

        self.initialize()
            
    def initialize(self):
        """
        Initialize the weights and biases by sampling from a 
        standard normal distribution.
        """
        # Input layer to context layer
        self.Wux = np.ascontiguousarray(np.random.randn(self.contextDim, self.inputDim))
        # Parametric bias to context layer
        self.Wup = np.ascontiguousarray(np.random.randn(self.contextDim, self.pbDim))
        # Previous context layer to context layer
        self.Wuc = np.ascontiguousarray(np.random.randn(self.contextDim, self.contextDim))
        # Context layer to output layer
        self.Wvc = np.ascontiguousarray(np.random.randn(self.outputDim, self.contextDim))
        
        self.bu = np.ascontiguousarray(np.random.randn(self.contextDim))
        self.bv = np.ascontiguousarray(np.random.randn(self.outputDim))

        # Set the momentum terms for adam
        self.m_dEdWux = np.zeros_like(self.Wux)
        self.m_dEdWup = np.zeros_like(self.Wup)
        self.m_dEdWuc = np.zeros_like(self.Wuc)
        self.m_dEdWvc = np.zeros_like(self.Wvc)
        
        self.v_dEdWux = np.zeros_like(self.Wux)
        self.v_dEdWup = np.zeros_like(self.Wup)
        self.v_dEdWuc = np.zeros_like(self.Wuc)
        self.v_dEdWvc = np.zeros_like(self.Wvc)

        self.m_dEdbu = np.zeros_like(self.bu)
        self.m_dEdbv = np.zeros_like(self.bv)

        self.v_dEdbu = np.zeros_like(self.bu)
        self.v_dEdbv = np.zeros_like(self.bv)

        # Number of times we have updated by adam
        # (needed for bias correction)
        self.t = 1
       
        # This is so that we can predict with 1 step increments
        # and carry over the context state
        self.resetContext()

        # Set all gradients to zero
        self._resetGradients()
   
    def resetContext(self):
        self.mostRecentContextState = np.zeros(self.contextDim)

    def _forwardStep(self, inputArr, parametricBiasArr, prevContextArr):
        """
        Perform a single forward pass of the network, calculating the output and
        values of context neurons.
        
        Parameters
        ----------
        inputArr : np.ndarray[self.inputDim,]
            The input values for the current time step, t.
            
        prevContextArr : Np.ndarray[self.contextDim,]
            The context neuron values for the previous time step, t - 1.
            
        Returns
        -------
        c : np.ndarray[self.contextDim,]
            The context neuron values for the current time step, t.
            
        o : np.ndarray[self.outputDim,]
            The output values for the current time step, t.
        """
        u = self.Wux @ inputArr + self.Wup @ parametricBiasArr + self.Wuc @ prevContextArr + self.bu
        c = sigmoid(u)
        v = self.Wvc @ c + self.bv
        o = sigmoid(v)
        return c, o
    
   
    def forwardSequence(self, inputArr, parametricBiasArr):
        """
        Compute the output and context values for an input sequence.
        
        Parameters
        ----------
        inputArr : np.ndarray[T, self.inputDim]
            The input sequence for T timesteps
            
        Returns
        -------
        contextArr : np.ndarray[T, self.contextDim]
            The values of the context neurons at each timestep.
            
        outputArr : np.ndarray[T, self.outputDim]
            The output values at each timestep.
        """

        contextArr = np.zeros((len(inputArr), self.contextDim))
        outputArr = np.zeros((len(inputArr), self.outputDim))
        
        # Numba can get annoyed depending on how these are constructed,
        # so it's easiest just to always cast as contiguous arrays
        inputArrC = np.ascontiguousarray(inputArr)
        parametricBiasArrC = np.ascontiguousarray(parametricBiasArr)

        for i in range(1, len(inputArr)):
            # Reshape part adds [[ ]] around the value (or as many brackets are needed
            # to end up with two around the elements)
            #print(inputArr[i].reshape(self.inputDim))
            #print(parametricBiasArr.reshape(self.pbDim))
            #print(contextArr[i-1])
            c, o = self._forwardStep(inputArrC[i].reshape(self.inputDim), parametricBiasArrC.reshape(self.pbDim), contextArr[i-1])
            contextArr[i] = c
            outputArr[i] = o
                    
        return contextArr, outputArr
        
        
    def _resetGradients(self):
        """
        Set all gradients to zero, presumably after updating learning parameters.
        """
        # Set all of the gradients to zero
        self.dEdWuc = np.ascontiguousarray(np.zeros_like(self.Wuc))
        self.dEdWup = np.ascontiguousarray(np.zeros_like(self.Wup))
        self.dEdWux = np.ascontiguousarray(np.zeros_like(self.Wux))
        self.dEdWvc = np.ascontiguousarray(np.zeros_like(self.Wvc))
        self.dEdbu = np.ascontiguousarray(np.zeros_like(self.bu))
        self.dEdbv = np.ascontiguousarray(np.zeros_like(self.bv))
    
    
    def _backwardStep(self, inputArr, parametricBiasArr, contextArr, outputArr, targetArr, prevContextArr, prevdEdu):
        """
        Compute a single backward step for the backpropagation.
        
        Parameters
        ----------
        inputArr : np.ndarray[self.inputDim,]
            The input to the network at the current time step, t.
        
        contextArr : np.ndarray[self.contextDim,]
            The context layer values for the network at the current time
            step, t.

        outputArr : np.ndarray[self.outputDim,]
            The output of the network at the current time step, t.
            
        targetArr : np.ndarray[self.outputDim,]
            The target output for the network at the current time step, t.
                        
        prevContextArr : np.ndarray[self.contextDim,]
            The context layer values for the network at the 'previous' current
            time step, t + 1 (since we are going backwards).

        prevdEdu : np.ndarray[self.contextDim,]
            The gradient of the error with respect to the inputs for the
            'previous' time step, t + 1 (since we are going backwards). If
            on the final timestep, `np.zeros((self.contextDim, 1))`
            should be passed.
        """
        #print(inputArr.shape)
        #print(contextArr.shape)
        #print(outputArr.shape)
        #print(targetArr.shape)
        #print(prevContextArr.shape)
        #print(prevdEdu.shape)

        # Derivative of mse is just difference
        dEdy = outputArr - targetArr
                
        # x (1 - x) form comes from derivative of the sigmoid
        dEdv = dEdy * (1 - outputArr) * outputArr
        
        # Has two contributions: one from current step and one from previous
        # Note that 'prevdEdu' is actually the derivative for the
        # next step in time, since we are going backwards
        dEdc = self.Wvc.T @ dEdv + self.Wuc.T @ prevdEdu
        #print(np.shape(dEdv))
        #print(np.shape(self.Wvc))
        #dEdc = dEdv.T @ self.Wvc + prevdEdu.T @ self.Wuc
        
        # Now calculate this derivative for the current timestep
        dEdu = dEdc * (1 - contextArr) * contextArr
        
        #print(dEdv.shape)
        #print(contextArr.T.shape)
        #print(dEdu.shape)
        #print(prevContextArr.T.shape)
        #print(inputArr.T.shape)
        #print(parametricBiasArr.T.shape)
        # Now we can calculate the gradients for our learnable
        # parameters from the ones above
        # We want to keep track of these as cumulative values
        # Weights
        self.dEdWvc += dEdv @ contextArr.T
        self.dEdWuc += dEdu @ prevContextArr.T
        self.dEdWux += dEdu @ inputArr.T
        self.dEdWup += dEdu @ parametricBiasArr.T

        # Biases
        #print(dEdv.shape)
        #print(self.dEdbv.shape)
        self.dEdbv += dEdv.reshape(self.dEdbv.shape)
        self.dEdbu += dEdu.reshape(self.dEdbu.shape)

        # We need this back at the end, otherwise we can't compute the next step
        return dEdu
    
    
    def backwardSequence(self, inputArr, parametricBiasArr, contextArr, outputArr, targetArr):
        """
        Compute the backpropagation computation through the network.
        
        Parameters
        ----------
        inputArr : np.ndarray[T, self.inputDim]
            The input array at each timestep.
            
        contextArr : np.ndarray[T, self.contextDim]
            The values of the context neurons at each timestep.
            
        outputArr : np.ndarray[T, self.outputDim]
            The output of the network at each timestep.
            
        targetArr : np.ndarray[T, self.outputDim]
            The target output at each timestep.
            
        Returns
        -------
        error : np.ndarray[T]
            The MSE at each time step.
        """
        #print(inputArr.shape)
        #print(contextArr.shape)
        #print(outputArr.shape)

        
        # Numba can get annoyed depending on how these are constructed,
        # so it's easiest just to always cast as contiguous arrays
        inputArrC = np.ascontiguousarray(inputArr)
        parametricBiasArrC = np.ascontiguousarray(parametricBiasArr)
        contextArrC = np.ascontiguousarray(contextArr)
        outputArrC = np.ascontiguousarray(outputArr)
        targetArrC = np.ascontiguousarray(targetArr)

        # Number of timesteps
        T = inputArrC.shape[0]
        
        # Add a plus 1 so that we can pass zeros to the last
        # step without extra work (and we clip it off later)
        dEdu = np.zeros((T+1, self.contextDim))
        error = 0
        
        # Reset gradients
        self._resetGradients()
        
        # Count down from last time step to first and compute gradients
        for i in sorted(range(T), reverse=True):
            # Compute gradients for learnable quantities
            dEdu[i] = self._backwardStep(inputArrC[i].reshape(-1, 1),
                                         parametricBiasArrC.reshape(-1, 1),
                                         contextArrC[i].reshape(-1, 1),
                                         outputArrC[i].reshape(-1, 1),
                                         targetArrC[i].reshape(-1, 1),
                                         contextArrC[i+1].reshape(-1, 1) if i < T-1 else np.zeros((self.contextDim, 1)),
                                         dEdu[i+1].reshape(-1, 1))[:,0]
            # Compute MSE error from output and target
            error += mse(outputArrC, targetArrC)
            
        # Clip off the extra entry (see above)
        # Not totally necessary since we don't do anything with it, but :/
        dEdu = dEdu[:-1]
        
        return error
        
    def updateParameters(self):
        """
        Update the learning parameters from the stored gradients
        using the Adam optimization scheme.
        """

        if self.optimizer == 'adam':
            # Update momentum terms
            self.m_dEdWux = self.beta1 * self.m_dEdWux + (1 - self.beta1) * self.dEdWux
            self.m_dEdWup = self.beta1 * self.m_dEdWup + (1 - self.beta1) * self.dEdWup
            self.m_dEdWuc = self.beta1 * self.m_dEdWuc + (1 - self.beta1) * self.dEdWuc
            self.m_dEdWvc = self.beta1 * self.m_dEdWvc + (1 - self.beta1) * self.dEdWvc

            self.m_dEdbu = self.beta1 * self.m_dEdbu + (1 - self.beta1) * self.dEdbu
            self.m_dEdbv = self.beta1 * self.m_dEdbv + (1 - self.beta1) * self.dEdbv

            # Update rms terms
            self.v_dEdWux = self.beta2 * self.v_dEdWux + (1 - self.beta2) * self.dEdWux**2
            self.v_dEdWup = self.beta2 * self.v_dEdWup + (1 - self.beta2) * self.dEdWup**2
            self.v_dEdWuc = self.beta2 * self.v_dEdWuc + (1 - self.beta2) * self.dEdWuc**2
            self.v_dEdWvc = self.beta2 * self.v_dEdWvc + (1 - self.beta2) * self.dEdWvc**2

            self.v_dEdbu = self.beta2 * self.v_dEdbu + (1 - self.beta2) * self.dEdbu**2
            self.v_dEdbv = self.beta2 * self.v_dEdbv + (1 - self.beta2) * self.dEdbv**2

            # Bias correction
            mCorr = 1 - self.beta1**self.t
            vCorr = 1 - self.beta2**self.t

            eps = 1e-8 # Some very small value to prevent div by 0 errors

            # Now actually update everything
            # Weights
            dWvc = (self.m_dEdWvc * self.learningRate / mCorr) / (np.sqrt(self.v_dEdWvc / vCorr) + eps)
            dWuc = (self.m_dEdWuc * self.learningRate / mCorr) / (np.sqrt(self.v_dEdWuc / vCorr) + eps)
            dWux = (self.m_dEdWux * self.learningRate / mCorr) / (np.sqrt(self.v_dEdWux / vCorr) + eps)
            dWup = (self.m_dEdWup * self.learningRate / mCorr) / (np.sqrt(self.v_dEdWup / vCorr) + eps)
            # Biases
            dbv = (self.m_dEdbv * self.learningRate / mCorr) / (np.sqrt(self.v_dEdbv / vCorr) + eps)
            dbu = (self.m_dEdbu * self.learningRate / mCorr) / (np.sqrt(self.v_dEdbu / vCorr) + eps)

            # Reset gradients so we don't accidentally update
            # twice
            self._resetGradients()

            # Increment the update counter (used for bias)
            self.t += 1

            #sumArr = [np.sum(dWvc), np.sum(dWuc), np.sum(dWux), np.sum(dbv), np.sum(dbu)]

            self.Wvc -= dWvc
            self.Wuc -= dWuc
            self.Wux -= dWux
            self.Wup -= dWup

            self.bu -= dbu
            self.bv -= dbv
        
        else:
            # Update via (first order) gradient descent
            # Weights
            self.Wvc -= self.dEdWvc * self.learningRate
            self.Wuc -= self.dEdWuc * self.learningRate
            self.Wux -= self.dEdWux * self.learningRate
            self.Wup -= self.dEdWup * self.learningRate
            # Biases
            self.bv -= self.dEdbv * self.learningRate
            self.bu -= self.dEdbu * self.learningRate
        
    def predict(self, inputArr, parametricBiasArr, predictionSteps=1, carryForwardContext=False):
        """
        Predict the next point(s) in a series using the trained
        parameters.
        
        Parameters
        ----------
        inputArr : np.ndarray[self.inputDim,]
            The single time step to predict from.
        
        predictionSteps : int
            The number of steps to predict into the future.
            
        Returns
        -------
        outputArr : np.ndarray or float
            Returns the sequence of predicted points (including the original
            input point).
        """

        outputArr = np.zeros((predictionSteps+1,self.outputDim))
        contextArr = np.zeros((predictionSteps+1,self.contextDim))
        
        if carryForwardContext:
            contextArr[0] = self.mostRecentContextState
        else:
            self.resetContext()

        outputArr[0] = inputArr
        for i in range(1, predictionSteps+1):
            contextArr[i], outputArr[i] = self._forwardStep(outputArr[i-1], parametricBiasArr, contextArr[i-1])
       
        if carryForwardContext:
            self.mostRecentContextState = contextArr[-1]

        return outputArr
        
   
    def recognize(self, inputData, iterations=100, learningRate=1e-3):
        """

        """
        scatterSize = 1000
        testBiasArr = np.random.uniform(0, 1, size=(scatterSize, self.pbDim))
        errorArr = np.zeros(scatterSize)

        inputArr = inputData[:-1]
        targetArr = inputData[1:]

        for i in range(scatterSize):
            contextArr, tempOutputArr = self.forwardSequence(inputArr, testBiasArr[i])
            #tempOutputArr = self.predict(inputArr[0], testBiasArr[i], len(inputArr)-1)

            # Normalize the output and target to see if they are the same shape
            # This is just a usual normalization, but it looks gross because
            # numba doesn't support the axis kwarg of np.min or np.max, so we
            # have to split the components manually
            normMin, normMax = .15, .85
            outputArr = np.zeros_like(tempOutputArr)
            
            #targetArr = np.zeros_like(inputData[1:])
            #for k in range(2):
            #    outputArr[:,k] = tempOutputArr[:,k] - np.min(tempOutputArr[:,k])
            #    if np.max(outputArr[:,k]) > 0:
            #        outputArr[:,k] = outputArr[:,k] / np.max(outputArr[:,k]) * (normMax - normMin) + normMin

            #    targetArr[:,k] = inputData[1:,k] - np.min(inputData[1:,k])
            #    if np.max(targetArr[:,k]) > 0:
            #        targetArr[:,k] =  targetArr[:,k] / np.max(targetArr[:,k]) * (normMax - normMin) + normMin

            errorArr[i] = mse(tempOutputArr, targetArr)
       
        recognizedBias = testBiasArr[np.argmin(errorArr)]
        
        #return recognizedBias
        # Randomly initialize the parametric bias that we will update
        #recognizedBias = np.zeros(self.pbDim)
        #recognizedBias[0] = 1
        #recognizedBias = np.random.uniform(0, 1, size=self.pbDim)

        m_dEdp = np.zeros((self.pbDim, 1))
        v_dEdp = np.zeros((self.pbDim, 1))

        inputArr = inputData[:-1]
        #targetArr = inputData[1:]

        for j in range(iterations):
            # Compute forward sequence (same as prediction)
            contextArr, tempOutputArr = self.forwardSequence(inputArr, recognizedBias)
           

            # Normalize the output and target to see if they are the same shape
            # This is just a usual normalization, but it looks gross because
            # numba doesn't support the axis kwarg of np.min or np.max, so we
            # have to split the components manually
            normMin, normMax = .15, .85
            outputArr = np.zeros_like(tempOutputArr)
            targetArr = np.zeros_like(inputData[1:])
            for k in range(2):
                outputArr[:,k] = tempOutputArr[:,k] - np.min(tempOutputArr[:,k])
                if np.max(outputArr[:,k]) > 0:
                    outputArr[:,k] = outputArr[:,k] / np.max(outputArr[:,k]) * (normMax - normMin) + normMin

                targetArr[:,k] = inputData[1:,k] - np.min(inputData[1:,k])
                if np.max(targetArr[:,k]) > 0:
                    targetArr[:,k] =  targetArr[:,k] / np.max(targetArr[:,k]) * (normMax - normMin) + normMin

            # Number of timesteps
            T = inputArr.shape[0]
            
            # Add a plus 1 so that we can pass zeros to the last
            # step without extra work (and we clip it off later)
            dEdu = np.zeros((T+1, self.contextDim, 1))
            dEdp = np.zeros((self.pbDim, 1))
            errorArr = np.zeros((T, self.outputDim))

            # Reset gradients
            self._resetGradients()
            
            # Count down from last time step to first and compute gradients
            for i in sorted(range(T), reverse=True):
                # Compute gradients for only the parametric bias
                # This is similar to the calculation for the offsets (
                # Helps to define these earlier, so the rest of the code can be
                # similar to the self._backwardStep function
                currInputArr = inputArr[i].reshape(-1, 1)
                currContextArr = contextArr[i].reshape(-1, 1)
                currOutputArr = outputArr[i].reshape(-1, 1)
                currTargetArr = targetArr[i].reshape(-1, 1)
                prevContextArr = contextArr[i+1].reshape(-1, 1) if i < T-1 else np.zeros((self.contextDim, 1))
                prevdEdu = dEdu[i+1].reshape(-1, 1)

                # Derivative of mse is just difference
                dEdy = currOutputArr - currTargetArr
                #print(dEdy.shape)
                errorArr[i] = np.ascontiguousarray(dEdy[:,0])

                # Comes from derivative of the sigmoid
                dEdv = dEdy * (1 - currOutputArr) * currOutputArr
                
                # Has two contributions: one from current step and one from previous
                # Note that 'prevdEdu' is actually the derivative for the
                # next step in time, since we are going backwards
                #dEdc = self.Wvc.T @ dEdv + self.Wuc.T @ prevdEdu

                # Now calculate this derivative for the current timestep
                #dEdu[i] = dEdc * (1 - currContextArr) * currContextArr
               
                #dEdp += dEdu[i].T @ self.Wup
                #print(np.shape((currContextArr * (1 - currContextArr))))
                #print(np.shape(dEdv.T @ self.Wvc))
                #print(np.shape(self.Wup.T))
                dEdp += (dEdv.T @ self.Wvc + prevdEdu.T @ self.Wuc) * (currContextArr * (1 - currContextArr)).T @ self.Wup
                #dEdp += (dEdv.T @ self.Wvc) * (currContextArr * (1 - currContextArr)).T @ self.Wup
           
            #print(errorArr)
            #print(np.sum(errorArr))
            # Take an average
            dEdp /= T

            dp = dEdp * learningRate
            recognizedBias = recognizedBias - dp[:,0]
            #print(recognizedBias)

        return recognizedBias


def save(self, file):
    """
    Saves the trained network parameters to a file.
    
    Parameters
    ----------
    file : str
        The filename to save the model to; should be a .npz file.
    """
    np.savez(file=file, Wux=self.Wux, Wup=self.Wup, Wuc=self.Wuc, Wvc=self.Wvc, bu=self.bu, bv=self.bv,
             hyperparameters=[self.inputDim, self.pbDim, self.contextDim, self.outputDim, self.learningRate])
    

def load(file):

    params = np.load(file)
    
    inputDim = int(params["hyperparameters"][0])
    pbDim = int(params["hyperparameters"][1])
    contextDim = int(params["hyperparameters"][2])
    outputDim = int(params["hyperparameters"][3])
    learningRate = params["hyperparameters"][4]

    self = ElmanPBRNN(inputDim, pbDim, contextDim, outputDim, learningRate)
    
    self.Wux = params["Wux"]
    self.Wup = params["Wup"]
    self.Wuc = params["Wuc"]
    self.Wvc = params["Wvc"]
    
    self.bu = params["bu"]
    self.bv = params["bv"]

    return self
