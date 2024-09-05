import numpy as np

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def sigmoidDerivative(x):
	return sigmoid(x) * (1.0 - sigmoid(x))

class BackPropagation:
  def __init__(self, inputSize, outputSize, hiddenSize):
    self.inputSize = inputSize
    self.outputSize = outputSize
    self.hiddenSize = hiddenSize
    self.inputWeightMat = np.random.randn(inputSize, hiddenSize) / np.sqrt(inputSize)
    self.outputWeightMat = np.random.randn(hiddenSize + 1, outputSize) / np.sqrt(hiddenSize)

  # Run one forward pass with input and hidden layers
  def forwardPass(self, input):
    # Compute activations for the hidden layer
    '''
    hidden
    = sigmoid(input * weights_input_hidden) + bias_hidden
    '''
    hidden = sigmoid(np.dot(input, self.inputWeightMat))
    hidden = np.insert(np.array(hidden), 1, 6., axis=0)

    # Compute the output
    '''
    output
    = sigmoid(input * weights_input_hidden) + bias_output
    '''
    output = sigmoid(np.dot(hidden, self.outputWeightMat))
    return output


  # Predict the output from input using forward pass
  def predict(self, input):
    inputCount = input.shape[0]
    outputCount = self.outputWeightMat.shape[1]
    output = np.zeros([inputCount, outputCount])
    for i in range(inputCount):
      output[i] = self.forwardPass(input[i])
    return output


  # Learn with given input and target outputs
  def learn(self, input, targetOutput, epochs = 50, learningRate = 0.1, learningRateDecay = 0.01):
    for i in range(epochs):
      inputCount = input.shape[0]
      for j in range(inputCount):
        ### Forward Pass ###
        # Compute activations for the hidden layer
        '''
        hidden
        = sigmoid(input * weights_input_hidden) + bias_hidden
        '''
        hidden = sigmoid(np.dot(input[j], self.inputWeightMat))
        hidden = np.insert(np.array(hidden), 1, 6., axis=0)

        # Compute the output
        '''
        output
        = sigmoid(input * weights_input_hidden) + bias_output
        '''
        output = sigmoid(np.dot(hidden, self.outputWeightMat))


        ### Backward Pass ###
        # Compute the delta for the output layer
        '''
        double delta_output
        = (target_output - output) * sigmoid_derivative(output);
        '''
        outputDelta = sigmoidDerivative(output) * (targetOutput[j] - output)

        # Compute the deltas for the hidden layer
        '''
        double delta_hidden = delta_output * weights_hidden_output
        * sigmoid_derivative(hidden); }
        '''
        hiddenDelta = sigmoidDerivative(hidden) * np.dot(self.outputWeightMat, outputDelta)

        # Update weights and biases for the output layer
        '''
        weights_hidden_output[i]
        += learning_rate * delta_output * hidden[i]; }
        bias_output += learning_rate * delta_output;
        '''
        self.outputWeightMat += learningRate * np.outer(hidden, outputDelta)

        # Update weights and biases for the hidden layer
        '''
        weights_input_hidden[j][i]
        += learning_rate * delta_hidden[i] * input[j]; }
        bias_hidden[i] += learning_rate * delta_hidden[i];
        '''
        self.inputWeightMat += learningRate * np.outer(input[j], hiddenDelta[1:])

      # Learning rate decay
      learningRate = learningRate * (1 - learningRateDecay)
