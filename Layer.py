from Neuron import *

class Layer:
	
	def __init__(self,numNeurons):
		self.numNeurons = numNeurons;
		self.neurons = [Neuron() for i in range(numNeurons)]

	def fire(self):
		for neuron in self.neurons:
			neuron.fire()
			
	def connect(self,layer):
		for neuron in self.neurons:
			for layerNeuron in layer.neurons:
				neuron.addInput(layerNeuron)
	
	def randomizeWeights(self):
		for neuron in self.neurons:
			neuron.randomizeWeights()
	
	def adjustWeights(self):
		for neuron in self.neurons:
			neuron.adjustWeights()
			
	def getSumErrorSquared(self):
		avgError = 0
		for neuron in self.neurons:
			avgError = avgError + neuron.error*neuron.error
		
		return avgError
	
	def getSize(self):
		return self.numNeurons
			
class InputLayer(Layer):
	def __init__(self,numNeurons):
		self.numNeurons = numNeurons;
		self.neurons = [InputNeuron() for i in range(numNeurons)]

class HiddenLayer(Layer):
	def __init__(self,numNeurons):
		self.numNeurons = numNeurons;
		self.neurons = [HiddenNeuron() for i in range(numNeurons)]
		for i in range(numNeurons):
			self.neurons[i].position = i
	
	def adjustWeights(self,outputNeurons):
		for neuron in self.neurons:
			neuron.adjustWeights(outputNeurons)
	
		

class OutputLayer(Layer):
	def __init__(self,numNeurons):
		self.numNeurons = numNeurons;
		self.neurons = [OutputNeuron() for i in range(numNeurons)]
		
	def calculateGradients(self):
		for neuron in self.neurons:
			neuron.calculateGradient()