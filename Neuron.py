import random
import math

class InputNeuron:
	def __init__(self):
		self.input = 0
		self.hasFired = True
		self.output = 0
	
	def setInput(self,input):
		self.input = float(input)
		self.output = self.input
		
	def fire(self):
		return self.input	

class Neuron:
	
	learningRate = 0.01
	momentum = 0.001
	
	def __init__(self):
		self.weights = []
		self.prevWeightDeltas = []
		self.inputs = []
		self.output = 0
		self.desiredOutput = 0
		self.error = 0
		self.gradient = 0
		self.hasFired = False
		self.bias = 0
	
	def setDesiredOutput(self,desout):
		self.desiredOutput = desout
		
	def addWeight(self,wt):
		self.weights.append(wt)
		
	def setBias(self,bias):
		self.bias = bias
		
	def addInput(self,inpt):
		self.inputs.append(inpt)
		self.prevWeightDeltas.append(0)
		
	def setInputNeurons(self,inpts):
		self.inputs = []
		for inpt in inpts:
			self.addInput(inpt)
			
			
	def setWeights(self,wts):
		for wt in wts:
			self.weights.append(wt)
			
			
	def fire(self):
		if self.hasFired == False: 
			for i in range(len(self.inputs)):
				self.output = self.output + self.inputs[i].fire()*self.weights[i]
			self.hasFired = True
			self.output = math.tanh(self.output + self.bias)
			self.error = self.desiredOutput - self.output
		#self.print()
		return self.output

	def randomizeWeights(self):
		for i in self.inputs:
			self.weights.append((random.random()-0.5))
		
		self.bias = (random.random()-0.5)
	
	def resetOutput(self):
		self.output = 0
		self.hasFired = False
			
	def print_output(self):
		print('Output for Neuron : ',self.output)
		
class HiddenNeuron(Neuron):
	
	def __init__(self):
		super().__init__()
		self.position = 0
	
	def calculateGradient(self,outputNeurons):
		summation = 0
		
		for outputNeuron in outputNeurons:
			summation = summation + outputNeuron.gradient*outputNeuron.weights[self.position]
			
		self.gradient = 0.38852 * (1.7152 - self.output) * (1.7152 + self.output) * summation

	def adjustWeights(self,outputNeurons):
		self.calculateGradient(outputNeurons)
		
		for i in range(len(self.inputs)):
			weightDelta = Neuron.momentum*self.prevWeightDeltas[i] + Neuron.learningRate*self.gradient*self.inputs[i].fire()
			self.weights[i] = self.weights[i] + weightDelta
			
			self.prevWeightDeltas[i] = weightDelta
		
		self.bias = self.bias + Neuron.learningRate*self.gradient
					
class OutputNeuron(Neuron):
	
	def calculateGradient(self):
		self.gradient = 0.38852 * self.error * (1.7159 - self.output) * (1.7159 + self.output);
		
	def adjustWeights(self):
		self.calculateGradient()
		if self.error != 0.0:
			for i in range(len(self.inputs)):
				weightDelta = Neuron.momentum*self.prevWeightDeltas[i] + Neuron.learningRate*self.gradient*self.inputs[i].fire()
				self.weights[i] = self.weights[i] + weightDelta
		
		self.bias = self.bias + Neuron.learningRate*self.gradient