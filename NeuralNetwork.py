from Layer import *
from TrainingSet import *
from Neuron import *
		
class Network:
	
	def __init__(self,*neuronsForLayer):
		self.inputNeurons = neuronsForLayer[0]
		self.hiddenNeurons = neuronsForLayer[1]
		self.outputNeurons = neuronsForLayer[len(neuronsForLayer) - 1]
		
		self.initLayers()
		
	def initLayers(self):
		self.inputLayer = InputLayer(self.inputNeurons)
		self.hiddenLayer = HiddenLayer(self.hiddenNeurons)
		self.outputLayer = OutputLayer(self.outputNeurons)
		
		self.connectLayers()
		
		print('Neural Network Created\n')
		print('Input Neurons : ',self.inputNeurons,'\nHidden Neurons : ',self.hiddenNeurons,'\nOutputNeurons : ',self.outputNeurons,'\n')
		
	def connectLayers(self):
		self.outputLayer.connect(self.hiddenLayer)
		self.hiddenLayer.connect(self.inputLayer)
	
	def randomizeWeights(self):
		self.hiddenLayer.randomizeWeights()
		self.outputLayer.randomizeWeights()
		
	def randomizeInputs(self):
		for neuron in self.inputLayer.neurons:
			neuron.setInput(random.random())
	
	def setInput(self,inputs):
		for i in range(self.inputNeurons):
			self.inputLayer.neurons[i].setInput(inputs[i])
			
	def fireForInput(self,inputs):
		self.resetAllFiredStates()
		self.setInput(inputs)
		self.fire()
		print('Input : ',inputs[:],'\nOutput : ',self.getOutput())
			
	def fire(self):
		self.outputLayer.fire()
		
	def resetAllFiredStates(self):
		for neuron in self.outputLayer.neurons:
			neuron.resetOutput()
		
		for neuron in self.hiddenLayer.neurons:
			neuron.resetOutput()
		
	def getOutput(self):
		output = []
		for neuron in self.outputLayer.neurons:
			output.append(neuron.output)
		return output
	
	def saveNetworkToFile(self,filename):
		with open(filename,'w') as networkFile:
			networkFile.write(str(self.inputNeurons) + '\n')
			networkFile.write(str(self.hiddenNeurons) + '\n')
			networkFile.write(str(self.outputNeurons) + '\n')
			
			for i in range(self.hiddenNeurons):
				string = ''
				for weight in self.hiddenLayer.neurons[i].weights:
					string = string + str(weight) + ','
				if i == self.hiddenNeurons - 1:
					networkFile.write(string + str(self.hiddenLayer.neurons[i].bias))
				else:
					networkFile.write(string + str(self.hiddenLayer.neurons[i].bias) +',')
			networkFile.write('\n')
			
			for i in range(self.outputNeurons):
				string = ''
				for weight in self.outputLayer.neurons[i].weights:
					string = string + str(weight) + ','
				if i == self.outputNeurons - 1:
					networkFile.write(string[0:len(string) - 1]+ ','+str(self.outputLayer.neurons[i].bias))
				else:
					networkFile.write(string+ ','+ str(self.outputLayer.neurons[i].bias) +',')
				
	@staticmethod	
	def loadNetworkFromFile(filename):
		with open(filename,'r') as networkFile:
			
			inputNeurons = int(networkFile.readline())
			hiddenNeurons = int(networkFile.readline())
			outputNeurons = int(networkFile.readline())
			nn = Network(inputNeurons,hiddenNeurons,outputNeurons)

			weights = networkFile.readline().split(',')
			
			current = 0
			position = 0
			for i in range(len(weights)):		
				if current < inputNeurons:
					nn.hiddenLayer.neurons[position].addWeight(float(weights[i]))
					current = current + 1
				elif current == inputNeurons:
					nn.hiddenLayer.neurons[position].setBias(float(weights[i]))
					position = position + 1
					current = 0
	
			weights = networkFile.readline().split(',')
			
			current = 0
			position = 0
			for i in range(len(weights)):
				if current < hiddenNeurons:
					nn.outputLayer.neurons[position].addWeight(float(weights[i]))
					current = current + 1
				elif current == hiddenNeurons:
					nn.outputLayer.neurons[position].setBias(float(weights[i]))
					position = position + 1
					current = 0

		return nn		
		
	def train(self,trainingSet,numEpochs):
		errorForPrevEpoch = 0
		for e in range(numEpochs):
			errorForCurrEpoch = 0
			for i in range(len(trainingSet.inputSet)):
				avgError = 0
				self.resetAllFiredStates()
				for j in range(len(trainingSet.inputSet[i])):
					self.inputLayer.neurons[j].setInput(trainingSet.inputSet[i][j])
			
				for j in range(len(trainingSet.outputSet[i])):
					self.outputLayer.neurons[j].setDesiredOutput(trainingSet.outputSet[i][j])
			
				self.fire()
				self.outputLayer.calculateGradients()
				self.hiddenLayer.adjustWeights(self.outputLayer.neurons)
				self.outputLayer.adjustWeights()
				
				avgError = avgError + self.outputLayer.getSumErrorSquared()/2
				errorForCurrEpoch = errorForCurrEpoch + avgError
			
			# Normalize the total error
			errorForCurrEpoch = errorForCurrEpoch/(trainingSet.getSize()*self.outputLayer.getSize())
			errorChange = 999
			if  e > 0:
				errorChange = (abs(errorForPrevEpoch - errorForCurrEpoch)/ abs(errorForPrevEpoch))*100
			
			errorForPrevEpoch = errorForCurrEpoch
			
			print('Epoch(',e,'/',numEpochs-1,')\nMean Sq. Error : ',errorForCurrEpoch,' | Error Change : ',errorChange)
			
			if errorChange	<= 0.0001:
				print('Error Change sufficiently low. Training Complete')
				break

def quickTest():
	# Tests the neural network by trying to 
	# teach it to multiply two numbers 
	
	nn = Network(2,6,1);

	t = TrainingSet(2,1)

	for i in range(1000):
		a = random.random()
		b = random.random()
		c = a * b
		t.add([a,b],[c])
		
	nn.randomizeWeights()
	nn.train(t,1000)
	nn.fireForInput([0.2,0.4])
	

if __name__ == "__main__":
	quickTest()