class TrainingSet:
	
	def __init__(self,numInputs,numOutputs):
		self.numInputs = numInputs
		self.numOutputs = numOutputs
		self.inputSet = [[]]
		self.outputSet = [[]]
	
	def add(self,inputs,outputs):
		self.inputSet.append(inputs)
		self.outputSet.append(outputs)
	
	def getSize(self):
		return len(self.inputSet)

def trainingSetFromCSV(filename):
	with open(filename,'r') as inputFile:
		content = inputFile.read().splitlines()
	
	line = content[0]
	columnHeaders = line.split(',')
	numInputs = len(columnHeaders) - 1
	
	t = TrainingSet(numInputs,1)
	
	for i in range(1,numInputs - 1):
		line = content[i]
		inputs = line.split(',')
		t.add(inputs[1:],inputs[0:0])

	return t