
# coding: utf-8

# In[21]:


import string
import math
import random

class Neural:
	def __init__(self, pattern):
		#
		#lets take 2 input nodes, 3 hidden nodes and 1 output node.
		self.inputNode=3 #number of input nodes
        #introduce additional constant input value 1 and weight value -threshold. 
		self.hiddenNode=3 #number of hidden nodes
		self.outputNode=1 #number of output nodes
        
		#initialize two dimensional array for netwrok weights. It generate weight to connect layer node to next layer node
		self.wih = [] #array of weights from input to hidden layers
		for i in range(self.inputNode):
			self.wih.append([0.0]*self.hiddenNode) #initilaize weights with 0 at start
		self.who = [] #array of weights from hidden to output layer
		for j in range(self.hiddenNode):
			self.who.append([0.0]*self.outputNode) #initilaize weights with 0 at start
            
		#create activation function matrix for each layer and initilaize them with 1
		self.ai, self.ah, self.ao = [],[],[]
		self.ai=[1.0]*self.inputNode
		self.ah=[1.0]*self.hiddenNode
		self.ao=[1.0]*self.outputNode

		#assign random weight values to the connection call randomizeMatrix function some bounds on values
		randomizeMatrix(self.wih,-0.2,0.2) #random bound values
		randomizeMatrix(self.who,-2.0,2.0) #random bound values

	#backpropagate function defination it adjusts the weights according the the expected output and network output to minimize the error.
	def backpropagate(self, inputs, expected, output, N=0.2):
		#calculate error on output layer
		#introduce new matrix outputDeltas error for the output layer
		outputDeltas = [0.0]*self.outputNode
		for k in range(self.outputNode): #output nodes
			#error is equal to (Target value - Output value)
			error = expected[k] - output[k] #calculate error
			outputDeltas[k]=error*dsigmoid(self.ao[k]) #multiply error with differntiate sigmoid of output layer activations
		#update hidden to output layer weights
		for j in range(self.hiddenNode): #hidden nodes
			for k in range(self.outputNode): #output nodes
				deltaWeight = self.ah[j] * outputDeltas[k] #multiply hidden layer node activation with output layer node delta error
				self.who[j][k]+= N*deltaWeight #multiply weight error (delta weight) and learning rate then add it into previous weight

		#calculate error on hidden layer
		#introduce new matrix hiddenDeltas error for the hidden layer
		hiddenDeltas = [0.0]*self.hiddenNode
		for j in range(self.hiddenNode): #hidden nodes
			#error in hidden layer node is the sum of (hidden layer weights times output delta error of output node)
			error=0.0
			for k in range(self.outputNode): #output nodes
				error+=self.who[j][k] * outputDeltas[k] #sum of (each hidden layer node weight times output delta error of output node)
			hiddenDeltas[j]= error * dsigmoid(self.ah[j]) #multiply error with differntiate sigmoid of hidden layer activations
		#update input to hidden layer weights
		for i in range(self.inputNode):
			for j in range(self.hiddenNode):
				deltaWeight = hiddenDeltas[j] * self.ai[i] #multiply input layer node activation with hidden layer node delta error
				self.wih[i][j]+= N*deltaWeight #multiply weight error (delta weight) and learning rate then add it into previous weight

	#test function defination. To test the network after the training and Backpropagation is completed
	def test(self, patterns):
		for p in patterns:
			inputs = p[0]
			print ("For input:" , p[0] , " Output -->" , self.runNetwork(inputs) , "\tTarget: " , p[1])

#runNetwork fucntion defination. To run the network on specific set of input value
	def runNetwork(self, values):
		#check the number of values are equal to the number of input layer node        
		if(len(values)!=self.inputNode-1):
			print ("number of input values are not correct.")
            
		#activate the inputNodes with input values (inputNode-1) because of the additional input node for threshold
		for i in range(self.inputNode-1):
			self.ai[i]=values[i]
            
		#calculate activation functions of each nodes
		for j in range(self.hiddenNode): #hidden nodes
			sum=0.0
			for i in range(self.inputNode): #input nodes
				sum+=self.ai[i]*self.wih[i][j] #multiply the input value with their respective weight value and sum up the all values
			#call the sigmoid function for the actiation function of hidden layer nodes 
			self.ah[j]=sigmoid(sum)

		for k in range(self.outputNode): #output nodes
			sum=0.0
			for l in range(self.hiddenNode): #hidden nodes
				sum+=self.ah[l]*self.who[l][k] #multiply the activation value of hidden node with their respective weight value and sum up the all values
			#call the sigmoid function for the actiation function of output layer node
			self.ao[k]=sigmoid(sum)
		#return the activation fucntion value of output node
		return self.ao

#trainNetwork fucntion defination. To train the neural network
	def trainNetwork(self, pattern):
		for i in range(500):
			# Run the network for every set of input values, get the output values and Backpropagate them until satisfy the correct answers
			for p in pattern:
				# Run the network for every tuple in p.
				inputs = p[0] #select input values from pattern
				output = self.runNetwork(inputs) #run network for every input pair value in pattern
				expectedOutput = p[1] #expected output of the input pattern
				self.backpropagate(inputs,expectedOutput,output) #call backpropogate to update the weight values
		self.test(pattern) #test the input pattern wth updated weights

# End of class.

#randomizeMatrix fucntion defination. To generate random weight values
def randomizeMatrix ( matrix, a, b):
	for i in range ( len (matrix) ):
		for j in range ( len (matrix[0]) ):
			# For each connection in neural netwrok assign a random weight uniformly between the bound values
			matrix[i][j] = random.uniform(a,b)


#sigmoid function definition. To calculate the activation functions. You can change it to other activation functions like Relu etc
def sigmoid(x):
	return 1 / (1 + math.exp(-x))


#dsigmoid function definition. To calculate the derivative of the sigmoid function.
def dsigmoid(y):
	return y * (1 - y)

#main function
def main():
	# take the input pattern as a map. In the binary form to perform AND logic gate operation in 2D environment.
	pattern = [[[0,0], [0]],
		[[0,1], [0]],
		[[1,0], [0]],
		[[1,1], [1]]]
	neuralNetwork = Neural(pattern) #generate the basic structure of the network.
	neuralNetwork.trainNetwork(pattern) #train the network on the given input pattern


if __name__ == "__main__":
	main()

