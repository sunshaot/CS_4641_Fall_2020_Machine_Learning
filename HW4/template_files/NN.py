import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

'''
We are going to use Breast Cancer Wisconsin (Diagnostic) Data Set provided by sklearn
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
to train a 2 fully connected layer neural net. We are going to buld the neural network from scratch. 
'''


class dlnet:

    def __init__(self, x, y, lr = 0.003):
        '''
        This method inializes the class, its implmented for you. 
        Args:
            x: data
            y: labels
            Yh: predicted labels
            dims: dimensions of different layers
            param: dictionary of different layers parameters
            ch: Cache dictionary to store forward parameters that are used in backpropagation
            loss: list to store loss values
            lr: learning rate
            sam: number of training samples we have

        '''        
        self.X=x # features
        self.Y=y # ground truth labels

        self.Yh=np.zeros((1,self.Y.shape[1])) # estimated labels
        self.dims = [30, 15, 1] # dimensions of different layers

        self.param = { } # dictionary for different layer variables
        self.ch = {} # cache for holding variables during forward propagation to use them in back prop
        self.loss = []

        self.lr=lr # learning rate
        self.sam = self.Y.shape[1] # number of training samples we have
        self._estimator_type = 'classifier'
        self.neural_net_type = "Relu -> Sigmoid" #can change it to "Tanh -> Sigmoid" 

    def nInit(self): 
        '''
        This method initializes the neural network variables, its already implemented for you. 
        Check it and relate to mathematical the description above.
        You are going to use these variables in forward and backward propagation.
        '''   
        np.random.seed(1)
        self.param['theta1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
        self.param['b1'] = np.zeros((self.dims[1], 1))        
        self.param['theta2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
        self.param['b2'] = np.zeros((self.dims[2], 1))                
        
    

    def Relu(self, u):
        '''
        In this method you are going to implement element wise Relu. 
        Make sure that all operations here are element wise and can be applied to an input of any dimension. 
        Input: u of any dimension
        return: Relu(u) 
        '''
        #TODO: implement this

        return np.maximum(u, 0)


    def Sigmoid(self, u): 
        '''
        In this method you are going to implement element wise Sigmoid. 
        Make sure that all operations here are element wise and can be applied to an input of any dimension. 
        Input: u of any dimension
        return: Sigmoid(u) 
        '''
        #TODO: implement this

        return 1 / (1 + np.exp(-u))


    def Tanh(self, u):
        '''
        In this method you are going to implement element wise Tanh. 
        Make sure that all operations here are element wise and can be applied to an input of any dimension. 
        Input: u of any dimension
        return: Tanh(u) 
        '''
        #TODO: implement this

        return (np.exp(u) - np.exp(-u)) / (np.exp(u) + np.exp(-u))
    
    
    def dRelu(self, u):
        '''
        This method implements element wise differentiation of Relu.  
        Input: u of any dimension
        return: dRelu(u) 
        '''
        u[u<=0] = 0
        u[u>0] = 1
        return u


    def dSigmoid(self, u):
        '''
        This method implements element wise differentiation of Sigmoid.  
        Input: u of any dimension
        return: dSigmoid(u) 
        '''
        o = 1/(1+np.exp(-u))
        do = o * (1-o)
        return do

    def dTanh(self, u):
        '''
        This method implements element wise differentiation of Tanh. 
        Input: u of any dimension
        return: dTanh(u) 
        '''
        
        o = np.tanh(u)
        return 1-o**2
    
    
    def nloss(self,y, yh):
        '''
        In this method you are going to implement Cross Entropy loss. 
        Refer to the description above and implement the appropriate mathematical equation.
        Input: y 1xN: ground truth labels
               yh 1xN: neural network output after Sigmoid

        return: CE 1x1: loss value 
        '''
        
        #TODO: implement this

        return - (y @ np.log(yh).T + (1 - y) @ np.log(1 - yh).T) / self.sam


    def forward(self, x):
        '''
        Fill in the missing code lines, please refer to the description for more details.
        Check nInit method and use variables from there as well as other implemeted methods.
        Refer to the description above and implement the appropriate mathematical equations.
        do not change the lines followed by #keep. 
        '''  
       
        #Todo: uncomment the following 7 lines and complete the missing code
        self.ch['X'] = x #keep
        u1 = self.param['theta1'] @ x + self.param['b1'] 
        o1 = self.Relu(u1)
        self.ch['u1'], self.ch['o1'] = u1, o1 #keep 
        u2 = self.param['theta2'] @ o1 + self.param['b2']  
        o2 = self.Sigmoid(u2)
        self.ch['u2'], self.ch['o2'] = u2, o2 #keep
            
        return o2 #keep
    

    def backward(self, y, yh):
        '''
        Fill in the missing code lines, please refer to the description for more details
        You will need to use cache variables, some of the implemeted methods, and other variables as well
        Refer to the description above and implement the appropriate mathematical equations.
        do not change the lines followed by #keep.  
        '''    
        #TODO: implement this 
            
        dLoss_o2 = - (np.divide(y, yh ) - np.divide(1 - y, 1 - yh))   # partial l by partial o2  
            
        #Implement equations for getting derivative of loss w.r.t u2, theta2 and b2
        # set dLoss_u2, dLoss_theta2, dLoss_b2
        dLoss_u2 = dLoss_o2 * (yh * (1 - yh))
        dLoss_theta2 = dLoss_u2 @ self.ch['o1'].T
        dLoss_theta2 = dLoss_theta2 / self.ch['o1'].shape[1]
        dLoss_b2 = dLoss_u2 @ np.ones((1, dLoss_u2.shape[1])).T
        dLoss_b2 = dLoss_b2 / self.ch['o1'].shape[1]
        
        dLoss_o1 = np.dot(self.param["theta2"].T,dLoss_u2) # partial l by partial o1
       
        
        #Implement equations for getting derivative of loss w.r.t u1, theta1 and b1
        # set dLoss_u1, dLoss_theta1, dLoss_b1
        dLoss_u1 = dLoss_o1 * self.dRelu(self.ch['u1'])
        dLoss_theta1 = dLoss_u1 @ self.X.T
        dLoss_theta1 = dLoss_theta1 / self.X.shape[1] 
        dLoss_b1 = dLoss_u1 @ np.ones((1, dLoss_u1.shape[1])).T
        dLoss_b1 = dLoss_b1 / self.X.shape[1] 

        #parameters update, no need to change these lines
        self.param["theta2"] = self.param["theta2"] - self.lr * dLoss_theta2 #keep
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2 #keep
        self.param["theta1"] = self.param["theta1"] - self.lr * dLoss_theta1 #keep
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1 #keep
        return dLoss_theta2, dLoss_b2, dLoss_theta1, dLoss_b1


    def gradient_descent(self, x, y, iter = 60000):
        '''
        This function is an implementation of the gradient decent algorithm 
        '''    
        #Todo: implement this 
        self.nInit()
        for i in range(iter):
            o2 = self.forward(x)
            loss = self.nloss(y, o2)
            self.backward(y, o2)
            self.loss.append(loss)
            if i % 2000 == 0:
                print("Loss after iteration %i: %f" % (i, loss))
    
    #bonus for undergraduate students 
    def stochastic_gradient_descent(self, x, y, iter = 60000):
        '''
        This function is an implementation of the stochastic gradient decent algorithm

        Note: 
        1. SGD loops over all examples in the dataset one by one and learns a gradient from each example 
        2. One iteration here is one round of forward and backward propagation on one example of the dataset. 
           So if the dataset has 1000 examples, 1000 iterations will constitute an epoch 
        3. Append loss after every 2000 iterations for plotting loss plots
        4. It is fine if you get a noisy plot since learning on one example at a time adds variance to the 
           gradients learnt
        5. You can use SGD with any neural net type 
        '''
        
        #Todo: implement this 
        self.nInit()
        for i in range(iter):
            X = (x[:,0]).reshape((x[:,0].shape[0], 1))
            o2 = self.forward(X)
            Y = np.ones((1,1))
            Y *= y[0][0]
            loss = self.nloss(Y, o2)
            self.backward(Y, o2)
            if i % 2000 == 0:

                
                self.loss.append(loss)
                print("Loss after iteration %i: %f" % (i, loss))
                        
    def predict(self, x): 
        '''
        This function predicts new data points
        Its implemented for you
        '''
        Yh = self.forward(x)
        return np.round(Yh).squeeze()
