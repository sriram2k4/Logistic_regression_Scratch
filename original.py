import numpy as np # for scientific computation 
import pandas as pd # for working with data
from sklearn.model_selection import train_test_split # for splitting the data
from sklearn.datasets import load_breast_cancer # dataset that we will be using 
import warnings # optional 
warnings.filterwarnings( "ignore" ) # optional 

def sigmoid(z): 
    '''
    input:  
        z : a scalar or an array 
    output: 
        h : sigmoid of z 
    '''
    h = 1 / (1 + np.exp(-z))
    
    return h

def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        theta: your final weight vector
    '''
    m = len(y)
    
    for i in range(0, num_iters):
        
        # get z, the dot product of x and theta
        z =  np.dot(x, theta) 
        
        # get the sigmoid of z
        h = sigmoid(z) 
        
        # calculate the cost function
        J = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h))) 

        # update the weights theta
        theta =  theta -  alpha/m * (np.dot(x.T, (h-y))) 
        
    return theta

data = load_breast_cancer()  
# print(data.DESCR)

X = data.data  
y = data.target 

X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 0.3, random_state=0 )

### Now, we will train our model with gradient descent ### 
theta = gradientDescent(X_train, Y_train, np.zeros((30)), 0.001, 700)
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

def predictions(X, theta): 
    '''
    input: 
        X : input  
        theta : feature weights 
    output: 
        Y : 0 Or 1 
    '''
    Z = 1 / ( 1 + np.exp( - ( X.dot( theta )) ) )        
    Y = np.where( Z > 0.5, 1, 0 )        
    return Y

y_pred = predictions(X_test, theta) ### Predicting on the test set 

def test_logistic_regression(test_x, test_y, theta):
    y_hat = [] # making an empty list 
    for x in test_x:
        # get the label prediction 
        y_pred = predictions(x, theta) 
        
        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1)
        else:
            # append 0 to the list
            y_hat.append(0)
    accuracy = (y_hat==np.squeeze(test_y)).sum()/len(test_x)
    
    return accuracy

accuracy = test_logistic_regression(X_test, Y_test, theta)
print(f"Logistic regression model's accuracy = {accuracy:.4f}")