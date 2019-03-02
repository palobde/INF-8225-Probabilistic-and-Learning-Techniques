"""
Created on Fri Feb  2 14:44:04 2018

@author: palobde
"""

import numpy as np
import matplotlib
import sklearn
import matplotlib.pyplot as plt
from sklearn import datasets
import math

digits = datasets.load_digits()

X = digits.data

y = digits.target

y_one_hot = np.zeros((y.shape[0], len(np.unique(y))))
y_one_hot[np.arange(y.shape[0]), y]=1  # one hot target or shape NxK

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size =0.3, random_state=42)


X_test , X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


W = np.random.normal(0, 0.01, (len(np.unique(y)), X.shape[1])) # weights of shape KxL
Theta = np.concatenate((W,np.ones((W.shape[0],1))),axis=1)

# Parameters to change
lr = 0.01
nb_minibatches = 20



# --- COMPUTATION ---
minibatch_size = len(y) // nb_minibatches
losses = []
accuracies = []
best_W = W
best_Theta = None
best_accuracy = 1E9
nb_epochs = 500
losses = []
accuracies = []


for epoch in range(nb_epochs): # For each iteration t (epoch)
    
    loss = 0
    accuracy = 0
    nb_batch = 0 
    
    for i in range (0, X_train.shape[0], minibatch_size) :
        grads = 0
        nb_batch = nb_batch + 1
        X = X_train[i:i+minibatch_size]
        y = y_train[i:i+minibatch_size]
        X_1 = np.concatenate((X,np.ones((X.shape[0],1))),axis=1)
        Xt=X_1.T
        y_pred = np.dot(Theta,Xt).T
                       
        #Compute the average gradient or the set
        grads = get_grads(y,y_pred,X)
                       
        
        # update the loss on the train set
        batch_loss = get_loss(y, y_pred)
        loss = loss +  batch_loss
            
        # Update Theta (and W):
        Theta = Theta + lr*grads
    
    loss = loss/nb_batch
        
    # Record total loss on the train set 
#    print('Losses = ',loss)
    losses.append(loss) 
    
        
    # compute the accuracy on the validation set
#    W = np.array(Theta[:,0:-1])
    accuracy = get_accuracy(X_validation, y_validation, Theta) 
#    print('Accuracy = ', accuracy)
    accuracies.append(accuracy) 
        
    # select the best parameters based on the validation accuracy
    if best_accuracy > accuracy:
        best_Theta = Theta
        best_accuracy=accuracy
        
# Compute Test Accuracy
accuracy_on_unseen_data = get_accuracy(X_test, y_test, best_Theta)

print('Results for Learning Rate = ',lr, 'And Mini batch size = ',nb_minibatches)
print('Test accuracy = ',accuracy_on_unseen_data) 

losses=np.array(losses)
accuracies=np.array(accuracies)

plt.figure()
Loss, = plt.plot(losses, label="Losses (Training)")
Accuracy, = plt.plot(accuracies, label="Accuracy (Validation)")
plt.legend(handles=[Accuracy, Loss])
plt.ylabel('Average negative log likelihood')
plt.xlabel('Epoch')


W = np.array(best_Theta[:,0:-1])
plt.figure()
plt.imshow(best_W[4,:].reshape(8,8)) 
plt.title("Learned Weights",fontsize=16)




# Function to Compute softMax value
def softmax(x):
    if x.shape[0]==np.transpose(x).shape[0]:
        x=np.array([x])
    z = np.divide(np.exp(x),np.sum(np.exp(x),axis=1).reshape(x.shape[0],1)) 
    return np.amax(z,axis=1)
    
                                                    
# Function to compute Loss
def get_loss(y, y_pred):
    prob = softmax(y*y_pred)
    log_prob = -np.log(prob)
    loss = np.mean(log_prob)
    return loss

# Function to compute Accuracy
def get_accuracy(X, y, Theta):
    X_1 = np.concatenate((X,np.zeros((X.shape[0],1))),axis=1) # ne pas tenir compte du b...
    Xt=X_1.T
    y_pred = np.dot(Theta,Xt).reshape(X.shape[0],Theta.shape[0])  
    accuracy = get_loss(y, y_pred)
    return accuracy

# Function to compute Gradients for One batch
def get_grads(y,y_pred,X):
    Grads = np.zeros((y.shape[1],X.shape[1]+1))
    batch_size=X.shape[0]
    
    # For each xi in the batch
    for i in range(batch_size):
        
        X_i = np.array(X[i])
        X_i = np.concatenate((X_i,np.array([1])))
        y_i = np.array(y[i])
        y_pred_i = np.array(y_pred[i])    
        prob = softmax(y_i*y_pred_i)
        y_i = y_i.reshape(y_i.shape[0],1)
        y_pred_i = y_pred_i.reshape(y_i.shape[0],1)
        X_i = X_i.reshape(1,X_i.shape[0])     
        
        grad = np.dot(y_i,X_i) - np.dot(y_pred_i,X_i)*prob
           
        Grads = Grads+grad 
    
    # Get the average    
    Grads = Grads/batch_size
    return Grads