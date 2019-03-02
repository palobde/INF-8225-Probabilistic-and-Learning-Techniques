# -*- coding: utf-8 -*-
"""

@author: Gilles Eric Zagre
"""

# Importations
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

# Initialisation
digits = datasets.load_digits()

X = digits.data
# Asustuce du X=[X 1].T (pour le b)
X = np.insert(X,0,1, axis=1) 

y = digits.target
y_one_hot = np.zeros((y.shape[0], len(np.unique(y))))
y_one_hot[np.arange(y.shape[0]), y] = 1  # one hot target or shape NxK


X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)

X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


W = np.random.normal(0.0001, 0.01, (len(np.unique(y)), X.shape[1]))  # weights of shape KxL  : Initialize the weights

best_W = None
best_accuracy = 0
lr = 0.1
nb_epochs = 500
minibatch_size = 2#len(y) // 20

losses = []

losses_validation = []
losses_test = []
losses_train = []

accuracies_validation = []
accuracies_test = []
accuracies_train = []


def softmax(x):
    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def get_loss(X, y, W):
    return np.mean((-1) * np.sum(y * np.log(softmax(np.dot(X , W.T)))))
    
def get_accuracy(X, y, W):
    return np.sum(np.argmax(softmax(np.dot(X , W.T)), axis=1) == np.argmax(y, axis=1))/(float(len(y)))

def get_grads(y, y_pred, X):
    return X.reshape((1, X.size)) * (y - y_pred).reshape((len(y),1))
    
           

for epoch in range(nb_epochs):
    loss = 0
    accuracy = 0
    for i in range(0, X_train.shape[0], minibatch_size):
        
        batch_gradient = np.random.normal(0, 0.01, (len(np.unique(y)), X.shape[1]))
        
        for j in range(i, min(i + minibatch_size, X_train.shape[0])):
            h_theta = softmax((W * X_train[j]).sum(1))   
            gradient = get_grads(y_train[j], h_theta, X_train[j])   
            batch_gradient += gradient
            loss += get_loss(X_train[j], y_train[j], W)
        
        loss /= minibatch_size
        batch_gradient /= minibatch_size
        W = W + lr * batch_gradient
        
    accuracy = get_accuracy(X_validation, y_validation, W)
    accuracies_validation.append(accuracy)
    accuracies_test.append(get_accuracy(X_test, y_test, W))
    accuracies_train.append(get_accuracy(X_train, y_train, W))
    
    losses_validation.append(get_loss(X_validation, y_validation, W))
    losses_test.append(get_loss(X_test, y_test, W))
    losses_train.append(get_loss(X_train, y_train, W))
    
    print("Epoch {} \t:::\tval_acc : {}\tval_loss : {} ".format(epoch,accuracies_validation[-1], losses_validation[-1]))
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_W = W
    
accuracy_on_unseen_data = get_accuracy(X_test, y_test, best_W)

print(" Test data :: Accuracy = {}".format(accuracy_on_unseen_data))

plt.subplot(1, 3, 1)
accuracy, = plt.plot(accuracies_validation, label="Accuracy")
loss, = plt.plot(losses_validation , label="Loss")
plt.legend(handles=[accuracy, loss])
plt.ylabel('Average negative log likelihood', fontsize=18)
plt.title("Validatiaon")
plt.subplot(1, 3, 2)
accuracy, = plt.plot(accuracies_test, label="Accuracy")
loss, = plt.plot(losses_test, label="Loss")
plt.legend(handles=[accuracy, loss])
plt.xlabel('Epoch', fontsize=18)
plt.title("Test ")
plt.subplot(1, 3, 3)
accurary, = plt.plot(accuracies_train, label="Accuracy")
loss, = plt.plot(losses_train, label="Loss")
plt.legend(handles=[accuracy, loss])
plt.title("Tain ")
plt.show()

plt.imshow(best_W[4, :].reshape(8,8))
