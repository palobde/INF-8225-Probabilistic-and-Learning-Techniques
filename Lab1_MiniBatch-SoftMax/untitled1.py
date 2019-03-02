# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 08:47:13 2018

@author: palobde
"""

        




for epoch in range(nb_epochs):
    losses = 0
    accuracy = 0
    
    for i in range (0, X_train.shape[0], minibatch_size) :
        pass # TODO
        losses.append(loss ) # compute the loss on the train set
        accuracy = None # TODO
        accuracies.append(accuracy) # compute the accuracy on the validation set
        if accuracy > best_accuracy:
            pass # select the best parameters based on the validation accuracy

accuracy_on_unseen_data = get_accuracy(X_test, y_test, best_W )
print(accuracy_on_unseen_data) # 0. 8 9 7 5 0 6 9 2 5 2 0 8

plt.plot(losses)

plt.imshow(best_W[4,:].reshape(8,8)) 




"""        print('sf :  ', x_i)
        
    z = [np.exp(x_i)/sum(np.exp(x_i)) for x_i in x]
    z=z[0]
    
    
    z = np.zeros(x.shape)   
        
    for i in range(x.shape[0]):
        somme = sum(np.exp(x[i])
        z[i] = np.exp(x[i])/somme
    
    
    
    
    """


def get_loss(y, y_pred):
#    loss = -np.dot(y,np.log(y_pred).reshape(y.shape[1],y.shape[0]))
    loss = -sum(sum(y*np.log(y_pred)))
    return loss


ex = np.array([[1, 2, 3],[15, 2, 8]])
print('voici: ',softmax(ex))

def softmax(x):
    if x.shape[0]==np.transpose(x).shape[0]:
        x=np.array([x])
    z = np.divide(np.exp(x),np.sum(np.exp(x),axis=1).reshape(x.shape[0],1))       
    z = np.array(z)
    return z



def get_grads(y,y_pred,X):
    Grads = np.zeros((y.shape[1],X.shape[1]))
    
    for i in range(X.shape[0]):
        X_i = np.array(X[i])
        y_i = np.array(y[i])
        y_pred_i = np.array(y_pred[i])    
        prob = softmax(y_i*y_pred_i)
        prob = prob.reshape(y_i.shape[0],1)
        y_i = y_i.reshape(y_i.shape[0],1)
        y_pred_i = y_pred_i.reshape(y_i.shape[0],1)
        X_i = X_i.reshape(1,X_i.shape[0])        
           
        grad = np.dot(y_pred_i,X_i) - np.dot(prob, np.average(np.dot(y_i,X_i),axis=1).reshape(1,X_i.shape[1]))
        Grads = Grads+grad 
    Grads = Grads/X.shape[0]
    return Grads


def get_loss(y, y_pred):
    prob = softmax(y*y_pred)
    log_prob = -y*np.log(prob)
    loss = np.average(log_prob,axis=1)
    loss = np.average(loss)
    return loss


"""
        # select the best parameters based on the validation accuracy
        if best_loss > batch_loss:
            best_loss = batch_loss
            best_W2 = W  """
            
            