import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    fun_class_res = np.exp(np.dot(X[i],W))
    current_score = fun_class_res[y[i]]
    #temp_loss = 0
    for j in range(num_class):
        #temp_loss += np.exp(fun_class_res[j])
        if j==y[i]:
            dW[:,y[i]] += -X[i] + current_score / np.sum(fun_class_res)*X[i]
        else:
            dW[:,j] += fun_class_res[j]/np.sum(fun_class_res)*X[i]
    loss += (-np.log(current_score/np.sum(fun_class_res)))
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss /= num_train
  loss += reg*np.sum(W*W)

  dW /= num_train
  dW += 2*reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  res_dot = np.dot(X,W)
  res_dot_exp = np.exp(res_dot)
  loss = np.sum(-np.log((res_dot_exp[range(num_train),list(y)].reshape(-1,1)/np.sum(res_dot_exp,axis = 1).reshape(-1,1))))
  loss /= num_train
  loss += reg*np.sum(W*W)
  
 # res_dot_exp_copy = res_dot_exp
  #res_dot_exp_copy[range(num_train),list(y)]-=1
  dTemp = res_dot_exp/np.sum(res_dot_exp,axis = 1).reshape(-1,1)
  dTemp[range(num_train),list(y)]-=1
  dW = np.dot(X.T,dTemp)
  dW /= num_train
  dW += 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

