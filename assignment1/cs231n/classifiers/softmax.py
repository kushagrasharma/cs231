from builtins import range
import numpy as np
from random import shuffle, randrange
from past.builtins import xrange

def softmax(x, i):
    x -= x.max()
    return np.exp(x[i] - x.max()) / np.sum(np.exp(x - x.max()))

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    preds = np.matmul(X, W)

    for i in range(X.shape[0]):
        pred = preds[i]
        total_sum = np.sum(np.exp(pred - pred.max()))
        class_sum = np.exp(pred[y[i]] - pred.max())
        L_i = class_sum / total_sum
        loss += -1 * np.log(L_i)

        dW_i = np.zeros_like(W)

        for m in range(W.shape[0]):
            dW_i[m, y[i]] += total_sum * X[i, m] * class_sum
            for n in xrange(W.shape[1]):
                dW_i[m, n] -= class_sum * X[i, m] * np.exp(pred[n] - pred.max())

        dW_i /= (total_sum ** 2)

        dW_i /= L_i

        dW += dW_i

    loss /= X.shape[0]
    loss += reg * np.sum(W ** 2)
    dW /= -1 * X.shape[0]
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

def compare_gradients(W, W1, num_checks=10):
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in W.shape])

        grad_numerical = W[ix]
        grad_analytic = W1[ix]
        rel_error = (abs(grad_numerical - grad_analytic) /
                    (abs(grad_numerical) + abs(grad_analytic)))
        print('G1: %f G2: %f, relative error: %e'
              %(grad_numerical, grad_analytic, rel_error))


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    preds = np.matmul(X, W)

    max_score = np.max(preds, 1) # should return Nx1 vector
    normalized_preds = (preds.transpose() - max_score).transpose() # broadcast across
    normalized_preds = np.exp(normalized_preds)
    softmax_sums = np.sum(normalized_preds, 1)
    softmax_preds = ((normalized_preds.transpose()) / softmax_sums).transpose()

    class_softmax = -1 * np.log(softmax_preds[np.arange(len(softmax_preds)), y])

    loss += class_softmax.sum()
    loss /= X.shape[0]
    loss += reg * np.sum(W ** 2)

    ind = np.zeros_like(softmax_preds)
    ind[np.arange(X.shape[0]), y] = 1
    dW = X.T.dot(softmax_preds - ind)

    dW /= X.shape[0]
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
