from builtins import range
import numpy as np
from random import shuffle
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

    max_score = np.max(preds, 0) # should return Nx1 vector
    normalized_preds = preds - max_score # broadcast across
    normalized_preds = np.exp(normalized_preds)
    softmax_sums = np.sum(normalized_preds, 1)
    softmax_preds = ((normalized_preds.transpose()) / softmax_sums).transpose()

    class_softmax = softmax_preds[np.arange(len(softmax_preds)), y]
    class_softmax = -1 * np.log(class_softmax)

    loss += class_softmax.sum()
    loss /= X.shape[0]
    loss += reg * np.sum(W ** 2)

    # for i in range(X.shape[0]):
    #     pred = preds[i]
    #     total_sum = np.sum(np.exp(pred - pred.max()))
    #     class_sum = np.exp(pred[y[i]] - pred.max())
    #     L_i = class_sum / total_sum
    #     loss += -1 * np.log(L_i)

    #     dW_i = np.zeros_like(W)

    #     for m in range(W.shape[0]):
    #         dW_i[m, y[i]] += total_sum * X[i, m] * class_sum
    #         for n in xrange(W.shape[1]):
    #             dW_i[m, n] -= class_sum * X[i, m] * np.exp(pred[n] - pred.max())

    #     dW_i /= (total_sum ** 2)

    #     dW_i /= L_i

    #     dW += dW_i

    for i in range(X.shape[0]):
        dW_i = np.zeros_like(W)
        dW_i[:, y[i]] = 1
        dW_i = (dW_i.transpose() * softmax_sums[i] * (X[i] * normalized_preds[i, y[i]])).transpose()

        # original unvectorized code for check
        pred = preds[i]
        total_sum = np.sum(np.exp(pred - pred.max()))
        class_sum = np.exp(pred[y[i]] - pred.max())

        dW_check = np.zeros_like(W)

        for m in range(W.shape[0]):
            dW_check[m, y[i]] += total_sum * X[i, m] * class_sum

        print("difference bw gradients: ", np.sum(dW_i - dW_check))
        return dW_i, dW_check

        # vectorized code

        summand = np.ones_like(W)
        summand *= normalized_preds[i, y[i]]
        summand = (summand.transpose() * X[i]).transpose()
        summand *= normalized_preds[i]
        dW_i -= summand
        dW_i /= (softmax_sums[i] ** 2)
        dW += dW_i

    dW /= -1 * X.shape[0]
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
