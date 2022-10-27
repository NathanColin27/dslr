import pandas as pd
import numpy as np
from data.data import Data



class LogisticRegression:
    """
        A class to perform Logistic Regression
    """

    def __init__(self):
        """
            Define attributes which will be passed later
        """
        # train data and label
        self.X = None
        self.y = None

        # m: number of observations
        self.m = None

        # n: number of independent variables (X)
        self.n = None

        # store historic value of cost function. init as +infinity
        self.costs = [np.inf]

        self.J_history = []
        self.w_history = []

        # parameteres (weights)
        self.w = None
        self.b = None

        # regularization constant lambda_ (scalar, float)
        self.lambda_ = None

    def save_weights(self):
        return {'w': self.w.tolist(), 'b': self.b}

    def sigmoid(self, z):
        """
        Compute the sigmoid of z
        Args:
            z (ndarray): A scalar, numpy array of any size.
        Returns:
            g (ndarray): sigmoid(z), with the same shape as z
        """
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y, w, b):
        """
        Computes the cost over all examples
        Args:
          X : (ndarray Shape (m,n)) data, m examples by n features
          y : (array_like Shape (m,)) target value 
          w : (array_like Shape (n,)) Values of parameters of the model      
          b : scalar Values of bias parameter of the model
          lambda_: unused placeholder
        Returns:
          total_cost: (scalar)         cost 
        """

        m, n = X.shape

        z = np.dot(X, w) + b
        pred = self.sigmoid(z)

        pred[pred == 1] = 1-1e-9  # hard cap max threshold

        cost = np.dot(-y, np.log(pred)) - (np.dot(1 - y, np.log(1 - pred)))
        total_cost = np.sum(cost) / m

        reg_cost = np.sum(np.dot(w, w.T))
        total_cost = total_cost + (self.lambda_/(2 * m)) * reg_cost

        return total_cost

    def compute_gradient(self, X, y, w, b, lambda_):
        """
        Computes the gradient for logistic regression 

        Args:
          X : (ndarray Shape (m,n)) variable such as house size 
          y : (array_like Shape (m,1)) actual value 
          w : (array_like Shape (n,1)) values of parameters of the model      
          b : (scalar)                 value of parameter of the model 
          lambda_: unused placeholder.
        Returns
          dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w. 
          dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b. 
        """
        m, n = X.shape
        dj_dw = np.zeros(w.shape)
        dj_db = 0.0

        z = np.dot(X, w) + b
        f_wb = self.sigmoid(z)

        for j in range(n):
            dj_dw[j] = (np.sum(np.dot(f_wb - y, X.T[j])))

        dj_dw = dj_dw / m
        dj_dw += np.dot((lambda_ / m), w)  # add regularization
        dj_db = np.sum(f_wb - y) / m

        return dj_db, dj_dw

    def gradient_descent(self, cost_function, gradient_function, alpha, num_iters, show_every):
        """
        Performs batch gradient descent to learn theta. Updates theta by taking 
        num_iters gradient steps with learning rate alpha

        Args:
          X :    (array_like Shape (m, n)
          y :    (array_like Shape (m,))
          w_in : (array_like Shape (n,))  Initial values of parameters of the model
          b_in : (scalar)                 Initial value of parameter of the model
          cost_function:                  function to compute cost
          alpha : (float)                 Learning rate
          num_iters : (int)               number of iterations to run gradient descent
          lambda_ (scalar, float)         regularization constant

        Returns:
          w : (array_like Shape (n,)) Updated values of parameters of the model after
              running gradient descent
          b : (scalar)                Updated value of parameter of the model after
              running gradient descent
        """

        w_in = self.w
        b_in = self.b
        for i in range(num_iters):

            # Calculate the gradient and update the parameters
            dj_db, dj_dw = gradient_function(
                self.X, self.y, w_in, b_in, self.lambda_)

            # Update Parameters using w, b, alpha and gradient
            w_in = w_in - alpha * dj_dw
            b_in = b_in - alpha * dj_db

            # Save cost J at each iteration
            if i < 100000:      # prevent resource exhaustion
                cost = cost_function(self.X, self.y, w_in, b_in)
            # Print cost every at intervals 10 times or as many iterations if < 10
            if i % show_every == 0 or i == num_iters-1:
                self.J_history.append(cost)
#                 self.w_history.append(self.w)
                print(
                    f"Iteration {i:4}: Cost {float(self.J_history[-1]):8.2f}   ")
        self.w = w_in
        self.b = b_in

    def fit(self, X, y, alpha=0.001, iterations=1500, show_every=None, lambda_=1):
        """
        setup attributes and apply training
        """

        # train data and label
        self.X = X
        self.y = y

        # m: number of observations
        self.m, self.n = X.shape

        # regularization coefficient
        self.lambda_ = lambda_

        # init weights if first call
        if type(self.w) is not np.ndarray:
            self.w = 0.01 * (np.random.rand(self.n).reshape(-1, 1) - 0.5)
        if self.b == None:
            self.b = -8
        if show_every == None:
            show_every = iterations // 10

        # Perform Gradient Descent
        self.gradient_descent(
            self.compute_cost, self.compute_gradient, alpha, iterations, show_every)

    def predict(self, X, decision_boundary=None):
        """
        Predict whether the label is 0 or 1 using learned logistic
        regression parameters w

        Args:
        X : (ndarray Shape (m, n))
        w : (array_like Shape (n,))      Parameters of the model
        b : (scalar, float)              Parameter of the model

        Returns:
        p: (ndarray (m,1))
            The predictions for X using a threshold at 0.5
        """

        # Check if number of features matches our model

        # number of training examples
        m, n = X.shape
        p = np.zeros(m)

        f_wb = self.sigmoid(np.dot(X, self.w) + self.b)

        if decision_boundary != None:
            for i, prob in enumerate(f_wb):
                p[i] = 1 if prob >= decision_boundary else 0
            return p

        return f_wb


class MultipleLogisticRegression:

    def __init__(self):

        self.X = None
        self.y = None

        # C: number of categories for Y
        # -> create C models and train them each one at teh time
        self.c = None

        self.models = []

    def save_weights(self):
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        weights = {}
        for index, model in enumerate(self.models):
            weights[houses[index]] = model.save_weights()
        return {'weights': weights}

    def softmax(self, X):
        """
        takes the predicted values from the sub_models
        """
        pred = []
        for i in range(self.c):
            pred.append(self.models[i].predict(X))
        pred = np.array(pred)
        res = []
        for row in pred.T[0]:
            # for each row (student), return the column (house) with the highest probability
            res.append(np.argmax(row / np.sum(row)))
        res = np.array(res)
        return res

    def accuracy(self, pred, true_labels):
        """
        calculate the accuracy of the model prediction given the true labels
        """
        if pred.shape != true_labels.shape:
            print('Error in the shape of predictions and true_labels')
            return
        accuracy = np.mean(pred == true_labels) * 100
        print('Accuracy: %f' % (accuracy))
        return accuracy

    def score_matrix(self, pred, true_labels):
        """
        calculate scores matrix
        """
        # Calculate overall precision
        score_matrix = np.zeros((self.c, self.c))
        for i in range(0, len(pred)):
            pred_label = pred[i]
            true_label = true_labels[i]
            score_matrix[pred_label][true_label] += 1

        precision_per_class = []
        for i in range(self.c):
            precision_per_class.append(
                score_matrix[i][i] / score_matrix[i].sum())

        return score_matrix, precision_per_class

    def fit(self, X, y, alpha=0.00003, iterations=100, show_every=None, lambda_=1):
        """
        setup attributes and apply training
        """

        # train data and label
        self.X = X
        self.y = y

        # m: number of observations
        self.m, self.n = X.shape

        # c: number of category
        self.c = y.shape[1]

        if show_every == None:
            show_every = iterations // 10

        # init models
        if not self.models:
            for i in range(self.c):
                self.models.append(LogisticRegression())

        # train models
        decision_boundary = 0.5
        for i in range(self.c):
            model = self.models[i]
            print(f"training model {i+1} with alpha= {0.00003}")
            model.fit(X, y[:, i], alpha, iterations, lambda_=1)
            p = model.predict(X, decision_boundary)
            print('Train Accuracy: %f' % (np.mean(p == y[:, i]) * 100))

    def predict(self, X):
        return self.softmax(X)
