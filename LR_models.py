import pandas as pd
import numpy as np
from data.data import Data
import matplotlib.pyplot as plt
import json


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

        # store historic value of cost function.
        self.costs = []

        self.J_history = []
        self.w_history = []

        # parameteres (weights)
        # thetas[0] => b, thetas[1 to n] => w
        self.thetas = None

        # regularization constant lambda_ (scalar, float)
        self.lambda_ = None

    def initialize_thetas(self):
        """
        Initialize thetas (weights) where 1st entry is the b (intercept) and the rest w (therefore size n+1)
        X is concatenated with an np.ones of shape (m, 1) for the b (intercept)
        """
        self.thetas = np.zeros((self.n + 1, 1))


    def save_weights(self):
        return {'b': self.thetas[0].tolist(), 'w': self.thetas[1:].tolist()}

    def load_weights(self, weights):
        self.thetas = np.reshape(weights, (-1, 1))
        self.n = len(weights) - 1

    def sigmoid(self, z):
        """
        Compute the sigmoid of z
        Args:
            z (ndarray): A scalar, numpy array of any size.
        Returns:
            g (ndarray): sigmoid(z), with the same shape as z
        """
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y, thetas):
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

        z = np.dot(X, thetas)
        pred = self.sigmoid(z)        
        
        cost = np.dot(-y.T, np.log(pred)) - (np.dot( (1-y).T, np.log(1 - pred)))
        total_cost = cost / m
    
        # Regularization
        # reg_cost = np.sum(np.dot(w, w.T))
        # total_cost = total_cost + (self.lambda_/(2 * m)) * reg_cost
        
        return total_cost

    def compute_gradient(self, X, y, thetas, lambda_):
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
        dj = np.zeros(thetas.shape)

        z = np.dot(X, thetas)
        f_wb = self.sigmoid(z)

        dj = np.dot(X.T, f_wb - np.reshape(y,(len(y),1)) )
        
        return dj

    def gradient_descent(self, cost_function, gradient_function, alpha, num_iters, show_every):
        """
        Performs batch gradient descent to learn theta. Updates theta by taking 
        num_iters gradient steps with learning rate alpha

        Args:
          cost_function:                  function to compute cost
          alpha : (float)                 Learning rate
          num_iters : (int)               number of iterations to run gradient descent
          lambda_ (scalar, float)         regularization constant
        """

        for i in range(num_iters):

            # Calculate the gradient and update the parameters
            dj = gradient_function(self.X, self.y, self.thetas, self.lambda_)   

            # Update Parameters using w, b, alpha and gradient
            self.thetas = self.thetas - alpha * dj

            # Save cost J at each iteration
            if i<100000:      # prevent resource exhaustion 
                cost =  cost_function(self.X, self.y, self.thetas)
                self.J_history.append(float(cost))

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i % show_every == 0 or i == num_iters-1:
                # print(float(cost))
                # self.w_history.append(self.w)
                print(f"Iteration {i:4}: Cost {float(self.J_history[-1]):8.2f}   ")

    def fit(self, X, y, alpha=0.001, iterations=1500, show_every=None, lambda_= 1):
        """
        setup attributes and apply training
        """

        # m: number of observations
        self.m, self.n = X.shape
        
        # train data and label
        self.X = np.c_[np.ones((self.m, 1)), X]
        self.y = y

        # regularization coefficient
        self.lambda_ = lambda_

        # init weights if first call
        if type(self.thetas) is not np.ndarray:
            self.initialize_thetas()
        if show_every == None:
            if (iterations <= 10):
                show_every = 1
            else:
                show_every = iterations // 10 
            
        # Perform Gradient Descent
        self.gradient_descent(self.compute_cost, self.compute_gradient, alpha, iterations, show_every)

        return self.J_history

    def predict(self, X, decision_boundary = None): 
        """
        Predict whether the label is 0 or 1 using learned logistic
        regression parameters w

        Args:
        X : (ndarray Shape (m, n))

        Returns:
        p: (ndarray (m,1))
            The predictions for X using a threshold at 0.5
        """

        # Check if number of features matches our model

        # number of training examples
        m, n = X.shape
        p = np.zeros(m)

        X = np.c_[np.ones((m, 1)), X]
        f_wb = self.sigmoid(np.dot(X, self.thetas))

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

    def save_weights(self, filename, category_list):
        weights = {}
        for index, model in enumerate(self.models):
            weights[category_list[index]] = model.save_weights()
        with open(filename, "w") as outfile:
            json.dump(weights, outfile)

    def load_weights(self, filename):
        
        f = open(filename)
        data = json.load(f)  # returns JSON object as a dictionary
        
        # init models
        if not self.models:
            self.c = len(data)  # get number of category
            for i in range(self.c):
                self.models.append(LogisticRegression())

        for idx, category in enumerate(data):
            # concat to get the thetas for each model
            w = np.array(data[category]['w'])
            b = np.array(data[category]['b'])
            thetas = np.concatenate((b, w), axis=None)
            self.models[idx].load_weights(thetas)
            


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

    def accuracy(self, pred_labels, true_labels):
        """
        calculate the accuracy of the model prediction given the true labels
        """
        if pred_labels.shape != true_labels.shape:
            print('Error in the shape of predictions and true_labels')
            return
        accuracy = np.mean(pred_labels == true_labels) * 100
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
        self.cost_histories = []
        if show_every == None:
            if (iterations <= 10):
                show_every = 1
            else:
                show_every = iterations // 10 

        # init models
        if not self.models:
            for i in range(self.c):
                self.models.append(LogisticRegression())

         # train models
        decision_boundary = 0.5
        for i in range(self.c):
            model = self.models[i]
            print(f"training model {i+1} with alpha= {alpha}")
            self.cost_histories.append(model.fit(X, y[: , i], alpha=alpha, iterations=iterations, lambda_ = lambda_))
            p = model.predict(X, decision_boundary)
            print('Train Accuracy: %f'%(np.mean(p == y[:, i]) * 100))
            print()

        plt.figure(figsize=(30, 15))
        plt.suptitle(f"Logistic regression cost history by models (alpha = {alpha}, iterations = {iterations})", fontsize=18, y=0.95)
        for index,i in enumerate(self.cost_histories):
            ax = plt.subplot(2,2, index + 1)
            ax.plot(i)
            ax.set_title(f'model_{index}')

        plt.show()

    def predict(self, X):
        return self.softmax(X)
