import copy
import autograd.numpy as np
from tqdm import tqdm
from autograd import grad
from sklearn.metrics import accuracy_score

"""
Based on https://github.com/casperbh96/Logistic-Regression-From-Scratch
"""
class FairLogisticRegression():
    def __init__(self, fairness_metric="eo_sum", lam = 0.95):
        self.fairness_metric = fairness_metric
        self.losses = []
        self.train_accuracies = []
        
        #Tradeoff between fairness and accuracy
        self.lam = lam

    def fit(self, x, y, z, epochs):
        x = self._transform_x(x)
        y = self._transform_y(y)
        z = self._transform_y(z)

        self.weights = np.zeros(x.shape[1], dtype='float')
        self.bias = 0

        for i in range(epochs):
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            fair_error_w = self.fair_grad(self.weights, x.transpose(),y,z,self.fairness_metric)
            #print("difference_error: ", error_w-fair_error_w)
            error_w = (1-self.lam)*error_w + self.lam*fair_error_w
            
            self.update_model_parameters(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)

    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def compute_gradients(self, x, y_true, y_pred):
        # derivative of binary cross entropy
        difference =  y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b
    
    def spd(self, pred, protected_class, positive=True):
        """
        Equation: |P(Y_pred = y | Z = 1) - P(Y_pred = y | Z = 0)|
        Assumes that the positive class is the desired outcome and
            that the protected_class is 0/1 binary"""
        z_1 = [y_hat for y_hat, z in zip(
            pred, protected_class) if z == 1]
        z_0 = [y_hat for y_hat, z in zip(
            pred, protected_class) if z == 0]

        if not positive:
            z_1 = [0 if z == 1 else 1 for z in z_1]
            z_0 = [0 if z == 1 else 1 for z in z_1]
        """if len(z_1)+len(z_0)!=len(pred):
            print("NOT EQUAL")"""
        return abs(sum(z_1)-sum(z_0))
    
    def eo_sum(self, pred, prot, true):
        """
        Equation: |P(Y_pred = y_pred | Y_true = y_true, Z = 1) - P(Y_pred = y_pred | Y_true = y_true, Z = 0)|
        Assumes prot is 0/1 binary"""
        z1_y0 = [y_hat for y_hat, z, y in zip(
            pred, prot, true) if z == 1 and y == 0]
        z0_y0 = [y_hat for y_hat, z, y in zip(
            pred, prot, true) if z == 0 and y == 0]
        z1_y1 = [y_hat for y_hat, z, y in zip(
            pred, prot, true) if z == 1 and y == 1]
        z0_y1 = [y_hat for y_hat, z, y in zip(
            pred, prot, true) if z == 0 and y == 1]
        return abs(sum(z1_y1)-sum(z0_y1)) + abs(sum(z1_y0)-sum(z0_y0))
    
    def fairness(self, weights, x_transpose, true, prot, metric="spd"):
        x_dot_weights = np.matmul(weights, x_transpose) + self.bias
        pred = self._sigmoid(x_dot_weights)
        if metric == "spd":
            return self.spd(pred, prot)
        else:
            return self.eo_sum(pred,prot,true)
        
    def fair_grad(self, weights, x_transpose,true, prot, metric="spd"):
        g = grad(self.fairness, 0)
        return g(weights, x_transpose, true, prot, metric)
    
        
    def update_model_parameters(self, error_w, error_b):
        self.weights = self.weights - 0.1 * error_w
        self.bias = self.bias - 0.1 * error_b

    def predict(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    def _sigmoid_function(self, x):
        if x >= 0.0:
            z = np.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = np.exp(x)
            return z / (1.0 + z)

    def _transform_x(self, x):
        x = copy.deepcopy(x)
        return x.values

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.values.reshape(y.shape[0], 1)
    