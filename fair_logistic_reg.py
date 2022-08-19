import copy
import autograd.numpy as np
from tqdm import tqdm
from autograd import grad
from sklearn.metrics import accuracy_score

"""
Based on https://github.com/casperbh96/Logistic-Regression-From-Scratch
"""
np.random.seed(1337)
class FairLogisticRegression():
    def __init__(self, fairness_metric="eo_sum", lam = 0.983, model = None):
        self.fairness_metric = fairness_metric
        self.losses = []
        self.train_accuracies = []
        self.model = model
        #Tradeoff between fairness and accuracy
        self.lam = lam

    def pre_fit(self, x, y, epochs = 150):
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for i in range(epochs):
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.update_model_parameters(error_w, error_b)

            
            
    def fit(self, x, y, z, epochs, data = None, missing = None):
        x = self._transform_x(x)
        y = self._transform_y(y)
        z = self._transform_y(z)
        #print(data.values)
        data = self._transform_x(data)
        
        #print("WEIGHTS", self.weights)

        for i in range(epochs):
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            if self.model is not None:
                fair_error_w = self.fair_grad_model(self.weights, self.weights_pred, x.transpose(),y,z,
                                                    model = self.model, data = data, missing = missing,
                                                    metric = self.fairness_metric)
                
            else:
                
                fair_error_w = self.fair_grad(self.weights, x.transpose(),y,z,self.fairness_metric)
            #print("difference_error: ", error_w-fair_error_w)
            #Scaling the fairness error
            if np.linalg.norm(fair_error_w) ==0:
                scaler = 1
            else:
                scaler = (np.linalg.norm(error_w)/np.linalg.norm(fair_error_w))
            #print("SCALER", scaler)
            #scaler=1
            #fair_error_w = fair_error_w*(np.linalg.norm(error_w)/np.linalg.norm(fair_error_w))
            if i > 300:
                print(np.sum(self.predict(x))/len(x))
                print(fair_error_w)
            #print("FAIR_ERROR_W", fair_error_w)
            #print("U", (1-self.lam)*error_w)
            #print("F UNSCALED",self.lam*fair_error_w)
            #print("F SCALED", self.lam*fair_error_w*scaler )
            error_w = (1-self.lam)*error_w + self.lam*fair_error_w*scaler
            #print("ERROR_W",error_w)
            if np.linalg.norm(error_w) < 0.000001:
                print("BREAK AT EPOCH ", i)
                break
            
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
        Assumes prot is 0/1 binary
        
        Can consider the alternative where we take the raw probabilities from the sigmoid output
        """
        z1_y0 = [y_hat for y_hat, z, y in zip(
            pred, prot, true) if z == 1 and y == 0]
        z0_y0 = [y_hat for y_hat, z, y in zip(
            pred, prot, true) if z == 0 and y == 0]
        z1_y1 = [y_hat for y_hat, z, y in zip(
            pred, prot, true) if z == 1 and y == 1]
        z0_y1 = [y_hat for y_hat, z, y in zip(
            pred, prot, true) if z == 0 and y == 1]
        #print("EOSUM", (np.sum(z1_y1) - np.sum(z0_y1)) + (np.sum(z1_y0)-np.sum(z0_y0)))
        return (np.sum(z1_y1) - np.sum(z0_y1)) + (np.sum(z1_y0)-np.sum(z0_y0))
        #-return abs(sum(z1_y1)/len(z1_y1)-sum(z0_y1)/len(z0_y1)) + abs(sum(z1_y0)/len(z1_y0)-sum(z0_y0)/len(z0_y0))
        
    def custom_fair(self, pred, prot):
        z_1 = [y_hat for y_hat, z in zip(pred, prot) if z==1]
        z_0 = np.array([y_hat for y_hat, z in zip(pred, prot) if z==0])
        print("CUSTOM FAIR", np.sum([np.sum((z1 - z_0)**2) for z1 in z_1]))
        return np.sum([np.sum((z1 - z_0)**2) for z1 in z_1])
    
    def fairness(self, weights, x_transpose, true, prot, metric="spd"):
        x_dot_weights = np.matmul(weights, x_transpose) + self.bias
        pred = self._sigmoid(x_dot_weights)
        if metric == "spd":
            return self.spd(pred, prot)
        else:
            return self.eo_sum(pred,prot,true)
        
    def fair_grad(self, weights,  x_transpose,true, prot, metric="spd"):
        g = grad(self.fairness, 0)
        return g(weights, x_transpose, true, prot, metric)
    
    def fairness_model(self, weights,weights_pred, x_transpose, true, prot, model, data, missing, metric="spd"):
    
        x_beta = np.matmul(weights, x_transpose) + self.bias

        pred =  self._sigmoid(x_beta)
        #print(pred[:5])
        new_data = np.hstack((data, pred.reshape(-1,1)))
        #print("new_data")
        x_gamma= np.matmul(weights_pred, new_data.transpose()) + self.bias_pred
        #print("x_gamma")
        pred = self._sigmoid(x_gamma)
        #print("PRED_GAMMA")
        if metric == "spd":
            return self.spd(pred, prot)
        elif metric == "custom":
            #print("custom")
            return self.custom_fair(pred,prot)
        else:
            return self.eo_sum(pred,prot,true)
        
        
    def fair_grad_model(self, weights, weights_pred, x_transpose, true, prot, model, data, missing, metric = "spd"):
        g = grad(self.fairness_model, 0)
        return g(weights, weights_pred, x_transpose, true, prot, model, data, missing, metric)
        
    def update_model_parameters(self, error_w, error_b, main = True):
        if main:
            self.weights = self.weights - 0.1* error_w
            self.bias = self.bias - 0.1 * error_b
        else:
            self.weights_pred = self.weights_pred - 0.1* error_w
            self.bias_pred = self.bias_pred - 0.1*error_b

    def predict(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return np.around(probabilities)
    


    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def _transform_x(self, x):
        x = copy.deepcopy(x)
        return x.values

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.values.reshape(y.shape[0], 1)
    
    def predict_predictive(self, x):
        x_dot_weights = np.matmul(x, self.weights_pred.transpose()) + self.bias_pred
        probabilities = self._sigmoid(x_dot_weights)
        return np.around(probabilities)
    
    def fit_predicitve(self, x, y, epochs = 20):
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.weights_pred = np.zeros(x.shape[1])
        self.bias_pred = 0

        for i in tqdm(range(epochs)):
            x_dot_weights = np.matmul(self.weights_pred, x.transpose()) + self.bias_pred
            pred = self._sigmoid(x_dot_weights)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.update_model_parameters(error_w, error_b, main=False)
    