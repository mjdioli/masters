
"""Christos' paper: https://arxiv.org/abs/1706.00119
Uses multinomial with the
factorization Pθ (x, y, z) = Pθ (y | x, z)Pθ(x | z)Pθ (z)
The conjugate prior distribution to this model
is a simple Dirichlet-product
"""
import numpy as np

import sgd_utils

UTILS = np.array([[0, -0.5],[-0.5, -1]])
class SGD_bayes():
    """
    lam = tradeoff between utility and fairness
    """
    def __init__(self, num_x_features,num_y_features,num_z_features, lam = 0.5):
        self.dirich_x = np.array([0.5]*num_x_features)
        self.dirich_y_x = np.zeros(num_y_features, num_x_features) +0.5
        self.dirich_z_yx =  np.zeros(num_z_features, num_y_features, num_x_features) +0.5
        self.policy = np.random.uniform(0,1,size=(2,num_x_features))
        self.normalise_policy()
        
        
    def normalise_policy(self):
        #Policy is a 2xn matrix
        self.policy[self.policy<0]=0
        self.policy[self.policy>1]=1
        for x_val in range(len(self.policy[0])):
            self.policy[:,x_val] /= np.sum(self.policy[:,0])
    
    def get_model_delta(self):
        return self.px_y - self.px_yz
    
    
       
    def calculate_marginals(self):
        nx, ny, nz =len(self.dirich_x), len(self.dirich_y_x), len(self.dirich_z_yx)
        self.px_yz = np.zeros(nx, ny, nz)
        self.pz_xy = np.zeros(nz, nx, ny)
        self.pz_y = np.zeros(nz, ny)
        self.pyz = np.zeros(ny, nz)
        self.px_y = np.zeros(nx, ny)
        self.py_x = np.zeros(ny, nx)
        self.pxy = np.zeros(nx, ny)
        self.py = np.zeros(ny)
        self.px = np.zeros(nx)
        
        #Could probably improve this section with list comprehensions if necessary
        
        #Cleaning up use of ijk indexing and binding them to a variable/use a variable instead might be useful.
        for j in range(ny):
            for k in range(nz):
                self.pyz[i,j] = np.sum(self.pxyz[:,j,k])
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    self.px_yz[i,j,k] = self.pxyz[i,j,k]/self.pyz[j,k]
        
        for i in range(nx):
            self.px[i] = np.sum(np.sum(self.pxyz[i,:,:]))
        
        for j in range(ny):
            self.py[j] = np.sum(np.sum(self.pxyz[:,j,:]))
            for i in range(nx):
                self.pxy[i,j] = np.sum(self.pxyz[i,j,:])
                self.px_y[i,j] = self.pxy[i,j]/self.py[j]
                self.py_x[i,j] = self.pxy[i,j]/self.px[i]
                for k in range(nz):
                    self.pz_xy[k,i,j] = self.pxyz[i,j,k]*self.pxy[i,j]
                    
        
    
    def calculate_marginal_model(self):
        nx, ny, nz =len(self.dirich_x), len(self.dirich_y_x), len(self.dirich_z_yx)
        px = self.dirich_x/np.sum(self.dirich_x)
        self.pxyz = np.zeros(nx,ny,nz)
        
        #Consider simplifying and speeding up with an  np.sum along an axis
        # and dispensing with for loops
        for i in range(nx):
            py_x = self.dirich_y_x[:,i]/np.sum(self.dirich_y_x[:,i])
            for j in range(ny):
                pz_yx = self.dirich_z_yx[:,j,i]/np.sum(self.dirich_z_yx[:,j,i])
                for k in range(nz): 
                    self.pxyz[i,j,k] = pz_yx[k]*py_x[j]*px[i]
        
    
    def sample(self):
        nx, ny, nz =len(self.dirich_x), len(self.dirich_y_x), len(self.dirich_z_yx)
        px = np.random.dirichlet(self.dirich_x)
        model = SGD_bayes(self.num_x_features, self.num_y_features, self.num_z_features)
        model.pxyz = np.zeros(nx,ny,nz)
        for i in range(nx):
            py_x = np.random.dirichlet(self.dirich_y_x[:,i])
            for j in range(ny):
                pz_yx = np.random.dirichlet(self.dirich_z_yx[:,j,i])
                for k in range(nz):
                    model.pxyz[i,j,k] = pz_yx[k]*py_x[j]*px[i]
        model.calculate_marginals()
        return model
        
        
    #TODO
    def impute(self, length, impute_index):
        return [0 if np.random.uniform(0,1)<self.policy[0][impute_index] else 1 for i in range(length)]
    
    def fit(self, data):
        #Assumes dataframe
        self.dirichlet_params = np.array(self.dirichlet_params+data.sum())
        print("Data fitted")
    

    
    def load(savepath):
        pass
    
    def save(savepath):
        #Probs have to convert the dtype of above to save to json
        #Could alternatively consider saving to pickle
        pass
    
    