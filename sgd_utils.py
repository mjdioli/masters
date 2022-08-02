from autograd import grad
import numpy as np
def fairness(model)-> int:
    ny, nz =len(model.dirich_y_x), len(model.dirich_z_yx)
    policy = model.policy
    model_delta = model.get_model_delta()
    fair_val = 0
    for j in range(ny):
        for k in range(nz):
            delta = policy*model_delta[:,j,k]
def utility(model) -> int:
    #using only the identifier function as opposed to a more complex 
    # utility function from Christos' paper
    #Column indeces of probs indicates the outcome being predicted
    #Does not contain negative utility if false
    #Can maybe resample from the training data to get a probability estimate of the 
    sum([1* for pr, tr in zip(probs,true)])

def cost(policy:np.array, pred:list, true:list, lam:int,) -> int:
    return (1-lam)*utility(policy, pred,true) - lam*fairness()

def autograd_cost(lam):
    g = grad(cost, 0)
    return g(lam)

def analytical_grad(lam):
    pass

def normalise_policy(policy):
    #Policy is a 2xn matrix
    policy[policy<0]=0
    policy[policy>1]=1
    for x_val in range(len(policy[0])):
        policy[:,x_val] /= np.sum(policy[:,0])

def sgd(model, n_iter = 10000):
    pass 
    
