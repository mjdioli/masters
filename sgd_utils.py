from autograd import grad
import numpy as np

def fairness(policy, model)-> int:
    ny, nz =len(model.dirich_y_x), len(model.dirich_z_yx)
    model_delta = model.get_model_delta()
    fair_val = 0
    for j in range(ny):
        for k in range(nz):
            delta = policy*model_delta[:,j,k]
            fair_val += np.linalg.norm(delta, 1)
    return fair_val

def utility(policy, model) -> int:
    #since this is only used for imputation we have utility being the identification function i.e. accuracy
    nx, ny, nz =len(model.dirich_x), len(model.dirich_y_x)
    utility = 0
    for i in range(nx):
        for j in range(ny):
            if j == 0:
                utility += policy[0, i]*model.pxy(i,j)
            else:
                utility += policy[1, i]*model.pxy(i,j)
    return utility
def cost(policy:np.array, model, lam:int) -> int:
    return (1-lam)*utility(policy, model) - lam*fairness(policy, model)

def autograd_cost(policy, model, lam):
    g = grad(cost, 0)
    return g(policy, model, lam)

def analytical_grad(lam):
    pass

def normalise_policy(policy):
    #Policy is a 2xn matrix
    policy[policy<0]=0
    policy[policy>1]=1
    for x_val in range(len(policy[0])):
        policy[:,x_val] /= np.sum(policy[:,0])

def sgd(model, alpha, lam, n_iter = 10):
    for _ in range(n_iter):
        sample = model.sample()
        gradient = autograd_cost(model.policy, sample, lam)
        #Perhaps do mean
        model.policy -= alpha*gradient
    print("SGD DONE")
    
