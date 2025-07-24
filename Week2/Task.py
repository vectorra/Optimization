
import optimizers
import visualizer
from autograd import grad, hessian
import autograd.numpy as anp
from mpl_toolkits.mplot3d import Axes3D
from autograd import grad, hessian
import autograd.numpy as anp
from visualizer import visualize
from optimizers import Levenberg_Marquardt, L_BFGS, AMSGrad_noisy, PolakRibiereCG_noisy, MomentumGD_noisy, MomentumGD, PolakRibiereCG, AMSGrad

def himmelblau(x):
    return (x[0]**2 + x[1] - 11.0)**2 + (x[0] + x[1]**2 - 7.0)**2

objective = himmelblau
objective_grad = grad(objective)

optimizers_dict = {
    "Levenberg-Marquardt": Levenberg_Marquardt(f=objective),
    "L-BFGS": L_BFGS(objective, objective_grad),  
    "Momentum": MomentumGD(),
    "Momentum Noisy": MomentumGD_noisy(),
    "Polak-Ribiere CG": PolakRibiereCG(),
    "Polak-Ribiere CG Noisy": PolakRibiereCG_noisy(),
    "AMSGrad": AMSGrad(),
    "AMSGrad Noisy": AMSGrad_noisy()
}
visualize(optimizers_dict)