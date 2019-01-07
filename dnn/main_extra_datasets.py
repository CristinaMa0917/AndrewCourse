# Package imports
from model_package import *
from lr import logistic_regression
from nn_model import nn_model
from extra_datasets import *
np.random.seed(1) # set a seed so that the results are consistent

## datasets = {"noisy_circles","noisy_moons","blobs","gaussian_quantiles"}
X,Y=load_datasets("noisy_moons",isshow=False)

hidden_layer_sizes = [50]
# hidden_layer_sizes = [1, 2, 3, 4, 5, 20]
nn_model(X,Y,hidden_layer_sizes,num_iterations= 10000,print_cost= False)