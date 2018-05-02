
# coding: utf-8

# In[4]:


import numpy as np
import h5py
import matplotlib.pyplot as plt
from activations_utils import sigmoid, sigmoid_backward, relu, relu_backward

get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().magic(u'reload_ext autoreload')
get_ipython().magic(u'autoreload 2')

np.random.seed(1)


# In[5]:


def load_data(hdf5_filepath):
    dataset = h5py.File(hdf5_filepath, 'r')
    
    print('Keys : %s' % dataset.keys())
    
    train_X = np.array(dataset["train_X"][:])
    train_Y = np.array(dataset["train_Y"][:])
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
#     train_Y = np.reshape(train_Y, (len(train_Y),1))
    
    test_X = np.array(dataset["test_X"][:])
    test_Y = np.array(dataset["test_Y"][:])
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
#     test_Y = np.reshape(test_Y, (len(test_Y),1))
    
    print('train_X is a {} array and has {} examples' .format(train_X.shape, len(train_X)))
    print('train_Y is a {} array and has {} examples' .format(train_Y.shape, len(train_Y)))
    
    print('test_X is a {} array and has {} examples' .format(test_X.shape, len(test_X)))
    print('test_Y is a {} array and has {} examples' .format(test_Y.shape, len(test_Y)))
    
    return train_X, train_Y, test_X, test_Y


# In[6]:


def initialize_parameters(layer_dims, init_type):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1,L):
        if init_type == "random":
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        elif init_type == "he":
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters


# In[7]:


def linear_forward(Aprev, W, b):
    Z = np.dot(W, Aprev) + b
    
    assert(Z.shape == (W.shape[0], Aprev.shape[1]))
    linear_cache = (Aprev, W, b)

    return Z, linear_cache


# In[8]:


def linear_activation_forward(Aprev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(Aprev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(Aprev, W, b)
        A, activation_cache = relu(Z)
        
    assert (A.shape == (W.shape[0], Aprev.shape[1]))
    cache = (linear_cache, activation_cache)
    
    return A, cache


# In[9]:


def L_model_forward(X, parameters, keep_prob=1):
    caches = []
    A = X
    L = len(parameters) // 2     # number of layers in net
    
    for l in range(1, L):
        Aprev = A
        A, cache = linear_activation_forward(Aprev, parameters["W" + str(l)], parameters["b" + str(l)], activation = "relu")
        
        if keep_prob < 1:
            D = np.random.randn(*A.shape)
            D = (D < keep_prob).astype(int)
            A = np.multiply(A,D)
            A = A / keep_prob
            linear_cache, activation_cache = cache
            cache = (linear_cache, activation_cache, D)
            
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
    
    return AL, caches


# In[15]:


def compute_cost(AL, Y, parameters, lambd=0):
    m = Y.shape[1]
    L = len(parameters) // 2
    
    cross_entropy_cost = -(1.0/m) * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
    
    L2_weight_penalty = 0
    for l in range(1,L):
        L2_weight_penalty += np.sum(np.square(parameters["W" + str(l)]))
    
    L2_regularization_cost = (lambd/(2*m)) * L2_weight_penalty
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    cost = np.squeeze(cost)     # turns [[x]] to x
    
    assert(cost.shape == ())
    
    return cost


# In[16]:


def linear_backward(dZ, linear_cache, lambd):
    Aprev, W, b = linear_cache
    m = Aprev.shape[1]
    
    dW = (1.0/m) * np.dot(dZ, Aprev.T) + (lambd/m) * W
    db = (1.0/m) * np.sum(dZ, axis = 1, keepdims = True)
    dAprev = np.dot(W.T, dZ)
    
    assert (dAprev.shape == Aprev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dAprev, dW, db


# In[12]:


def linear_activation_backward(dA, cache, activation, lambd):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dAprev, dW, db = linear_backward(dZ, linear_cache, lambd)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dAprev, dW, db = linear_backward(dZ, linear_cache, lambd)
        
    return dAprev, dW, db        


# In[20]:


def L_model_backward(AL, Y, caches, lambd=0, keep_prob=1):
    grads = {}
    L = len(caches)     # number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -np.divide(Y,AL) + np.divide((1-Y),(1-AL))
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid", lambd)
    
    
    # loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        
        dA = grads["dA" + str(l+1)]
        _,_,D = current_cache
        dA = np.multiply(dA, D)
        grads["dA" + str(l+1)] = dA / keep_prob
        
        dAprev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache[:2], "relu", lambd)     # as reversed(range(L-1)) starts from L-2, L-3,..., 3, 2, 1, 0
                        
        grads["dA" + str(l)] = dAprev_temp  
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp     # l+1 because labels are not 0 (no 0th layer)
        
    return grads


# In[14]:


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(1,L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
        
    return parameters

