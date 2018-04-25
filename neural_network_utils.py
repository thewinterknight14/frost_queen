
# coding: utf-8

# In[39]:


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


# In[50]:


def load_data(hdf5_filepath):
    dataset = h5py.File(hdf5_filepath, 'r')
    
    print('Keys : %s' % dataset.keys())
    
    train_X = np.array(dataset["train_X"][:])
    train_Y = np.array(dataset["train_Y"][:])
    train_Y = np.reshape(train_Y, (len(train_Y),1))
    
    test_X = np.array(dataset["test_X"][:])
    test_Y = np.array(dataset["test_Y"][:])
    test_Y = np.reshape(test_Y, (len(test_Y),1))
    
    print('train_X is a {} array and has {} examples' .format(train_X.shape, len(train_X)))
    print('train_Y is a {} array and has {} examples' .format(train_Y.shape, len(train_Y)))
    
    print('test_X is a {} array and has {} examples' .format(test_X.shape, len(test_X)))
    print('test_Y is a {} array and has {} examples' .format(test_Y.shape, len(test_Y)))
    
    return train_X, train_Y, test_X, test_Y


# In[41]:


def initialize_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
    
    return parameters


# In[42]:


def linear_forward(Aprev, W, b):
    Z = np.dot(W, Aprev) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    linear_cache = (Aprev, W, b)

    return Z, linear_cache


# In[43]:


def linear_activation_forward(Aprev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(Aprev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(Aprev, W, b)
        A, activation_cache = relu(Z)
        
    cache = (linear_cache, activation_cache)
    
    return A, cache


# In[44]:


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2     # number of layers in net
    
    for l in range(1, L):
        Aprev = A
        A, cache = linear_activation_forward(Aprev, parameters["W" + str(l)], parameters["b" + str(l)], activation = "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
    
    return AL, caches


# In[45]:


def compute_cost(AL, Y):
    m = Y.shape[1]
    
    cost = -(1/m) * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
    cost = np.squeeze(cost)     # turns [[x]] to x
    
    assert(cost.shape == ())
    
    return cost


# In[46]:


def linear_backward(dZ, linear_cache):
    Aprev, W, b = linear_cache
    m = Aprev.shape[1]
    
    dW = (1/m) * np.dot(dZ, Aprev.T)
    db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    dAprev = np.dot(W.T, dZ)
    
    return dAprev, dW, db


# In[47]:


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dAprev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dAprev, dW, db = linear_backward(dZ, linear_cache)
        
    return dAprev, dW, db        


# In[48]:


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)     # number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -np.divide(Y,AL) + np.divide((1-Y),(1-AL))
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
        
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache, "relu")     # as reversed(range(L-1)) starts from L-2, L-3,..., 3, 2, 1, 0
        
        grads["dA" + str(l+1)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp     # l+1 because labels are not 0 (no 0th layer)
        
    return grads


# In[49]:


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(L):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["db" + str(l)] = parameters["db" + str(l)] - learning_rate * grads["dW" + str(l)]

