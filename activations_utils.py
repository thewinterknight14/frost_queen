
# coding: utf-8

# In[41]:


import numpy as np


# In[42]:


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    activation_cache = Z
    
    assert (A.shape == Z.shape)
    
    return A, activation_cache


# In[43]:


def relu(Z):
    A = np.maximum(0,Z)
    activation_cache = Z
    
    assert (A.shape == Z.shape)
    
    return A, activation_cache


# In[44]:


def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


# In[45]:


def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

