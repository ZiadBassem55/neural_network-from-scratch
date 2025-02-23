#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #في الtorch الداتا اللي بتدخل لازم تكون tensor ولكن في الnumpy الداتا اللي بتدخل لازم تكون array
import random as r #لاني هستخدم الrandom في الweights


# In[11]:


def init_parameters(): #لاني هستخدم الrandom في الweights فهنا هستخدم الseed علشان اقدر اعمل تكرار للنتائج
    r.seed(42)
    w1= np.random.uniform(-0.5,0.5)
    b1= 0.5
    w2= np.random.uniform(-0.5,0.5)
    b2=0.7
    return w1,w2,b1,b2


# In[5]:


def tanh(z): #الtanh هو activation function
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


# In[20]:


def forwardstep(w1,w2,b1,b2,x): #هنا هعمل الforward step وبيحصل فيها ان الinput بيتم تحويلها للhidden layer ومن الhidden layer بتتحول للoutput layer
    z1= np.dot(w1,x)+b1
    F1= tanh(z1)
    z2= np.dot(w2,x)+b2
    F2= tanh(z2)
    return z1,F1,z2,F2
def mean_squared_error(y_true, y_pred):#هنا هعمل الmean squared error
    return np.mean((y_true - y_pred) ** 2)

# In[7]:


x = np.array([[0.5, 0.3], [0.2, 0.8]]) #هنا هعمل الinput
y_true = np.array([[0.1, 0.9], [0.8, 0.2]])

# In[12]:


w1,w2,b1,b2=init_parameters() #هنا هعمل الweights والbias



# In[21]:


 #هنا هعمل الforward step

y_pred, hidden_output, Z1, Z2 = forwardstep(w1,w2,b1,b2,x)
#هنا هعمل الmean squared error
error = mean_squared_error(y_true, y_pred)

# In[26]:

#هنا هطبع الoutput والhidden layer output والweights والbias
print("Input (X):\n", x)
print("Weights (W1):\n", w1)
print("Weights (W2):\n", w2)
print("Bias (b1):\n", b1)
print("Bias (b2):\n", b2)
print("Hidden Layer Output (A1):\n", hidden_output)
print("Output Layer Output (A2):\n", y_pred)
print("Mean Squared Error (MSE):\n", error)
print("Target (y_true):\n", y_true)

# In[ ]:




