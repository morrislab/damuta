#!/usr/bin/env python
# coding: utf-8

# In[9]:


import damuta as da
import os
os.getcwd()


# # Load package provided config

# In[10]:


dataset_args, model_args, pymc3_args = da.load_default_config()
print(dataset_args)


# Load a custom config

# In[12]:


dataset_args, model_args, pymc3_args = da.load_config("/lila/home/harrigan/damuta/config/tandtiss-defaults.yaml")
print(dataset_args)

