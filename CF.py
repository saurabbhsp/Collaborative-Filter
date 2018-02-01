
# coding: utf-8

# In[1]:


import pandas as pd
import sys
import numpy as np
from random import shuffle
import time
import pickle

# # Reading data

# In[2]:


def readValidationSet(fileName, validationDict):
    currentItem = None
    currentUserDictionary = None
    with open(fileName) as f:
        for line in f:
            if str(line).endswith(":\n"):
                """Reading new item"""
                if currentItem != None:
                    validationDict[currentItem] = currentUserDictionary
                currentItem = int(str(line).replace(":\n",""))
                currentUserDictionary = {}
            else:
                """Reading user rating. Currently rating is not known so placing 0 as temp value.
                Will be replaced while reading data from combined set"""
                currentUserDictionary[int(str(line).replace("\n",""))] = 0


# In[3]:


validationSet = {}
readValidationSet("data/probe.txt", validationSet)


# In[4]:


def readTrainSet(fileName, trainSet, validationSet):
    currentItem = None
    currentUserDictionary = None

    with open(fileName) as f:
        for line in f:
            if str(line).endswith(":\n"):
                """Reading new item"""
                if currentItem != None:
                    trainSet[currentItem] = currentUserDictionary
                currentItem = int(str(line).replace(":\n",""))
                currentUserDictionary = {}
 
            else:
                """Reading user rating. """
                data = str(line).split(",")
                if currentItem in validationSet and int(data[0]) in validationSet[currentItem]:
                    """Item-User found in validation set"""
                    validationSet[currentItem][int(data[0])] = int(data[1])
                else:
                    """Item-User belongs to test set"""
                    currentUserDictionary[int(data[0])] = int(data[1])


# In[5]:


trainSet = {}
readTrainSet("data/combined_data_1.txt", trainSet, validationSet)
time.sleep(5)

# In[6]:


readTrainSet("data/combined_data_2.txt", trainSet, validationSet)
time.sleep(5)

# In[7]:

readTrainSet("data/combined_data_3.txt", trainSet, validationSet)
time.sleep(5)

# In[8]:


readTrainSet("data/combined_data_4.txt", trainSet, validationSet)
time.sleep(5)

# In[9]:


sys.getsizeof(trainSet)


# In[49]:


A = np.array(range(1,26)).reshape(5,5)


# In[52]:


A


# In[51]:


A[2,:]*5


# # Train model
# ## Input format
# {
#     item1:{
#                 user1:rating1,
#                 user2:rating2
#                 },
#     item2:{
#                 user1:rating1,
#                 user2:rating2
#                 }
# }

# In[10]:


"""
Input for Collaborative filter

"""
class CollaborativeFilter:
    """P matrix"""
    p = None
    """Q matrix"""
    q = None
    """global bias"""
    g = None
    """user bias"""
    ub = None
    """item bias"""
    ib = None
    
    
    def learnMatrices(self, data, latentFactors, epochs, learningRate, regFactor):
        
        self.p = np.random.normal(scale = 1.0/latentFactors, size = (max(self.ub.keys()) + 1, latentFactors))
        self.q = np.random.normal(scale = 1.0/latentFactors, size = (max(self.ib.keys()) + 1, latentFactors))
        for i in range(0, epochs):
            print("Epoch - "+str(i))
            
            actual = []
            prediction = []
            
            items = list(data.keys())
            shuffle(items)
            for item in items:
                users = list(data[item].keys())
                shuffle(users)
                time.sleep(0.05)
                for user in users:
                    actualRating = data[item][user]


                    predictedRating = self.g + self.ub[user] + self.ib[item] + np.dot(self.p[user,:].T, self.q[item,:])
                                       
                    actual.append(actualRating)
                    prediction.append(predictedRating)
                    
                    residual = actualRating - predictedRating
                    """Making update for all latent variables"""
                    self.p[user, :] = self.p[user, :] + (learningRate * ((self.q[item, :] * residual) - (regFactor * self.p[user, :])))
                    self.q[item, :] = self.q[item, :] + (learningRate * ((self.p[user, :] * residual) - (regFactor * self.q[item, :])))
            print(np.sqrt(np.sum((np.array(actual) - np.array(prediction)) ** 2)/len(actual)))

            with open('PQ-'+str(i)+'.pkl', 'wb') as f:
                pickle.dump([self.p, self.q], f) 

    
    def __init__(self, data, useBias = True):
        self.ub = {}
        self.ib = {}
        self.g = 0
        userRatingCount = {}
        itemRatingCount = {}
        globalRatingCount = 0
        
        if useBias:
            """Learn userBias, globalBias, itemBias"""
            for item, userRating in data.items():
                for user, rating in userRating.items():
                    """For global bias"""
                    globalRatingCount = globalRatingCount + 1
                    self.g = self.g + rating
                
                    """For user bias"""
                    if user in self.ub:
                        self.ub[user] = self.ub[user] + rating
                        userRatingCount[user] = userRatingCount[user] + 1
                    else:
                        self.ub[user] = rating
                        userRatingCount[user] = 1
                
                    """For item bias"""
                    if item in self.ib:
                        self.ib[item] = self.ib[item] + rating
                        itemRatingCount[item] = itemRatingCount[item] + 1
                    else:
                        self.ib[item] = rating
                        itemRatingCount[item] = 1
            
            self.g = (self.g * 1.0)/globalRatingCount
            print(globalRatingCount)
            print(globalRatingCount%1000)
            for user in self.ub.keys():
                self.ub[user] = ((self.ub[user] * 1.0)/userRatingCount[user]) - self.g
                
            for item in self.ib.keys():
                self.ib[item] = ((self.ib[item] * 1.0)/itemRatingCount[item]) - self.g               

            with open('bias.pkl', 'wb') as f:
                pickle.dump([self.g, self.ub, self.ib], f) 
# In[11]:


with open('trainSet.pkl', 'wb') as f:
     pickle.dump(trainSet, f)

with open('validationSet.pkl', 'wb') as f:
     pickle.dump(validationSet, f)

cf = CollaborativeFilter(trainSet)


# In[12]:


import gc
gc.collect()


# In[13]:


cf.learnMatrices(trainSet, 100, 5, 2e-4, 0.02)

