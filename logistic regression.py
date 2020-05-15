#LOGISTIC REGRESSION

from PIL import Image
import numpy as np
import sys
import os
import csv
import pandas as pd
from sys import argv
from scipy.linalg import eigh

#making of train data
path=sys.argv[1]
fp =open(path,"r")
filelist=[]
labels=[]
for line in fp:
    a=line.split(" ")
    filelist.append(a[0]) 
    labels.append(a[1].rstrip("\n"))

imageset=[]
labels_name=dict()
it=0
y_train=[]
for filepath in filelist:
    im1 = Image.open(filepath)
    newsize = (64,64) 
    im1 = im1.resize(newsize) 
 # Make image Greyscale
    im1 = im1.convert('L')
 # Save Greyscale values
    value = np.asarray(im1.getdata(), dtype=np.int)
    value = value.flatten()
    imageset.append(value)
    t=filepath.split('/')
    p=t[-1].split('_')
    y_train.append(int(p[0]))
    labels_name[int(p[0])]=labels[it]
    it+=1

#making of test data
path=sys.argv[2]
fp =open(path,"r")
filelist1=[]
for line in fp:
    filelist1.append(line.rstrip("\n")) 

imageset1=[]
y_test=[]
for filepath in filelist1:
    im1 = Image.open(filepath)
    newsize = (64,64) 
    im1 = im1.resize(newsize) 
# Make image Greyscale
    im1 = im1.convert('L')
# Save Greyscale values
    value = np.asarray(im1.getdata(), dtype=np.int)
    value = value.flatten()
    imageset1.append(value)
    t=filepath.split('/')
    p=t[-1].split('_')
    y_test.append(int(p[0]))

#converting into numpy array
train=np.asarray(imageset)

# print(len(imageset1))
test=np.asarray(imageset1)
# print(test.shape)

#Normalising Data using Standard Scaler
from sklearn.preprocessing import StandardScaler
X_train=StandardScaler().fit_transform(train)
X_test=StandardScaler().fit_transform(test)

#PCA Function

def PCA(A,K):
    covarMat=np.cov(A.T)
    values, vectors=eigh(covarMat)
    idx= np.argsort(values)
    idx = idx[::-1]
    eigvec = vectors[:,idx]
    eigval = values[idx]
    subeigvec=eigvec[:,:K]
    newdata=np.dot(A,subeigvec)
    return newdata

#Applying PCA
K=400
X_train=PCA(X_train,K)
X_test=PCA(X_test,K)
#converting into numpy array
X_train=np.array(X_train)
X_test=np.array(X_test)
#appending column of 1's in train data
li_Train=[1.0]*(X_train.shape[0])
li_Train=np.array(li_Train)
li_Train.shape=(X_train.shape[0],1)
# print(li_Train.shape,X_train.shape)
X_train= np.hstack((li_Train,X_train))
#appending column of 1's in test data
li_Test=[1.0]*(X_test.shape[0])
li_Test=np.array(li_Test)
li_Test.shape=(X_test.shape[0],1)
X_test= np.hstack((li_Test,X_test))

def logisticfun(z):
  return 1.0 / (1 + np.exp(-z))

def fit(parameters,X_train,y_train):
  alpha=0.02
  for j in range(0,1000):
    z=np.dot(X_train,parameters)
    pred_list=logisticfun(z)
    predlist=np.asarray(pred_list)
    predlist=np.subtract(predlist,y_train)
    df1=X_train.transpose()
    intermediate= np.dot(df1,predlist)
    for i in range(0,401):
      intermediate[i]=intermediate[i]*(1/y_train.shape[0])
    intermediate*=alpha
    parameters=np.subtract(parameters,intermediate)
  return parameters

#creating parameters list
parameters = np.zeros(X_train.shape[1])
parameters=np.expand_dims(parameters, axis=1)

#Training model and finding parameters for multiclass classification
Rparameters=[]
y_train = np.asarray(y_train)
# print(y_train.shape)
for i in range(0,8):
  temp=y_train
  for j in range(0,X_train.shape[0]):
    if(y_train[j]==i):
      temp[j]=1
    else:
      temp[j]=0
  temp = np.asarray(temp)    
  temp.shape=(y_train.shape[0],1)
  Rparameters.append(fit(parameters,X_train,temp))

#Returning predicted list for Xtest
def predict_prob(X_test,Rparameters):
  problist=[]
  for k in range(0,X_test.shape[0]):
    temp=[]
    for i in range(0,8):
      temp.append(logisticfun(np.dot(X_test[k,:],Rparameters[i])))
    # print(temp)
    maxpos = temp.index(max(temp))
    # print(maxpos)
    problist.append(maxpos)
  return problist

t_predlist=predict_prob(X_test,Rparameters)
output=[]

for i in range(len(t_predlist)):
    output.append(labels_name[t_predlist[i]])

for b in output:
  print(b)






