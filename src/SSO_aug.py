
import numpy as np
import time
import copy
import random

p=[10,10,13,4,9,4,8,15,7,1,9,3,15,9,11,6,5,14,18,3]
d=[50,38,49,12,20,105,73,45,6,64,15,6,92,43,78,21,15,50,150,99]
w=[10,5,1,5,10,1,5,10,5,1,5,10,10,5,1,10,5,5,1,5]

Ngen=1200 # default value is 2000
Njob=20 # number of jobs
Nsol=50 # default value is 30

X, F, pX, pF, gBest, genBest = [], [], [], [], 0, 0
for sol in range(Nsol):
    rX=list(np.random.permutation(Njob)) # generate a random permutation of 0 to Njob-1
    X.append(rX) # add to the X2.
    pX.append(rX) # add to the X2.
    ptime, tardiness=0, 0
    for job in range(Njob):
        ptime=ptime+p[X[sol][job]]
        tardiness=tardiness+w[X[sol][job]]*max(ptime-d[X[sol][job]],0)
        #print(ptime,X[sol][job],p[X[sol][job]])
    F.append(tardiness)
    pF.append(tardiness)    
    if F[sol]<F[gBest]: gBest=sol 
    #print(F[sol], X[sol])  
#print(gBest,pF[gBest]) 