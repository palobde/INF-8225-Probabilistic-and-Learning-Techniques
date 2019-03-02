# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 11:09:52 2018

@author: palobde
"""

import numpy as np
import matplotlib.pyplot as plt

# Arrays = pluie, arroseur, waston, holmes (faux, vrai)

# Entree des donnees du probleme
prob_pluie = np.array([0.8, 0.2]).reshape(2, 1, 1, 1)
print("Prob(pluie)={}\n".format(np.squeeze(prob_pluie)))

prob_arroseur = np.array([0.9, 0.1]).reshape(1, 2, 1, 1)
print("Prob(arrosseur)={}\n".format(np.squeeze(prob_arroseur)))


watson = np.array([[0.8, 0.2], [0, 1]]).reshape(2, 1, 2, 1)
print( "Prob(watson|Pluie)={}\n".format(np.squeeze(watson)))


holmes = np.array([[1, 0], [0.1, 0.9], [0, 1], [0, 1]]).reshape(2, 2, 1 ,2)
print( "Prob(holmes|Pluie, Arroseur)={}\n".format(np.squeeze(holmes)))


# Reponse a la question 1
pr_h_1 = (prob_arroseur * holmes * prob_pluie).sum(1).sum(0).squeeze()[1]
print("a)  Prob(H=1) = {}".format(pr_h_1))

# Reponse a la question 2
pr_A_1_H_1 = (holmes[:,1,:,1].squeeze() * prob_arroseur[:,1,:,:].squeeze() * prob_pluie.squeeze()).sum(0)/pr_h_1
print("b)  Pr(A=1|H=1) = {}".format(pr_A_1_H_1))

# Reponse a la question 3
pr_A_1_H_1_W_1_numerateur = (prob_arroseur[:,1,:,:].squeeze() * holmes[:,1,:,1].squeeze() * watson[:,:,1,:].squeeze() * prob_pluie.squeeze()).sum()
pr_A_1_H_1_W_1_denominateur_1 = (holmes[:,0,:,1].squeeze() * watson[:,:,1,:].squeeze() * prob_pluie.squeeze() * prob_arroseur[:,0,:,:].squeeze())
pr_A_1_H_1_W_1_denominateur_2 = (holmes[:,1,:,1].squeeze() * watson[:,:,1,:].squeeze() * prob_pluie.squeeze() * prob_arroseur[:,1,:,:].squeeze())
denominateur = sum(pr_A_1_H_1_W_1_denominateur_1 + pr_A_1_H_1_W_1_denominateur_2) 
print("c)  Pr(A=1|H=1,W=1) = {}".format(pr_A_1_H_1_W_1_numerateur / denominateur))

