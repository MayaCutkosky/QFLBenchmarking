#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 15:12:55 2025

@author: maya
"""

import fedjax
import jax
from sklearn.decomposition import PCA
import numpy as np

def EMNIST_dataset():
    pca = PCA(256)
    
    dset = fedjax.datasets.emnist.load_data()
    client_ids = list(dset[0].client_ids())[:5]
    clients_ids_and_data = list(dset[0].get_clients(client_ids))
    
    rng = jax.random.PRNGKey(0)
    
    mydset = []
    for i, (rng, old_dset) in enumerate(zip(jax.random.split(rng, 5), clients_ids_and_data)):
        old_dset = old_dset[1].all_examples()
        x = old_dset['x'].reshape(len(old_dset['x']),-1)
        try :
            x = pca.fit_transform(x)
        except: #cheating...
            x = pca.transform(x)
        
        y = old_dset['y']
        inds = np.where( np.isin(y, np.arange(10)) )
        y = y[inds]
        x = x[inds]
        new_dset = {
                'x' : x, 
                'y' : y,
                'client_id' : jax.numpy.ones(len(x))*i
            }
        mydset.append([ i, fedjax.ClientDataset(new_dset), rng ])
    return mydset

    
