#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 18:36:05 2025

@author: maya
"""
from flower.server_app import app as serverapp
from flower.client_app import app as clientapp
from flwr.simulation import run_simulation
run_simulation(serverapp, clientapp, num_supernodes = 4)