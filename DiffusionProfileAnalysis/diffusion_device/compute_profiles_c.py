#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 17:56:05 2018

@author: quentinpeter
"""
import numpy as np
import C_diffusion

# @profile


def compute_profiles(NSteps_binary, idx_sorted, profilespos, Fdic):
    """
    Do the calculations slightly slower with C!
    """
    Nbinary = np.shape(NSteps_binary)[1]

    if Nbinary > len(Fdic["Flist"]):
        Flist = np.zeros((Nbinary, *np.shape(Fdic["Flist"])[1:]))
        Flist[:len(Fdic["Flist"])] = Fdic["Flist"]
        for i in range(len(Fdic["Flist"]), Nbinary):
            Flist[i] = np.dot(Flist[i-1], Flist[i-1])
        Fdic["Flist"] = Flist

    Flist = Fdic["Flist"]

    C_diffusion.compute_profiles(NSteps_binary, idx_sorted, profilespos, Flist)
    return profilespos
