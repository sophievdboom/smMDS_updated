#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 17:56:10 2018

@author: quentinpeter
"""
import numpy as np
# @profile


def compute_profiles(NSteps_binary, idx_sorted, profilespos, Fdic):
    """
    Do The calculations
    """
    Nbinary = np.shape(NSteps_binary)[1]

    if Nbinary > len(Fdic["Flist"]):
        Flist = np.zeros((Nbinary, *np.shape(Fdic["Flist"])[1:]))
        Flist[:len(Fdic["Flist"])] = Fdic["Flist"]
        for i in range(len(Fdic["Flist"]), Nbinary):
            Flist[i] = np.dot(Flist[i-1], Flist[i-1])
        Fdic["Flist"] = Flist

    Flist = Fdic["Flist"]

    # for each unit
    for i in range(Nbinary):
        F = Flist[i]
        # save previous number
        prev = np.zeros(i + 1, dtype=bool)
        for j in range(np.shape(NSteps_binary)[0]):
            bs = NSteps_binary[idx_sorted[j], i]
            act = NSteps_binary[idx_sorted[j], :i + 1]
            # If we have a one, multiply by the current step function
            if bs:
                prof = profilespos[idx_sorted[j], :]
                # If this is the same as before, no need to recompute
                if (act == prev).all():
                    prof[:] = profilespos[idx_sorted[j - 1]]
                else:
                    prof[:] = np.dot(F, prof)
            prev = act
    return profilespos
