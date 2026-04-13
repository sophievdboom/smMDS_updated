# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 07:55:13 2018

@author: quentinpeter
"""
import numpy as np
from scipy.optimize import minimize
import diffusion_device.basis_generate as ddbg


def gaussian(X0, sigma_l, sigma_r, width, A, X):
    width = np.abs(width)
    ret = np.zeros(len(X))
    Xl = X0 - width / 2
    Xr = X0 + width / 2
    if sigma_l > 0:
        ret[X < Xl] = A * np.exp(-(X[X < Xl] - Xl)**2 / (2 * sigma_l**2))

    ret[np.logical_and(X >= Xl, X <= Xr)] = A

    if sigma_r > 0:
        ret[X > Xr] = A * np.exp(-(X[X > Xr] - (Xr))**2 / (2 * sigma_r**2))
    return ret


def get_test(args, X, profs, Q, radius, reading_pos, beta, DRh
             ):
    X0, sigma_left, sigma_right, width = args

    prof_0 = gaussian(X0, sigma_left, sigma_right, width, 1, X)

    phi_prof = reading_pos / Q / radius * beta * DRh * 3600e9
    profs_test, phi_alt, dphi = ddbg.get_unitless_profiles(
        prof_0, phi_prof, beta, Zgrid=11)

    return profs_test


def fun(var, X, prof, Q, radius, reading_pos, beta, DRh, prof_slice):
    prof_test = get_test(var, X, prof, Q, radius, reading_pos, beta, DRh)

    prof_test /= np.mean(prof_test[..., prof_slice], -1)[..., np.newaxis]
    prof = prof / np.mean(prof[..., prof_slice], -1)[:, np.newaxis]
    return np.sqrt(np.mean(np.square(prof_test - prof)[..., prof_slice]))


class Minimizer():
    def __init__(self, profs, flow_rate, radius,
                 reading_pos, beta, DRh, prof_slice):
        self.flow_rate = flow_rate
        self.radius = radius
        self.Y_pos = np.arange(np.shape(profs)[-1])
        self.fit_param = np.array([np.argmax(np.mean(profs, 0)) + 5, 5, 5, 5])
        self.profs = profs
        self.pars = []
        self.lse = []
        self.var0 = 1
        self.reading_pos = reading_pos
        self.beta = beta
        self.DRh = DRh
        self.prof_slice = prof_slice

    def new_var(self, var):
        fit = minimize(
            fun, x0=self.fit_param,
            args=(self.Y_pos, self.profs, self.flow_rate, var * self.radius,
                  self.reading_pos, self.beta, self.DRh, self.prof_slice))
        self.pars.append(var)
        self.lse.append(fit.fun)
        if fit.fun == np.min(self.lse):
            self.fit_param = fit.x

    def minimize(self):
        self.new_var(self.var0)
        for idx in range(15):
            if np.argmin(self.lse) == np.argmin(self.pars):
                self.new_var(0.95 * np.min(self.pars))
            elif np.argmin(self.lse) == np.argmax(self.pars):
                self.new_var(1.05 * np.max(self.pars))
            else:
                xmin = self.pars[np.argmin(self.lse)]

                diff = np.array(self.pars) - xmin
                xl = np.max(diff[diff < 0]) / 2 + xmin
                xr = np.min(diff[diff > 0]) / 2 + xmin

                if np.abs(xmin - xl) < np.abs(xmin - xr):
                    self.new_var(xr)
                else:
                    self.new_var(xl)
        return self.pars[np.argmin(self.lse)] * self.radius
