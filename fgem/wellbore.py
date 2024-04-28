import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.integrate import solve_ivp
from pyXSteam.XSteam import XSteam

steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS) # m/kg/sec/°C/bar/W

def compute_f(rho, u, d, mu, e, correlation="Swamee-Jain"):
    
    Re = rho * u * d / mu
    
    if correlation == "Blasius":
        # Blasius
        f = 0.316 / Re**(1/4)
    elif correlation == "Swamee-Jain":
        # Swamee-Jain
        f = 0.25/(np.log10(e/(3.7*d)+5.74/Re**0.9))**2
    else:
        raise ValueError(f"Correlation '{correlation}' does not exist. Choose either 'Blasius' or 'Swamee-Jain'.")
    
    return f, Re


def FloWellFrictionCorrectionFactor(u, x, T, rhol, rhog, mul, mug, d, 
                                    correlation="Beattie", g=9.81):
    T += 273.15
    if correlation == "Friedel":
        # Friedel
        Rel = rhol * u * d / mul
        fl = 0.316 / Rel**(1/4)
        Reg = rhog * u * d / mug
        fg = 0.316 / Reg**(1/4)

        rhom = 1 / (x / rhog + (1-x) / rhol)
        Tc = 647.096
        sigma = 0.2358*(1-T/Tc)**1.256*(1-0.625*(1-T/Tc))

        E = (1 - x**2) + x**2 * rhol/rhog * fg / fl
        F = x**0.78 * (1 - x**2)**0.24
        H = (rhol/rhog)**0.91*(mug*rhol/(mul*rhog))**0.19*(1-rhog/rhol)**0.7
        Fr = rhol**2*u**2/(g*rhom**2*d)
        We = rhol**2*u**2*d/(sigma*rhom**2)

        phi2 = E + 3.24*F*H / (Fr**0.045*We**0.035)
        
    elif correlation == "Beattie":
        # Beattie
        phi2 = (1+x*(rhol/rhog-1))**0.8*(1+x*((3.5*mug+2*mul)/((mug+mul)*rhog)-1))**0.2
    else:
        raise ValueError(f"Correlation '{correlation}' does not exist. Choose either 'Friedel' or 'Beattie'.")
        
    return phi2


def water_viscosity(T, rho):
    # Dynamic viscosity of pure water in [Pa s]
    # Input temperature is in C and density in [kg/m^3]
    
    T = (T+273.15) / 647.096
    rho = rho / 322
    
    i = np.array([0, 1, 2, 3])
    H = np.array([1.67752, 2.20462, 0.6366564, -0.241605])
    
    mu_0 = 100 * np.sqrt(T) / np.sum(H / T ** i)
    
    i = np.array([0, 1, 2, 3, 0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 0, 1, 0, 3, 4, 3, 5])
    j = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6])
    h = np.array([5.20094e-1, 8.50895e-2, -1.08374, -2.89555e-1, 2.22531e-1, 9.99115e-1,
                  1.88797, 1.26613, 1.20573e-1, -2.81378e-1, -9.06851e-1, -7.72479e-1,
                  -4.89837e-1, -2.57040e-1, 1.61913e-1, 2.57399e-1, -3.25372e-2,
                  6.98452e-2, 8.72102e-3, -4.35673e-3, -5.93264e-4])
    H = np.zeros((6, 7))
    H[i, j] = h
    
    mu_1 = np.exp(rho * ((1 / T - 1) ** np.arange(6)).dot(H.dot((rho - 1) ** np.arange(7))))
    
    mu = 1e-6 * mu_0 * mu_1
    
    return mu

def FloWellVoidFraction(x, rhol, rhog, mul, mug, T, dotm, A, void_fraction_correlation, g=9.81):
    T += 273.15 # K
    if void_fraction_correlation == "Homogeneous":
        # Homogeneous
        S = 1
        alpha = x / rhog / (x / rhog + (1 - x) / rhol * S)
    elif void_fraction_correlation == "Zivi":
        # Zivi
        S = (rhol / rhog) ** (1 / 3)
        alpha = x / rhog / (x / rhog + (1 - x) / rhol * S)
    elif void_fraction_correlation == "Chrisholm":
        # Chrisholm
        S = (1 - x * (1 - rhol / rhog)) ** 0.5
        alpha = x / rhog / (x / rhog + (1 - x) / rhol * S)
    elif void_fraction_correlation == "Premoli":
        # Premoli
        d = (A * 4 / np.pi) ** 0.5
        sigma = 0.2358 * (1 - T / 647.096) ** 1.256 * (1 - 0.625 * (1 - T / 647.096))
        G = dotm / A
        Rel = G * d / mul
        Wel = G ** 2 * d / (sigma * rhol)
        y = (((1 - x) / x) * (rhog / rhol)) ** -1
        F2 = 0.0273 * Wel * Rel ** -0.51 * (rhol / rhog) ** -0.08
        F1 = 1.578 * Rel ** -0.19 * (rhol / rhog) ** 0.22
        Aprm = 1 + F1 * (y / (1 + y * F2) - y * F2)
        alpha = (1 + Aprm * ((1 - x) / x) * (rhog / rhol)) ** -1
    elif void_fraction_correlation == "Lockhart":
        # Lockhart Martinelli
        alpha = 1 / (1 + 0.28 * ((1 - x) / x) ** 0.64 * (rhog / rhol) ** 0.36 * (mul / mug) ** 0.07)
    elif void_fraction_correlation == "Rouhani-Axelsson":
        # Rouhani-Axelsson
        sigma = 0.2358 * (1 - T / 647.096) ** 1.256 * (1 - 0.625 * (1 - T / 647.096))
        G = dotm / A
        alpha = (x / rhog) * ((1 + 0.12 * (1 - x)) * (x / rhog + (1 - x) / rhol) + (1.18 * (1 - x)) * (
                    g * sigma * (rhol - rhog) ** 0.25) / (G * rhol ** 0.5)) ** -1
    
    return alpha

def FloWellGammaEta(p, h, dotm, A, void_fraction_correlation):
    T, x = steamTable.t_ph(p/1e5, h/1e3), steamTable.x_ph(p/1e5, h/1e3)
    rhol, rhog = steamTable.rhoL_p(p/1e5), steamTable.rhoV_p(p/1e5)
    mul, mug = water_viscosity(T, rhol), water_viscosity(T, rhog)

    alpha = FloWellVoidFraction(x, rhol, rhog, mul, mug, T, dotm, A, void_fraction_correlation)
    g = (1-x)**3 / (1-alpha)**2 + (rhol/rhog)**2 * x**3 / alpha**2
    e = (1-x)**2 / (1-alpha) + (rhol/rhog) * x**2 / alpha
        
    return g, e

def FloWellSystemOfEquations(z, s, d, dotm, e, dotQ,
                             friction_correlation="Swamee-Jain", 
                             friction_correction_correlation="Beattie",
                             void_fraction_correlation="Rouhani-Axelsson",
                             g=9.81,
                             return_ulug=False):
    A = np.pi * d**2 / 4
    u = s[0] #m/s
    p = s[1] #Pa
    h = s[2] #J/kg
    
    rho, rhol, rhog = steamTable.rho_ph(p/1e5, h/1e3), steamTable.rhoL_p(p/1e5), steamTable.rhoV_p(p/1e5)
    T, x = steamTable.t_ph(p/1e5, h/1e3), steamTable.x_ph(p/1e5, h/1e3)

    if (x == 0) or (x == 1):
        mu = steamTable.my_ph(p/1e5, h/1e3)
        f, Re = compute_f(rho, u, d, mu, e, friction_correlation)
        rho_p = (steamTable.rho_ph(p/1e5 * (1 + 1e-5), h/1e3) - rho) / (p * 1e-5)
        rho_h = (steamTable.rho_ph(p/1e5, h/1e3 * (1 + 1e-5)) - rho) / (h * 1e-5)
        C = np.array([[rho, u * rho_p, u * rho_h], [dotm * u, 0, dotm], [rho * u, 1, 0]])
        b = np.array([0, dotm * g + dotQ, rho * g + rho * f/(2*d) * u**2])

        if x == 0:
            ug = 0
            ul = u
        else:
            ug = u
            ul = 0

        ds = -np.linalg.solve(C, b)
    
    else:
        # Estimating derivatives 
        gamma, eta = FloWellGammaEta(p, h, dotm, A, void_fraction_correlation);
        gp,ep = FloWellGammaEta(p*(1+1e-5),h, dotm, A, void_fraction_correlation);
        gh,eh = FloWellGammaEta(p,h*(1+1e-5), dotm, A, void_fraction_correlation);
        gamma_p = (gp - gamma) / (p * (1+1e-5))
        gamma_h = (gh - gamma) / (h * (1+1e-5))
        eta_p = (ep - eta) / (p * (1+1e-5))
        eta_h = (eh - eta) / (h * (1+1e-5))

        rhol_p = (steamTable.rhoL_p(p/1e5 * (1 + 1e-5)) - rhol) / (p * 1e-5);

        mul = water_viscosity(T, rhol)
        mug = water_viscosity(T, rhog)
        alpha = FloWellVoidFraction(x, rhol, rhog, mul, mug, T, dotm, A, void_fraction_correlation);
        
        f, Re = compute_f(rhol, u, d, mul, e, friction_correlation)

        phi2 = FloWellFrictionCorrectionFactor(u, x, T, rhol, rhog, mul, mug, d, friction_correction_correlation)
        C = np.array([[rhol, u*rhol_p, 0], [gamma*u, u**2/2*gamma_p, (1+u**2/2*gamma_h)], 
                      [eta*rhol*u, (1+rhol*u**2*eta_p+eta*u**2*rhol_p), rhol*u**2*eta_h]])
        b = np.array([0, g+dotQ/dotm, ((1-alpha)*rhol+alpha*rhog)*g + phi2 * rhol * f / (2 * d) * u**2])

        ug = (x*rhol*u)/(alpha*rhog)
        ul = (1-x)*u/(1-alpha)

        ds = -np.linalg.solve(C, b)
    
    if return_ulug:
        return ds, ul, ug
    else:
        return ds

def ramey_dotQ(tramey, rs, ks, Tf, Te, kres, alpharamey):
    A = np.pi * rs[0]**2

    f = np.log(2 * np.sqrt(alpharamey * tramey)/rs[-1]) - 0.29
    U = 1/np.sum(rs[1:] * np.log(rs[1:]/rs[:-1]) / ks)
    omega = kres/(rs[1] * U)
    Th = (Tf * f + omega * Te)/(f + omega)
    dotQ = 2 * np.pi * kres * (Th - Te)/f

    Ts = [Tf]
    for i, (k, ro, ri) in enumerate(zip(ks, rs[1:], rs[:-1])):
        Ts.append(Ts[i] - dotQ * np.log(ro/ri)/(2 * np.pi * k))
    Ts.append(Th)
    
    return dotQ, Ts

def compute_Te(H, segments, surface_temp=20):
    sortmask = np.argsort([s["H"] for s in segments])
    depths = np.array([s["H"] for s in segments[sortmask]])
    ggs = np.array([s["gg"] for s in segments[sortmask]])

    Hmask = depths <= H
    Te = surface_temp + np.sum(np.diff(depths[Hmask])/1000 * ggs[Hmask][1:])
    return Te

if __name__ == '__main__':
	pass