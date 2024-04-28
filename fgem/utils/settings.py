import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pdb
from PyEMD import EMD
import random
from copy import deepcopy

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from fgem.world import *
from fgem.subsurface import *
from fgem.powerplant import *
from fgem.markets import *
from fgem.weather import *
from fgem.storage import *
from fgem.utils.utils import prepare_tabular_world
from pymoo.core.variable import Real, Integer, Choice, Binary

def create_IMFs(price, window, window_min=12, max_imf=5):
    window = max(window, window_min)
    price_means = np.zeros(len(price))
    price_stds = 100*np.ones(len(price))
    price_mins = np.zeros(len(price))
    price_maxs = 100*np.ones(len(price))
    
    for i in range(int(len(price)/window)):
        price_means[window*i:window*i+window] = price[window*i:window*i+window].mean()
        price_stds[window*i:window*i+window] = price[window*i:window*i+window].std()
        price_mins[window*i:window*i+window] = price[window*i:window*i+window].min()
        price_maxs[window*i:window*i+window] = price[window*i:window*i+window].max()

    price_window_standard = (price - price_means)/price_stds

    quantity = price_window_standard
    
    emd = EMD()
    IMFs = emd(quantity, max_imf=max_imf).T
    
    return IMFs

def IMFs_to_x(DV, IMFs, x_max, d1=5, transformation_type="neural_network"):
    len_IMFs = IMFs.shape[1]
    percentile1 = DV[0]
    percentile2 = DV[1]
    DV = DV[2:]

    if transformation_type == "linear":
        DV_in = DV[:len_IMFs]
        DV_out = DV[len_IMFs:2*len_IMFs]
        m_in = np.matmul(IMFs,DV_in).squeeze()# np.sum(DV_in * IMFs, axis=1)
        m_out = np.matmul(IMFs,DV_out).squeeze()# np.sum(DV_out * IMFs, axis=1)
        
    elif transformation_type == "quadratic":
        DV_in = DV[:len_IMFs]
        DV_out = DV[len_IMFs:2*len_IMFs]

        m_in = np.matmul(IMFs,DV_in).squeeze()# np.sum(DV_in * IMFs, axis=1)
        m_out = np.matmul(IMFs,DV_out).squeeze()# np.sum(DV_out * IMFs, axis=1)
    elif transformation_type == "neural_network":
        len_W1 = int(IMFs.shape[1]*d1)
        len_W2 = d1
        len_tot = len_W1+len_W2
        DV_in = DV[:len_tot]
        DV_out = DV[len_tot:2*len_tot]

        W1 = DV_in[:len_W1].reshape((IMFs.shape[1], d1))
        W2 = DV_in[len_W1:].reshape((d1, 1))
        m_in = np.matmul(np.maximum(np.matmul(IMFs,W1), 0.0), W2).squeeze()

        W1 = DV_out[:len_W1].reshape((IMFs.shape[1], d1))
        W2 = DV_out[len_W1:].reshape((d1, 1))
        m_out = np.matmul(np.maximum(np.matmul(IMFs,W1), 0.0), W2).squeeze()

    m_in = x_max*m_in
    m_in = np.clip(m_in, 0, np.abs(m_in.max()))
    m_in[m_in < np.percentile(m_in, percentile1)] = 0.0
    
    m_out = x_max*m_out
    m_out = np.clip(m_out, 0, np.abs(m_out.max()))
    m_out[m_out < np.percentile(m_out, percentile2)] = 0.0

    return m_in, m_out

class OptSettings:
    def __init__(self, config):
        self.config = config
        self.config_to_placeholders(self.config)
        self.world = World(self.config)

        self.num_temporal_variables = len(self.temporal_variables)
        self.num_integer_static_variables = len(self.integer_static_variables)
        self.num_real_static_variables = len(self.real_static_variables)
        self.df = prepare_tabular_world(self.world)
        self.annual_price_stats = self.df.groupby(by=["month", "day", "hour"]).mean(numeric_only=True).reset_index(drop=True)
        self.IMFs = create_IMFs(self.annual_price_stats.price.values, self.window, max_imf=self.max_imf)
        self.num_IMFs_vars = self.num_temporal_variables * self.d1 * (self.IMFs.shape[1] + 1)
        self.num_real_variables = int(self.num_IMFs_vars + self.num_real_static_variables)
        self.num_integer_variables = self.num_integer_static_variables
        self.n_var = self.num_real_variables + self.num_integer_variables
        self.xu = [i[1] for i in self.integer_static_variables.values()] + [i[1] for i in self.real_static_variables.values()] + [1 for i in range(self.num_IMFs_vars)]
        self.xl = [i[0] for i in self.integer_static_variables.values()] + [i[0] for i in self.real_static_variables.values()] + [-1 for i in range(self.num_IMFs_vars)]
        assert self.n_var == len(self.xu) == len(self.xl), "Double Check Optimization Setup. Mismatching Dimensions!"

        self.vars = {}
        counter = 1
        for bounds in self.integer_static_variables.values():
            self.vars[f"x{counter}"] = Integer(bounds=tuple(bounds))
            counter += 1
        for bounds in self.real_static_variables.values():
            self.vars[f"x{counter}"] = Real(bounds=tuple(bounds))
            counter += 1
        for _ in range(self.num_IMFs_vars):
            self.vars[f"x{counter}"] = Real(bounds=(-1, 1))
            counter += 1

    def DV_to_inputs_only_tank(self,
                           DV):
        tank_diameter = DV[0]
        x_max = DV[1]
        DV = DV[2:]

        m_tes_ins, m_tes_outs = IMFs_to_x(DV, 
                                        self.IMFs,
                                        d1=self.d1,
                                        transformation_type="neural_network", 
                                        x_max=x_max)
        m_tes_ins = np.tile(m_tes_ins, self.L)
        m_tes_outs = np.tile(m_tes_outs, self.L)

        return m_tes_ins, m_tes_outs, tank_diameter, x_max, self.ppc

    def DV_to_inputs_only_battery(self,
                                  DV):
        duration_1 = DV[0]
        power_capacity_1 = DV[1]
        duration_2 = DV[2]
        power_capacity_2 = DV[3]
        x_max = DV[4]
        DV = DV[5:]

        p_bat_ins, p_bat_outs = IMFs_to_x(DV, 
                                        self.IMFs,
                                        d1=self.d1,
                                        transformation_type="neural_network", 
                                        x_max=x_max)
        
        p_bat_ins = np.tile(p_bat_ins, self.L)
        p_bat_outs = np.tile(p_bat_outs, self.L)

        return p_bat_ins, p_bat_outs, duration_1, power_capacity_1, duration_2, power_capacity_2, x_max, self.ppc


    def DV_to_inputs_both_tank_battery(self,
                                       DV):
        tank_diameter = DV[0]
        x_max_tes = DV[1]
        duration_1 = DV[2]
        power_capacity_1 = DV[3]
        duration_2 = DV[4]
        power_capacity_2 = DV[5]
        x_max_bat = DV[6]
        DV = DV[7:]

        n = int(self.num_IMFs_vars/self.num_temporal_variables)
        tes_indices = np.r_[[0,1], 4:int(2*n)+4]
        bat_indices = np.r_[[2,3], int(2*n)+4:len(DV)]

        DV_tes = DV[tes_indices]
        DV_bat = DV[bat_indices]

        m_tes_ins, m_tes_outs = IMFs_to_x(DV_tes,
                                        self.IMFs,
                                        d1=self.d1,
                                        transformation_type="neural_network", 
                                        x_max=x_max_tes)

        p_bat_ins, p_bat_outs = IMFs_to_x(DV_bat, 
                                        self.IMFs,
                                        d1=self.d1,
                                        transformation_type="neural_network", 
                                        x_max=x_max_bat)
        
        m_tes_ins = np.tile(m_tes_ins, self.L)
        m_tes_outs = np.tile(m_tes_outs, self.L)

        p_bat_ins = np.tile(p_bat_ins, self.L)
        p_bat_outs = np.tile(p_bat_outs, self.L)

        return m_tes_ins, m_tes_outs, p_bat_ins, p_bat_outs, \
            x_max_bat, duration_1, power_capacity_1, duration_2, power_capacity_2, x_max_tes, tank_diameter, self.ppc

    def DV_to_inputs_ppc_only_tank(self,
                           DV):
        self.ppc = DV[0]
        tank_diameter = DV[1]
        x_max = DV[2]
        DV = DV[3:]

        m_tes_ins, m_tes_outs = IMFs_to_x(DV, 
                                        self.IMFs,
                                        d1=self.d1,
                                        transformation_type="neural_network", 
                                        x_max=x_max)
        m_tes_ins = np.tile(m_tes_ins, self.L)
        m_tes_outs = np.tile(m_tes_outs, self.L)

        return m_tes_ins, m_tes_outs, tank_diameter, x_max, self.ppc

    def DV_to_inputs_ppc_only_battery(self,
                                    DV):
        self.ppc = DV[0]
        duration_1 = DV[1]
        power_capacity_1 = DV[2]
        duration_2 = DV[3]
        power_capacity_2 = DV[4]
        x_max = DV[5]
        DV = DV[6:]

        p_bat_ins, p_bat_outs = IMFs_to_x(DV, 
                                        self.IMFs,
                                        d1=self.d1,
                                        transformation_type="neural_network", 
                                        x_max=x_max)
        
        p_bat_ins = np.tile(p_bat_ins, self.L)
        p_bat_outs = np.tile(p_bat_outs, self.L)

        return p_bat_ins, p_bat_outs, duration_1, power_capacity_1, duration_2, power_capacity_2, x_max, self.ppc

    def DV_to_inputs_ppc_both_tank_battery(self,
                                       DV):
        self.ppc = DV[0]
        tank_diameter = DV[1]
        x_max_tes = DV[2]
        duration_1 = DV[3]
        power_capacity_1 = DV[4]
        duration_2 = DV[5]
        power_capacity_2 = DV[6]
        x_max_bat = DV[7]
        DV = DV[8:]

        n = int(self.num_IMFs_vars/self.num_temporal_variables)
        tes_indices = np.r_[[0,1], 4:int(2*n)+4]
        bat_indices = np.r_[[2,3], int(2*n)+4:len(DV)]

        DV_tes = DV[tes_indices]
        DV_bat = DV[bat_indices]

        m_tes_ins, m_tes_outs = IMFs_to_x(DV_tes,
                                        self.IMFs,
                                        d1=self.d1,
                                        transformation_type="neural_network", 
                                        x_max=x_max_tes)

        p_bat_ins, p_bat_outs = IMFs_to_x(DV_bat, 
                                        self.IMFs,
                                        d1=self.d1,
                                        transformation_type="neural_network", 
                                        x_max=x_max_bat)
        
        m_tes_ins = np.tile(m_tes_ins, self.L)
        m_tes_outs = np.tile(m_tes_outs, self.L)

        p_bat_ins = np.tile(p_bat_ins, self.L)
        p_bat_outs = np.tile(p_bat_outs, self.L)

        return m_tes_ins, m_tes_outs, p_bat_ins, p_bat_outs, \
            x_max_bat, duration_1, power_capacity_1, duration_2, power_capacity_2, x_max_tes, tank_diameter, self.ppc

    def DV_to_inputs(self, DV):
        x_max_bat = 0
        duration_1, power_capacity_1 = 0, 0
        duration_2, power_capacity_2 = 0, 0
        x_max_tes = 0
        tank_diameter = 0

        if self.world_type == "tank_only":
            m_tes_ins, m_tes_outs, tank_diameter, x_max_tes, ppc = \
                self.DV_to_inputs_only_tank(DV)
            p_bat_ins = np.zeros(m_tes_ins.shape)
            p_bat_outs = np.zeros(m_tes_outs.shape)

        elif self.world_type == "battery_only":
            p_bat_ins, p_bat_outs, duration_1, power_capacity_1, duration_2, power_capacity_2, x_max_bat, ppc = \
                self.DV_to_inputs_only_battery(DV)
            m_tes_ins = np.zeros(p_bat_ins.shape)
            m_tes_outs = np.zeros(p_bat_outs.shape)

        elif self.world_type == "tank_battery":
            m_tes_ins, m_tes_outs, p_bat_ins, p_bat_outs, \
            x_max_bat, duration_1, power_capacity_1, duration_2, power_capacity_2, x_max_tes, tank_diameter, ppc = \
                self.DV_to_inputs_both_tank_battery(DV)

        elif self.world_type == "ppc_tank_only":
            m_tes_ins, m_tes_outs, tank_diameter, x_max_tes, ppc = \
                self.DV_to_inputs_ppc_only_tank(DV)
            p_bat_ins = np.zeros(m_tes_ins.shape)
            p_bat_outs = np.zeros(m_tes_outs.shape)

        elif self.world_type == "ppc_battery_only":
            p_bat_ins, p_bat_outs, duration_1, power_capacity_1, duration_2, power_capacity_2, x_max_bat, ppc = \
                self.DV_to_inputs_ppc_only_battery(DV)
            m_tes_ins = np.zeros(p_bat_ins.shape)
            m_tes_outs = np.zeros(p_bat_outs.shape)

        elif self.world_type == "ppc_tank_battery":
            m_tes_ins, m_tes_outs, p_bat_ins, p_bat_outs, \
            x_max_bat, duration_1, power_capacity_1, duration_2, power_capacity_2, x_max_tes, tank_diameter, ppc = \
                self.DV_to_inputs_ppc_both_tank_battery(DV)

        else:
            raise ValueError(f"World type {self.world_type} is invalid.")
        
        return m_tes_ins, m_tes_outs, p_bat_ins, p_bat_outs, \
            x_max_bat, duration_1, power_capacity_1, duration_2, power_capacity_2, \
            x_max_tes, tank_diameter, ppc

    def config_to_placeholders(self, config):
        for top_val in config.values():
            for key1, val1 in top_val.items():
                exec("self." + key1 + '=val1')
                if isinstance(val1, dict):
                    for key2, val2 in val1.items():
                        exec("self." + key2 + '=val2')

def prepare_decision_variables(config):
    world_type = config["metadata"]["world_type"]

    if world_type == "tank_only":
        temporal_variables={"imf_tes_in": [-1,1],
                    "imf_tes_out": [-1,1]}
        integer_static_variables= {"tank_diameter": [0,30],
                           "x_max": [0, 1000],
                           "percentile1": [0,100],
                           "percentile2": [0,100]}

    elif world_type == "battery_only":
        temporal_variables={"var_imf_bat_in": [-1,1], 
                    "var_imf_bat_out": [-1,1]}
        integer_static_variables= {"var_duration_1": [0, 24],
                        "var_power_capacity_1": [0, 5000],
                        "var_duration_2": [0, 24],
                        "var_power_capacity_2": [0, 5000],
                        "var_x_max": [0, 5000],
                        "var_percentile1": [0,100],
                        "var_percentile2": [0,100]}

    elif world_type == "tank_battery":
        temporal_variables={"var_imf_tes_in": [-1,1], 
                    "var_imf_tes_out": [-1,1],
                    "var_imf_bat_in": [-1,1],
                    "var_imf_bat_out": [-1,1]}
        integer_static_variables= {"var_tank_diameter": [0,30],
                        "var_x_max_tes": [0, 1000],
                        "var_duration_1": [0, 24],
                        "var_power_capacity_1": [0, 5000],
                        "var_duration_2": [0, 24],
                        "var_power_capacity_2": [0, 5000],
                        "var_x_max_bat": [0, 5000],
                        "var_percentile_tes1": [0,100],
                        "var_percentile_tes2": [0,100],
                        "var_percentile_bat1": [0,100],
                        "var_percentile_bat2": [0,100]}
    
    elif world_type == "ppc_tank_only":
        temporal_variables={"var_imf_tes_in": [-1,1],
                    "var_imf_tes_out": [-1,1]}
        integer_static_variables= {"var_ppc": [1, 1000],
                            "var_tank_diameter": [0,30],
                           "var_x_max": [0, 1000],
                           "var_percentile1": [0,100], 
                           "var_percentile2": [0,100]}

    elif world_type == "ppc_battery_only":
        temporal_variables={"var_imf_bat_in": [-1,1], 
                    "var_imf_bat_out": [-1,1]}
        integer_static_variables= {"var_ppc": [1, 1000],
                        "var_duration_1": [0, 24],
                        "var_power_capacity_1": [0, 5000],
                        "var_duration_2": [0, 24],
                        "var_power_capacity_2": [0, 5000],
                        "var_x_max": [0, 5000],
                        "var_percentile1": [0,100], 
                        "var_percentile2": [0,100]}

    elif world_type == "ppc_tank_battery":
        temporal_variables={"var_imf_tes_in": [-1,1], 
                    "var_imf_tes_out": [-1,1],
                    "var_imf_bat_in": [-1,1],
                    "var_imf_bat_out": [-1,1]}
        integer_static_variables= {"var_ppc": [1, 1000],
                        "var_tank_diameter": [0,30],
                        "var_x_max_tes": [0, 1000],
                        "var_duration_1": [0, 24],
                        "var_power_capacity_1": [0, 5000],
                        "var_duration_2": [0, 24],
                        "var_power_capacity_2": [0, 5000],
                        "var_x_max_bat": [0, 5000],
                        "var_percentile_tes1": [0,100],
                        "var_percentile_tes2": [0,100],
                        "var_percentile_bat1": [0,100],
                        "var_percentile_bat2": [0,100]}

    else:
        raise ValueError(f"World type {world_type} is invalid.")
    
    config["optimization"]["temporal_variables"] = temporal_variables
    config["optimization"]["real_static_variables"] = {}
    config["optimization"]["integer_static_variables"] = integer_static_variables

    settings = OptSettings(config)

    return settings