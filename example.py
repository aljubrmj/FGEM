# -*- coding: utf-8 -*-

"""
Created on January 1th 2023

@author: aljubrmj

"""

#FGEM
#build date: January 1th 2023
#https://github.com/aljubrmj/FGEM


# Import Modules

import os
import sys
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

from fgem import world, utils, subsurface, powerplant
from fgem.utils.utils import FastXsteam


# Load Project Configuration (Make sure you set it up to your desired parameters at first)
config_filename = "config.json"
config = utils.config.get_config_from_json(config_filename)
project = world.World(config)

# Trading Strategy: Fixed Mass Flow Rates
def constant_strategy(project, mass_flow=100):
    """Constant change producer mass flow rates"""
    m_prd = np.array(project.num_prd*[mass_flow]).astype(float)
    m_inj = np.array(project.num_inj*[m_prd.sum()/project.num_inj]).astype(float)
    return m_prd, m_inj

# Trading Stretegy: Wellhead Throttling
def maximal_power_generation_strategy(project, max_mass_flow=200):
    """Control wells to maintain a constant power plant output"""
    power_output_MWh_kg = project.pp.compute_power_output(project.reservoir.T_prd_wh.mean(), project.state.T0)
    required_mass_flow_per_well = project.ppc / (power_output_MWh_kg * 3600 * project.num_prd + utils.constants.SMALL_NUM)
    
    m_prd = np.minimum(max_mass_flow, np.array(project.num_prd*[required_mass_flow_per_well])).astype(float)
    m_inj = np.array(project.num_inj*[m_prd.sum()/project.num_inj]).astype(float)

    return m_prd, m_inj


# Run Simulation
for _ in tqdm(range(project.max_simulation_steps)):
    m_prd, m_inj = maximal_power_generation_strategy(project)
    project.step(m_prd=m_prd, m_inj=m_inj)
    
project.compute_economics()
NPV, ROI, PBP, PPA_NPV, PPA_ROI, PPA_PBP, LCOE, df_annual = utils.utils.compute_npv(project.df_records, project.capex_total, project.opex_total,
                                                                      project.baseline_year, project.L, project.d, ppa_price=75)
# Print and Visualize 
print(f"LCOE: {LCOE:.0f} $/MWh")
print(f"NPV: {NPV:.0f} $MM")
print(f"ROI: {ROI:.1f} %")
print(f"PBP: {PBP:.0f} yrs")

utils.utils.plot_ex({"": project.present_capex_per_unit}, figsize=(4,4), dpi=100, fontsize=6)

qdict = {"LMP [$/MWh]": "Electricity Price \n [$/MWh]",
          "Atm Temp [deg C]": "Ambient Temp. \n [$\degree C$]",
          "Res Temp [deg C]": "Reservoir Temp. \n [$\degree C$]",
          "WH Temp [deg C]": "Wellhead Temp. \n [$\degree C$]",
          'Inj Temp [deg C]': "Injection Temp. \n [$\degree C$]",
          "Net Power Output [MWe]": "Net Generation \n [MWh]",
          'M_Produced [kg/s]': "Field Production \n [kg/s]",
          "Pumping Power [MWe]": "Pumping Power \n [MWe]"}

quantities = list(qdict.keys())
ylabels = list(qdict.values())

span = range(1, project.max_simulation_steps)
utils.utils.plot_cols_v2({" ": project.df_records}, span, quantities, 
                         figsize=(10,13), ylabels=ylabels, legend_loc=False, dpi=100, formattime=False)

