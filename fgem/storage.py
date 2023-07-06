import os
import sys
import math
import numpy as np
from scipy.optimize import fsolve, root, leastsq
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
import pickle
from fgem.utils.utils import FastXsteam

steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)  # m/kg/sec/°C/bar/W
from fgem.utils.utils import densitywater, viscositywater, heatcapacitywater

class LiIonBattery:

	"""Energy storage class."""

	def __init__(self,
				 time_init,
				 duration=[4,4],
				 power_capacity=[5,5],
				 roundtrip_eff=0.90,
				 lifetime=20):

		"""Intiating attributes for LiIonBattery calss."""
		
		self.time_init = time_init
		self.start_year = time_init.year
		self.time_curr = time_init
		self.duration_list = duration
		self.power_capacity_list = power_capacity

		self.duration = self.duration_list[0] # if self.power_capacity_list[0] > 0.0 else 0.0
		self.power_capacity = self.power_capacity_list[0]# if self.duration_list[0] > 0.0 else 0.0
		self.energy_capacity = self.power_capacity * self.duration
		self.roundtrip_eff = roundtrip_eff
		self.energy_content = 0 #MWh
		self.SOC = self.energy_content / (self.energy_capacity+1e-3) * 100 #%
		self.lifetime = lifetime
		self.first_unit_active = True

	def step(self,
			 timestep,
			 p_bat_in=0,
             p_bat_out=0
			 ):

		"""Stepping energy storage unit over time."""

		violation = False
  
		self.time_curr += timestep
		# Install a new battery and block dis/charge
		if self.first_unit_active and (self.time_curr.year - self.start_year) == self.lifetime:
			self.duration = self.duration_list[1] # if self.power_capacity_list[1] > 0.0 else 0.0
			self.power_capacity = self.power_capacity_list[1] # if self.duration_list[1] > 0.0 else 0.0
			self.energy_capacity = self.power_capacity * self.duration
			self.energy_content = 0.0 #MWh
			self.SOC = self.energy_content / (self.energy_capacity+1e-3) * 100 #%
			self.first_unit_active = False
			return 0.0, 0.0

		# Make sure you respect the available battery energy capacity and content when dis/charging
		new_energy_content = self.energy_content + (p_bat_in - p_bat_out) * timestep.total_seconds()/3600
		if (new_energy_content > self.energy_capacity) or (new_energy_content < 0):
			p_bat_in, p_bat_out = 0.0, 0.0
			violation = True

		self.energy_content += (p_bat_in - p_bat_out) * timestep.total_seconds()/3600 # MWhe
		self.SOC = self.energy_content/ (self.energy_capacity+1e-3) * 100 #%
		return violation
		
class TES:

	"""Thermal energy storage class."""

	def __init__(self,
				 time_init,
				 Vw=250,
				 Tw=40,
				 pressurized_tank=False,
				 d=34,
				 H=13,
				 Lst=0.08,
				 Lins=0.05,
				 Tamb=20,
				 max_thresh=0.95):

		self.time_init = time_init
		self.time_curr = time_init

		self.fxsteam = pickle.load(open('FastXsteam.pkl', 'rb'))
		self.d, self.H, self.Lst, self.Lins = d, H, Lst, Lins
		self.Tw, self.Tamb, self.pressurized_tank, self.max_thresh = Tw, Tamb, pressurized_tank, max_thresh

		# Initiate tank geometric dimensions
		self.initiate_tank_dims()

		# Compute tank contents
		self.initiate_tank_contents(Vw)

	def initiate_tank_dims(self):

		"""Initiating volumetric and thermodynamic quantities in the storage tank."""

		self.VTank = math.pi/4*self.d**2*self.H
		self.V_limits = max((1 - self.max_thresh) * self.VTank, 500)
		self.r_inner_st = self.d/2
		self.r_outer_st = self.d/2 + self.Lst
		self.r_inner_ins = self.r_outer_st
		self.r_outer_ins = self.r_inner_ins + self.Lins
		self.Ast = math.pi*self.r_outer_st**2
		self.Ains = math.pi*self.r_outer_ins**2
		self.SA = 2*math.pi*self.r_outer_ins*self.H + math.pi*self.r_outer_ins**2

	def initiate_tank_contents(self, Vw):
     
		"""Initiating the storage tank contents of liquid and steam."""
  
		# if Vw input is fraction, then it is portion of VTank
		self.Vw = Vw*self.VTank if Vw < 1.0 else Vw
		self.Vl = self.Vw
		self.Va = self.VTank - self.Vl
		self.rhol = densitywater(self.Tw)
		self.mass_max_charge = self.max_thresh*self.Va*self.rhol
		self.massw = self.Vw*self.rhol

		# in case tank is pressurized (e.g. air over water), there will only be liquid inside the tank
		if self.pressurized_tank:
			self.v = 1/self.rhol
			self.x = 0.0
		else:
			self.v = self.VTank/self.massw
			self.x = (self.v - steamTable.vL_t(self.Tw)) / \
				(steamTable.vV_t(self.Tw) - steamTable.vL_t(self.Tw))

		self.massl = (1-self.x) * self.massw
		self.mass_max_discharge = 0.0
		self.massv = self.x * self.massw

	def heatloss(self,
				 timestep,
				 kst=45,
				 kins=0.17,
				 h=13,
				 emmissivity=1.0,
				 sig=5.67e-8):

		"""Estimating heat loss in the thermal water tank."""

		Th = self.Tw + 273.15
		Tamb = self.Tamb + 273.15
		self.cpl = heatcapacitywater(self.Tw)*1000
		self.cpv = steamTable.CpV_t(self.Tw)*1000

		C1 = 2*math.pi*self.H*kst / \
			np.log(self.r_outer_st/self.r_inner_st) + self.Ast*kst/self.Lst
		C2 = 2*math.pi*self.H*kins / \
			np.log(self.r_outer_ins/self.r_inner_ins) + \
			self.Ains*kins/self.Lins
		C3 = self.SA*h
		C4 = self.SA*sig*emmissivity
		C5 = (C3+C1*C2/(C1+C2))
		C6 = C4*Tamb**4+C3*Tamb+(C5-C3)*Th

		def func_explicit(temp):
			return [C4*temp[0]**4 + C5*temp[0] - C6]

		root = fsolve(func_explicit, [self.Tw+273.15])
		Tcold_ins = root[0]
		Tcold = (C1 * Th + C2 * Tcold_ins)/(C1+C2)

		heatLossSteel = 2*math.pi*self.H*kst * \
			(Th-Tcold) / np.log(self.r_outer_st/self.r_inner_st) + \
			self.Ast*kst*(Th-Tcold)/self.Lst
		tempLossSteel = heatLossSteel*timestep / \
			(self.cpl*self.massl + self.cpv*self.massv)
		Th = Th-tempLossSteel
		self.Q = float(heatLossSteel)
		self.HL = self.Q*timestep/1e3  # kJ #from both liquid/steam
		self.dT = float(tempLossSteel)
		self.Tw = float(Th-273.15)

	def conservation(self,
					 m_in,
					 m_out,
					 TWH):

		"""Solving for mass and energy conservation over a timestep."""

		self.massw = (self.massv + self.massl + m_in - m_out)
		
		# Ensure no excessive discharge and reset using threshold of 0.x
		if self.massw < 0:
			# m_in = m_out
			m_out = 0.75 * (self.massv + self.massl + m_in)
			self.massw = (self.massv + self.massl + m_in - m_out)

		hvt = self.fxsteam.hV_t(self.Tw)
		hlt = self.fxsteam.hL_t(self.Tw)
		hltwh = self.fxsteam.hL_t(TWH)

		def equations(variables):
			T = np.clip(variables, 0.1, TWH)
			x = np.clip((self.VTank/self.massw - self.fxsteam.vL_t(T)) / \
				(self.fxsteam.vV_t(T) - self.fxsteam.vL_t(T)), 0.0, 1.0)
			mv, ml = x*self.massw, (1-x)*self.massw
			# energy balance
			f1 = mv * self.fxsteam.hV_t(T) + ml * self.fxsteam.hL_t(T) - \
				(self.massv * hvt + self.massl * hlt +
				 m_in * hltwh - m_out * hlt - self.HL)
			return [f1]

		output = leastsq(equations,
						 (self.Tw),
						 xtol=1e-12,
						 gtol=1e-12,
						 ftol=1e-12)

		self.Tw = np.clip(output[0][0], 0.1, TWH)
		self.x = np.clip((self.VTank/self.massw - steamTable.vL_t(self.Tw)) / \
			(steamTable.vV_t(self.Tw) - steamTable.vL_t(self.Tw)), 0.0, 1.0)
		self.massv = self.x*self.massw
		self.massl = (1-self.x)*self.massw
		self.v = self.VTank/self.massw
		# energy change for the entire system (both water and steam)
		self.E = (self.massv * steamTable.hV_t(self.Tw) +
				  self.massl * steamTable.hL_t(self.Tw))/3600  # kWh

		self.Vw = self.massw*self.v
		self.rhol = densitywater(self.Tw)
		self.Vl = self.massl/self.rhol
		self.Va = self.VTank - self.Vl
		self.mass_max_charge = max((self.Va - self.V_limits)*self.rhol, 0.0)
		self.mass_max_discharge = max((self.Vl - self.V_limits)*self.rhol, 0.0)
		self.conservation_errors = 0

		return m_in, m_out

	def step(self,
			 timestep,
			 T_amb,
			 m_tes_in,
			 m_tes_out,
			 T_in
			 ):

		"""Stepping the thermal water tank over time."""

		self.time_curr += timestep
		self.T_amb = T_amb
		timestep_seconds = timestep.total_seconds()
		self.heatloss(timestep_seconds)
		m_in, m_out = self.conservation(m_tes_in * timestep_seconds, m_tes_out * timestep_seconds, T_in)
		m_tes_in, m_tes_out = m_in/timestep_seconds, m_out/timestep_seconds
		return m_tes_in, m_tes_out
		

