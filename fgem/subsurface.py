import pandas as pd
from datetime import timedelta, datetime
import numpy as np
import pdb
from copy import deepcopy
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
import math
from fgem.utils.utils import compute_f, densitywater, viscositywater, heatcapacitywater, vaporpressurewater, nonzero

steamtable = XSteam(XSteam.UNIT_SYSTEM_MKS)

class BaseReservoir(object):
	"""Base class defining subsurface reservoir and wellbore."""
	def __init__(self,
				 Tres_init,
				 geothermal_gradient,
				 surface_temp,
				 L,
				 time_init,
				 well_depth,
				 prd_well_diam,
				 inj_well_diam,
				 num_prd,
				 num_inj,
				 waterloss,
				 power_plant_type,
				 pumpeff,
				 ramey=True,
				 pumping=True,
     			 krock=3,
        		 rhorock=2700,
           		 cprock=1000,
              	 impedance = 0.1,
                 PI = 20, # kg/s/bar
                 II = 20, # kg/s/bar
                 SSR = 1.0,
                 N_ramey_mv_avg=168 # hrs, 1 week worth of accumulation
                 ):
		"""Define attributes of the Subsurface class."""

		self.geothermal_gradient = geothermal_gradient
		self.surface_temp = surface_temp
		self.L = L
		self.time_init = time_init
		self.time_curr = time_init
		self.well_depth = well_depth
		self.num_prd = num_prd
		self.num_inj = num_inj
		self.Tres_init = Tres_init #surface_temp + geothermal_gradient * well_depth/1e3
		self.steamtable = XSteam(XSteam.UNIT_SYSTEM_MKS) #m/kg/sec/°C/bar/W
		self.krock = krock #W/m/K
		self.rhorock = rhorock #J/kg/K
		self.cprock = cprock #J/kg/K
		self.prd_well_diam = prd_well_diam #m
		self.inj_well_diam = inj_well_diam
		self.alpharock = krock/(rhorock*cprock)
		self.ramey = ramey
		self.pumping = pumping
		self.impedance = impedance
		self.PI = SSR * PI if SSR > 0.0 else PI #0.3 Based on GETEM page 61
		self.II = SSR * II if SSR > 0.0 else II #0.3 Based on GETEM page 61
		self.waterloss = waterloss
		self.pumpeff = pumpeff
		self.power_plant_type = power_plant_type
		self.pumpdepth =np.zeros(self.num_prd)
		self.dT_prd = 0.0
		self.dT_inj = 0.0
		self.m_prd_ramey_mv_avg = 100*np.ones(self.num_prd) # randomly initialized, but it will quickly converge following the dispatch strategy
		self.N_ramey_mv_avg = N_ramey_mv_avg

		#initialize wellhead and bottomhole quantities
		self.T_prd_wh = np.array(self.num_prd*[self.Tres_init], dtype='float')
		self.T_prd_bh = np.array(self.num_prd*[self.Tres_init], dtype='float')
		self.T_inj_wh = np.array(self.num_inj*[75], dtype='float')
		self.T_inj_bh = np.array(self.num_inj*[75], dtype='float')
  
		#reservoir hydrostatic pressure [kPa] 
		self.CP = 4.64E-7
		self.CT = 9E-4/(30.796*self.Tres_init**(-0.552))
		self.Phydrostatic = 0+1./self.CP*(math.exp(densitywater(self.surface_temp)*9.81*self.CP/1000*(self.well_depth-self.CT/2*(self.geothermal_gradient/1000)*self.well_depth**2))-1)

	def model(self, t, m_prd, m_inj, T_inj, T_amb):
		raise NotImplementedError
	
	def wellbore_calculations(self, t, m_prd, m_inj, T_inj, T_amb):
		# Ramey's wellbore heat loss model
		if self.ramey:
			self.compute_ramey(t, m_prd, m_inj, T_inj, T_amb)
		else:
			self.T_prd_wh = self.T_prd_bh
			self.T_inj_bh = T_inj
   
		self.dT_prd = self.T_prd_bh - self.T_prd_wh
		self.dT_inj = self.T_inj_bh - T_inj
		self.T_inj_wh = np.array(self.num_inj*[T_inj], dtype='float')

		if self.pumping:
			self.compute_pumpingpower(m_prd, m_inj, T_inj)
		else:
			self.PumpingPower_ideal = 0.0
  
	def step(self, timestep, m_prd, m_inj, T_inj, T_amb):
		"""Stepping the reservoir and wellbore models."""
		self.time_curr += timestep
		self.model(self.time_curr, timestep, m_prd, m_inj, T_inj)
		self.wellbore_calculations(self.time_curr, m_prd, m_inj, T_inj, T_amb)

	def compute_ramey(self, t, m_prd, m_inj, T_inj, T_amb):
     
		"""Ramey's model."""

		# Producer calculations

		self.m_prd_ramey_mv_avg = ((self.N_ramey_mv_avg-1) * self.m_prd_ramey_mv_avg + m_prd)/self.N_ramey_mv_avg
		time_seconds = max((t - self.time_init).seconds, 8760*3600) # always assume one year passed across #
		cpwater = heatcapacitywater(self.T_prd_bh.mean()) # J/kg-degC
		framey = -np.log(1.1*(self.prd_well_diam/2.)/np.sqrt(4.*self.alpharock*time_seconds))-0.29
		rameyA = self.m_prd_ramey_mv_avg*cpwater*framey/2/math.pi/self.krock

		self.T_prd_wh = (self.Tres - (self.geothermal_gradient/1000)*self.well_depth) + \
		(self.geothermal_gradient/1000) * rameyA * (1 - np.exp(-self.well_depth/nonzero(rameyA))) + \
		(self.T_prd_bh - self.Tres) * np.exp(-self.well_depth/nonzero(rameyA))

		# Injector calculations
		cpwater = heatcapacitywater(T_inj)#J/kg-degC
		framey = -np.log(1.1*(self.inj_well_diam/2.)/np.sqrt(4.*self.alpharock*time_seconds))-0.29
		rameyA = m_inj*cpwater*framey/2/math.pi/self.krock

		self.T_inj_bh = (self.surface_temp + (self.geothermal_gradient/1000)*self.well_depth) - \
		(self.geothermal_gradient/1000) * rameyA + \
		(T_inj - self.surface_temp + (self.geothermal_gradient/1000) * rameyA) * np.exp(-self.well_depth/nonzero(rameyA))

		# In cases where a well is not flowing, we adjust welleahd temperature to be the same as surface temperature
		# mask_prd = np.where(m_prd == 0, 0.0, 1.0)
		# self.T_prd_wh = self.T_prd_wh * mask_prd + self.surface_temp * np.ones(len(m_prd))* (1-mask_prd)
		# mask_inj = np.where(m_inj == 0, 0.0, 1.0)
		# self.T_inj_bh = self.T_inj_bh * mask_inj + self.Tres * np.ones(len(m_inj))* (1-mask_inj)

	def compute_pumpingpower(self, m_prd, m_inj, T_inj):
		
		"""Pumping power calculations."""

		# Production wellbore fluid conditions [kPa]
		Tprodaverage = self.T_prd_bh-self.dT_prd/4. #most of temperature drop happens in upper section (because surrounding rock temperature is lowest in upper section)
		rhowaterprod = np.array(list(map(densitywater, Tprodaverage))) #replace with correlation based on Tprodaverage
		muwaterprod = np.array(list(map(viscositywater, Tprodaverage)))
		vprod = m_prd/rhowaterprod/(math.pi/4.*self.prd_well_diam**2)
		Rewaterprod = 4.*m_prd/(muwaterprod*math.pi*self.prd_well_diam) #laminar or turbulent flow?
		f3 = np.array(list(map(lambda x: compute_f(x, self.prd_well_diam), Rewaterprod)))

		# Injection well conditions
		Tinjaverage = T_inj
		rhowaterinj = densitywater(Tinjaverage)
		muwaterinj = viscositywater(Tinjaverage)
		vinj = m_inj*(1.+self.waterloss)/rhowaterinj/(math.pi/4.*self.inj_well_diam**2)
		Rewaterinj = 4.*m_inj*(1.+self.waterloss)/(muwaterinj*math.pi*self.inj_well_diam) #laminar or turbulent flow?
		f1 = np.array(list(map(lambda x: compute_f(x, self.inj_well_diam), Rewaterinj)))

		#reservoir hydrostatic pressure [kPa] 
		self.CT = 9E-4/(30.796*self.Tres**(-0.552))
		self.Phydrostatic = 0+1./self.CP*(math.exp(densitywater(self.surface_temp)*9.81*self.CP/1000*(self.well_depth-self.CT/2*(self.geothermal_gradient/1000)*self.well_depth**2))-1)
  
		# ORC power plant case, with pumps at both injectors and producers
		Pexcess = 344.7 #[kPa] = 50 psi. Excess pressure covers non-condensable gas pressure and net positive suction head for the pump
		self.Pprodwellhead = np.array(list(map(vaporpressurewater, self.T_prd_bh))) + Pexcess #[kPa] is minimum production pump inlet pressure and minimum wellhead pressure
			
		PIkPa = self.PI/(rhowaterprod/1000)/100 #convert PI from l/s/bar to kg/s/kPa

		# Calculate pumping depth ... note, highest pumping requirements happen on day one of the project where vapor pressure closer to wellhead is the highest before the reservoir starts to deplete
		if self.pumpdepth.sum() == 0.0:
			self.pumpdepth = self.well_depth + (self.Pprodwellhead - self.Phydrostatic + m_prd/PIkPa)/(f3*(rhowaterprod*vprod**2/2.)*(1/self.prd_well_diam)/1E3 + rhowaterprod*9.81/1E3)
			self.pumpdepth = np.clip(self.pumpdepth, 0, np.inf)
			pumpdepthfinal = np.max(self.pumpdepth)
			if pumpdepthfinal <= 0:
				pumpdepthfinal = 0
			elif pumpdepthfinal > 600:
				print("Warning: GEOPHIRES calculates pump depth to be deeper than 600 m. Verify reservoir pressure, production well flow rate and production well dimensions")  

		# Calculate production well pumping pressure [kPa]
		DP3 = self.Pprodwellhead - (self.Phydrostatic - m_prd/PIkPa - rhowaterprod*9.81*self.well_depth/1E3 - f3*(rhowaterprod*vprod**2/2.)*(self.well_depth/self.prd_well_diam)/1E3)
		PumpingPowerProd = DP3*m_prd/rhowaterprod/self.pumpeff/1e3 #[MWe] total pumping power for production wells
		self.PumpingPowerProd = np.clip(PumpingPowerProd, 0, np.inf)
		
		IIkPa = self.II/(rhowaterinj/1000)/100 #convert II from l/s/bar to kg/s/kPa
		
		# Necessary injection wellhead pressure [kPa]
		self.Pinjwellhead = self.Phydrostatic + m_inj*(1+self.waterloss)/IIkPa - rhowaterinj*9.81*self.well_depth/1E3 + f1*(rhowaterinj*vinj**2/2)*(self.well_depth/self.inj_well_diam)/1e3

		# Plant outlet pressure [kPa]
		DPSurfaceplant = 68.95 #[kPa] assumes 10 psi pressure drop in surface equipment
		Pplantoutlet = (self.Pprodwellhead - DPSurfaceplant).mean()

		# Injection pump pressure [kPa]
		DP1 = self.Pinjwellhead-Pplantoutlet
		PumpingPowerInj = DP1*m_inj/rhowaterinj/self.pumpeff/1e3 #[MWe] total pumping power for injection wells
		self.PumpingPowerInj = np.clip(PumpingPowerInj, 0, np.inf)

		# Total pumping power
		self.PumpingPower_ideal = self.PumpingPowerInj.sum() + self.PumpingPowerProd.sum()
   
class PercentageReservoir(BaseReservoir):
	"""A class defining subsurface reservoir and wellbore."""
	def __init__(self,
				 Tres_init,
				 geothermal_gradient,
				 surface_temp,
				 L,
				 time_init,
				 well_depth,
				 prd_well_diam,
				 inj_well_diam,
				 num_prd,
				 num_inj,
				 waterloss,
				 power_plant_type,
				 pumpeff,
				 ramey=True,
				 pumping=True,
     			 krock=3,
        		 rhorock=2700,
           		 cprock=1000,
              	 impedance = 0.1,
                 PI = 20, # kg/s/bar
                 II = 20, # kg/s/bar
                 SSR = 1.0,
                 N_ramey_mv_avg=168, # hrs, 1 week worth of accumulation
                 drawdp=0.005,
                 plateau_length=3
                 ):
		"""Define attributes of the Subsurface class."""
		super(PercentageReservoir, self).__init__(Tres_init,
											geothermal_gradient,
											surface_temp,
											L,
											time_init,
											well_depth,
											prd_well_diam,
											inj_well_diam,
											num_prd,
											num_inj,
											waterloss,
											power_plant_type,
											pumpeff,
											ramey,
											pumping,
											krock,
											rhorock,
											cprock,
											impedance,
											PI, # kg/s/bar
											II, # kg/s/bar
											SSR,
											N_ramey_mv_avg)
	
		self.Tres_arr = self.Tres_init*np.ones(self.L+1)
		for i in range(self.L+1):
			if i+1 > plateau_length:
				self.Tres_arr[i] = (1 - drawdp) * self.Tres_arr[i-1]
    
	def model(self, t, timestep, m_prd, m_inj, T_inj):

		"""Calculate reservoir properties and wellbore losses based on well control decisions."""
  
		self.Tres = self.Tres_arr[t.year - self.time_init.year]
		self.T_prd_bh = np.array(self.num_prd*[self.Tres], dtype='float')
		self.T_inj_bh = np.array(self.num_inj*[T_inj], dtype='float')

class EnergyDeclineReservoir(BaseReservoir):
	"""A class defining subsurface reservoir and wellbore."""
	def __init__(self,
				 Tres_init,
				 Pres_init,
				 geothermal_gradient,
				 surface_temp,
				 L,
				 time_init,
				 well_depth,
				 prd_well_diam,
				 inj_well_diam,
				 num_prd,
				 num_inj,
				 waterloss,
				 power_plant_type,
				 pumpeff,
				 ramey=True,
				 pumping=True,
     			 krock=3,
        		 rhorock=2700,
           		 cprock=1000,
              	 impedance = 0.1,
                 PI = 20, # kg/s/bar
                 II = 20, # kg/s/bar
                 SSR = 1.0,
                 N_ramey_mv_avg=24, # hrs, 1 week worth of accumulation
                 V_res=0.005,
                 phi_res=3
                 ):
		"""Define attributes of the Subsurface class."""
		super(EnergyDeclineReservoir, self).__init__(Tres_init,
											geothermal_gradient,
											surface_temp,
											L,
											time_init,
											well_depth,
											prd_well_diam,
											inj_well_diam,
											num_prd,
											num_inj,
											waterloss,
											power_plant_type,
											pumpeff,
											ramey,
											pumping,
											krock,
											rhorock,
											cprock,
											impedance,
											PI, # kg/s/bar
											II, # kg/s/bar
											SSR,
											N_ramey_mv_avg)

		self.Pres_init = Pres_init*1e2
		self.V_res = V_res * 1e9 # m3
		self.phi_res = phi_res
		self.h_res_init = steamtable.h_pt(self.Pres_init/1e2, self.Tres_init)
		self.rho_geof_init = densitywater(self.Tres_init)
		self.M_res = self.rho_geof_init * self.V_res * self.phi_res
		self.energy_res_init = self.M_res * self.h_res_init #kJ reservoir geofluid
		self.energy_res_curr = self.energy_res_init
  
	def model(self, t, timestep, m_prd, m_inj, T_inj):

		"""Calculate reservoir properties and wellbore losses based on well control decisions."""
		mass_produced = m_prd * timestep.total_seconds()
		mass_injected = m_inj * timestep.total_seconds()
		h_prd = np.array([steamtable.h_pt(self.Pres_init/1e2, T) for T in self.T_prd_bh])
		h_inj = np.array([steamtable.hL_t(T) for T in self.T_inj_bh])
		net_energy_produced = (h_prd * mass_produced).sum() - (h_inj * mass_injected).sum() #kg produced
		self.energy_res_curr -= net_energy_produced
		self.Tres = self.Tres_init * (self.energy_res_curr/self.energy_res_init)
		self.T_prd_bh = np.array(self.num_prd*[self.Tres], dtype='float')
  
if __name__ == '__main__':
	pass