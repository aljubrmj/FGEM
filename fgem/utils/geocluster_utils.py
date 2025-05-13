import numpy as np
import h5py
from scipy.interpolate import interpn
import scipy.io
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
import sys
import os

import numpy as np
import h5py
from scipy.interpolate import interpn
import itertools as iter

class data:
  def __init__(self, fname, case, fluid):

    self.fluid = fluid
    self.case = case

    with h5py.File(fname, 'r') as file:
      fixed_loc = "/" + case + "/fixed_params/"
      input_loc = "/" + case + "/" + fluid + "/input/"
      output_loc = "/" + case + "/" + fluid + "/output/"

      # independent vars
      self.mdot = file[input_loc + "mdot"][:]  # i0
      self.L2 = file[input_loc + "L2"][:]  # i1
      self.L1 = file[input_loc + "L1"][:]  # i2
      self.grad = file[input_loc + "grad"][:]  # i3
      self.D = file[input_loc + "D"][:]  # i4
      self.Tinj = file[input_loc + "T_i"][:]  # i5
      self.k = file[input_loc + "k_rock"][:]  # i6
      self.time = file[input_loc + "time"][:]  # i7
      self.ivars = (self.mdot, self.L2, self.L1, self.grad, self.D, self.Tinj, self.k, self.time)

      # fixed vars
      self.Pinj = file[fixed_loc + "Pinj"][()]
      self.Tamb = file[fixed_loc + "Tamb"][()]

      # dim = Mdot x L2 x L1 x grad x D x Tinj x k
      self.Wt = file[output_loc + "Wt"][:]  # int mdot * dh dt
      self.We = file[output_loc + "We"][:]  # int mdot * (dh - Too * ds) dt

      self.GWhr = 1e6 * 3600000.0

      self.kWe_avg = self.We * self.GWhr / (1000. * self.time[-1] * 86400. * 365.)
      self.kWt_avg = self.Wt * self.GWhr / (1000. * self.time[-1] * 86400. * 365.)

      # dim = Mdot x L2 x L1 x grad x D x Tinj x k x time
      self.shape = (
          len(self.mdot),
          len(self.L2),
          len(self.L1),
          len(self.grad),
          len(self.D),
          len(self.Tinj),
          len(self.k),
          len(self.time))
      self.Tout = self.__uncompress(file, output_loc, "Tout")
      self.Pout = self.__uncompress(file, output_loc, "Pout")

    self.CP_fluid = "CO2"
    if (fluid == "H2O"):
      self.CP_fluid = "H2O"

  def __uncompress(self, file, output_loc, state):
    U = file[output_loc + state + "/" + "U"][:]
    sigma = file[output_loc + state + "/" + "sigma"][:]
    Vt = file[output_loc + state + "/" + "Vt"][:]
    M_k = np.dot(U, np.dot(np.diag(sigma), Vt))

    shape = self.shape
    valid_runs = np.argwhere(np.isfinite(self.We.flatten()))[:, 0]
    M_k_full = np.full((shape[-1], np.prod(shape[:-1])), np.nan)
    M_k_full[:, valid_runs] = M_k
    return np.reshape(M_k_full.T, shape)

  def interp_outlet_states(self, point):

    points = list(iter.product(
            (point[0],),
            (point[1],),
            (point[2],),
            (point[3],),
            (point[4],),
            (point[5],),
            (point[6],),
            self.time))
    Tout = interpn(self.ivars, self.Tout, points)
    Pout = interpn(self.ivars, self.Pout, points)

    return Tout, Pout


  def interp_kWe_avg(self, point):
    ivars = self.ivars[:-1]
    return self.GWhr * interpn(ivars, self.We, point) / (1000. * self.time[-1] * 86400. * 365.)
    
  def interp_kWt_avg(self, point):
    ivars = self.ivars[:-1]
    return self.GWhr * interpn(ivars, self.Wt, point) / (1000. * self.time[-1] * 86400. * 365.)

class TEA:
    def __init__(self, Fluid,End_use,Configuration,Flow_user,Hor_length_user,Depth_user,Gradient_user,Diameter_user,Tin_user,krock_user,Drilling_cost_per_m,O_and_M_cost_plant,Discount_rate,Pump_efficiency,Lifetime,Direct_use_heat_cost_per_kWth,Electricity_rate,Power_plant_cost_per_kWe,T0,P0,Turbine_isentropic_efficiency,Generator_efficiency,Compressor_isentropic_efficiency,Pre_Cooling_Delta_T,Turbine_outlet_pressure):
        self.Fluid = Fluid
        self.End_use = End_use
        self.Configuration = Configuration
        self.Flow_user = Flow_user
        self.Hor_length_user = Hor_length_user
        self.Depth_user = Depth_user
        self.Gradient_user = Gradient_user
        self.Diameter_user = Diameter_user
        self.Tin_user = Tin_user
        self.krock_user = krock_user
        self.Drilling_cost_per_m = Drilling_cost_per_m
        self.O_and_M_cost_plant = O_and_M_cost_plant
        self.Discount_rate = Discount_rate
        self.Pump_efficiency = Pump_efficiency
        self.Lifetime = Lifetime
        self.Direct_use_heat_cost_per_kWth = Direct_use_heat_cost_per_kWth
        self.Electricity_rate = Electricity_rate
        self.Power_plant_cost_per_kWe = Power_plant_cost_per_kWe
        self.T0 = T0
        self.P0 = P0
        self.Turbine_isentropic_efficiency = Turbine_isentropic_efficiency
        self.Generator_efficiency = Generator_efficiency
        self.Compressor_isentropic_efficiency = Compressor_isentropic_efficiency
        self.Pre_Cooling_Delta_T = Pre_Cooling_Delta_T
        self.Turbine_outlet_pressure = Turbine_outlet_pressure

        #self.filename = 'clgs_results.h5'                #Filename of h5 database with simulation results [-]
        self.filename = 'clgs_results_final.h5'                #Filename of h5 database with simulation results [-]
        self.Number_of_points_per_year = 4               #Number of time steps per year in database [-] (must be 4)
        
        self.point = (Flow_user, Hor_length_user, Depth_user, Gradient_user, Diameter_user, Tin_user, krock_user)
        
        self.P_in = 2e7         #Constant Injection pressure [Pa]
        self.T_in = Tin_user-273.15   #Injection temperature [deg.C]
        
    def verify(self): #Verify inputs are within allowable bounds
        self.error = 0 
        if self.Fluid != 1 and self.Fluid !=2:
            print("Error: Fluid must be 1 (H2O) or 2 (CO2). Simulation terminated.")
            self.error = 1
        if self.End_use != 1 and self.End_use !=2:
            print("Error: End_use must be 1 (Direct-Use) or 2 (Electricity). Simulation terminated.")
            self.error = 1
        if self.Flow_user < 5 or self.Flow_user > 100:
            print("Error: Flow rate must be between 5 and 100 kg/s. Simulation terminated.")
            self.error = 1
        if self.Hor_length_user < 1000 or self.Hor_length_user > 20000:
            print("Error: Horizontal length must be between 1,000 and 20,000 m. Simulation terminated.")
            self.error = 1
        if self.Depth_user < 1000 or self.Depth_user > 5000:
            print("Error: Vertical depth must be between 1,000 and 5,000 m. Simulation terminated.")
            self.error = 1
        if self.Gradient_user < 0.03 or self.Gradient_user > 0.07:
            print("Error: Geothermal gradient must be between 0.03 and 0.07 degrees C per m. Simulation terminated.")
            self.error = 1
        if self.Diameter_user < 0.2159 or self.Diameter_user > 0.4445:
            print("Error: Wellbore diameter must be between 0.2159 and 0.4445 m. Simulation terminated.")
            self.error = 1
        if self.Tin_user < 303.15 or self.Tin_user > 333.15:
            print("Error: Injection temperature must be between 303.15 and 333.15 K. Simulation terminated.")
            self.error = 1
        if self.krock_user < 1.5 or self.krock_user > 4.5:
            print("Error: Rock thermal conductivity must be between 1.5 and 4.5 W/m/K. Simulation terminated.")
            self.error = 1
        if self.Drilling_cost_per_m < 0 or self.Drilling_cost_per_m > 10000:
            print("Error: Drilling costs per m of measured depth must be between 0 and 10,000 $/m. Simulation terminated.")
            self.error = 1
        if self.O_and_M_cost_plant < 0 or self.O_and_M_cost_plant > 0.2:
            print("Error: Operation & maintance cost of surface plant (expressed as fraction of total surface plant capital cost) must be between 0 and 0.2. Simulation terminated.")
            self.error = 1
        if self.Discount_rate < 0 or self.Discount_rate > 0.2:
            print("Error: Discount rate must be between 0 and 0.2. Simulation terminated.")
            self.error = 1
        if self.Pump_efficiency < 0.5 or self.Pump_efficiency > 1:
            print("Error: Pump efficiency must be between 0.5 and 1. Simulation terminated.")
            self.error = 1
        if self.Lifetime < 5 or self.Lifetime > 40:
            print("Error: System lifetime must be between 5 and 40 years. Simulation terminated.")
            self.error = 1
        if isinstance(self.Lifetime, int) == False:
            print("Error: System lifetime must be integer. Simulation terminated.")
            self.error = 1
        if self.End_use == 1:    
            if self.Direct_use_heat_cost_per_kWth < 0 or self.Direct_use_heat_cost_per_kWth > 10000:
                print("Error: Capital cost for direct-use surface plant must be between 0 and 10,000 $/kWth. Simulation terminated.")
                self.error = 1
            if self.Electricity_rate < 0 or self.Electricity_rate > 0.5:
                print("Error: Electricity rate in direct-use for pumping power must be between 0 and 0.5 $/kWh. Simulation terminated.")
                self.error = 1
        if self.End_use == 2:    
            if self.Power_plant_cost_per_kWe < 0 or self.Power_plant_cost_per_kWe > 10000:
                print("Error: Power plant capital cost must be between 0 and 10,000 $/kWe. Simulation terminated.")
                self.error = 1
            if self.T0 < 278.15 or self.T0 > 303.15:
                print("Error: Dead-state temperature must be between 278.15 and 303.15 K. Simulation terminated.")
                self.error = 1
            if self.P0 < 0.8e5 or self.P0 > 1.1e5:
                print("Error: Dead state pressure must be between 0.8e5 and 1.1e5 Pa. Simulation terminated.")
                self.error = 1
        if self.Fluid == 2 and self.End_use == 2:
            if self.Turbine_isentropic_efficiency < 0.8 or self.Turbine_isentropic_efficiency > 1:
                print("Error: Turbine isentropic efficiency must be between 0.8 and 1. Simulation terminated.")
                self.error = 1
            if self.Generator_efficiency < 0.8 or self.Generator_efficiency > 1:
                print("Error: Generator efficiency must be between 0.8 and 1. Simulation terminated.")
                self.error = 1
            if self.Compressor_isentropic_efficiency < 0.8 or self.Compressor_isentropic_efficiency > 1:
                print("Error: Compressor isentropic efficiency must be between 0.8 and 1. Simulation terminated.")
                self.error = 1
            if self.Pre_Cooling_Delta_T < 0 or self.Pre_Cooling_Delta_T > 15:
                print("Error: CO2 temperature decline after turbine and before compressor must be between 0 and 15 degrees C. Simulation terminated.")
                self.error = 1
            if self.Turbine_outlet_pressure < 75 or self.Turbine_outlet_pressure > 200:
                print("Error: CO2 turbine outlet pressure must be between 75 and 200 bar. Simulation terminated.")
                self.error = 1
        return self.error
    
    def initialize(self):
        
        if self.Fluid == 1:
            if self.Configuration == 1:
                self.u_H2O = data(self.filename, "utube", "H2O")
            elif self.Configuration == 2:
                self.u_H2O = data(self.filename, "coaxial", "H2O")
            self.timearray = self.u_H2O.time
            self.FlowRateVector = self.u_H2O.mdot #length of 26
            self.HorizontalLengthVector = self.u_H2O.L2 #length of 20
            self.DepthVector = self.u_H2O.L1 #length of 9
            self.GradientVector = self.u_H2O.grad #length of 5
            self.DiameterVector = self.u_H2O.D #length of 3
            self.TinVector = self.u_H2O.Tinj #length of 3
            self.KrockVector = self.u_H2O.k #length of 3
            self.Fluid_name = 'Water'
        elif self.Fluid == 2:
            if self.Configuration == 1:
                self.u_sCO2 = data(self.filename, "utube", "sCO2")
            elif self.Configuration == 2:
                self.u_sCO2 = data(self.filename, "coaxial", "sCO2")
            self.timearray = self.u_sCO2.time
            self.FlowRateVector = self.u_sCO2.mdot #length of 26
            self.HorizontalLengthVector = self.u_sCO2.L2 #length of 20
            self.DepthVector = self.u_sCO2.L1 #length of 9
            self.GradientVector = self.u_sCO2.grad #length of 5
            self.DiameterVector = self.u_sCO2.D #length of 3
            self.TinVector = self.u_sCO2.Tinj #length of 3
            self.KrockVector = self.u_sCO2.k #length of 3   
            self.Fluid_name = 'CarbonDioxide'            
            
        self.numberofcases = len(self.FlowRateVector)*len(self.HorizontalLengthVector)*len(self.DepthVector)*len(self.GradientVector)*len(self.DiameterVector)*len(self.TinVector)*len(self.KrockVector)
        
        
        self.Time_array = np.linspace(0,self.Lifetime*365*24*3600,1+self.Lifetime*self.Number_of_points_per_year) #[s]
        self.Linear_time_distribution = self.Time_array/365/24/3600
        self.TNOP = (self.Lifetime*self.Number_of_points_per_year+1)      #Total number of points for selected lifetime
        #Find closests lifetime
        closestlifetime = self.timearray.flat[np.abs(self.timearray - self.Lifetime).argmin()]    
        self.indexclosestlifetime = np.where(self.timearray == closestlifetime)[0][0]

        #load property data
        if self.Fluid == 1:
            mat = scipy.io.loadmat('properties_H2O.mat') 
        else:
            mat = scipy.io.loadmat('properties_CO2v2.mat') 
            additional_mat = scipy.io.loadmat('additional_properties_CO2v2.mat')
        self.Pvector = mat['Pvector'][0]
        self.Tvector = mat['Tvector'][0]
        self.density = mat['density']
        self.enthalpy = mat['enthalpy']
        self.entropy = mat['entropy']
        if self.Fluid == 2:
            self.Pvector_ap = additional_mat['Pvector_ap'][0]
            self.hvector_ap = additional_mat['hvector_ap'][0]
            self.svector_ap = additional_mat['svector_ap'][0]
            self.TPh = additional_mat['TPh']
            self.hPs = additional_mat['hPs']
    
        #Define ORC power plant conversion efficiencies
        self.Utilization_efficiency_correlation_temperatures = np.array([100, 200, 385]) #Linear correlation assumed here based on GEOPHIRES ORC correlation between 100 and 200 deg C [deg.C] plus plateaued above 200 deg. C
        self.Utilization_efficiency_correlation_conversion = np.array([0.2, 0.45, 0.45])  #Efficiency of ORC conversion from production exergy to electricity based on GEOPHIRES correlation [-]
        self.Heat_to_power_efficiency_correlation_temperatures = np.array([100, 200, 385]) #Linear correlation based on Chad Augustine's thesis [deg.C] plus plateaued above 200 deg. C
        self.Heat_to_power_efficiency_correlation_conversion = np.array([0.05, 0.14, 0.14]) #Conversion from enthalpy to electricity [-]

        #Calculate dead-state enthalpy and entropy in case of electricity production
        if self.End_use == 2:   
            self.h_0 = interpn((self.Pvector,self.Tvector),self.enthalpy,np.array([self.P0,self.T0]))[0] #dead-state enthalpy [J/kg]
            self.s_0 = interpn((self.Pvector,self.Tvector),self.entropy,np.array([self.P0,self.T0]))[0] #dead-state entropy [J/kg/K]


        #Pre-populate specific heat capacity of air in case of electricity production
        if self.End_use == 2:
            self.Tair_for_cp_array = np.linspace(0,100,num=10)
            #self.cp_air_array = CP.PropsSI('C','P',self.P0,'T',self.Tair_for_cp_array+273.15,'air')
            self.cp_air_array = np.array([1005.65818063, 1005.87727966, 1006.19281999, 1006.60616167, 1007.11890862, 1007.73265999, 1008.44882744, 1009.26850304, 1010.19236691, 1011.2206266])
              
        #Initialize heat/electricity arrays
        self.Instantaneous_production_enthalpy = np.zeros(len(self.Time_array))
        self.Instantaneous_temperature_after_isenthalpic_throttling = np.zeros(len(self.Time_array))
        self.Instantaneous_heat_production = np.zeros(len(self.Time_array))
        self.Annual_heat_production = np.zeros(self.Lifetime)
        self.Annual_pumping_power = np.zeros(self.Lifetime)
        self.Average_fluid_density = np.zeros(len(self.Time_array))
        if self.End_use == 2: #electricity generation
            self.Instantaneous_exergy_production = np.zeros(len(self.Time_array))  #Produced exergy only (independent from injection conditions)
            self.Instantaneous_exergy_extraction = np.zeros(len(self.Time_array))  #Difference between produced exergy and injected exergy
            self.Instantaneous_electricity_production_method_1 = np.zeros(len(self.Time_array)) #based on exergy produced (only for water)
            self.Instantaneous_electricity_production_method_2 = np.zeros(len(self.Time_array)) #based on exergy extracted
            self.Instantaneous_electricity_production_method_3 = np.zeros(len(self.Time_array)) #based on thermal efficiency
            self.Instantaneous_electricity_production_method_4 = np.zeros(len(self.Time_array)) #based on direct turbine expansion (for CO2)
            self.Instantaneous_utilization_efficiency_method_1 = np.zeros(len(self.Time_array)) #conversion from produced exergy to electricity
            self.Instantaneous_utilization_efficiency_method_2 = np.zeros(len(self.Time_array)) #conversion from extracted exergy to electricity
            self.Instantaneous_themal_efficiency = np.zeros(len(self.Time_array)) #conversion from enthalpy to electricity
            self.Annual_electricity_production = np.zeros(self.Lifetime)
        if self.Fluid == 2:
            self.Instantaneous_turbine_power = np.zeros(len(self.Time_array)) #Direct turbine expansion considered for systems using sCO2

        #Initialize error code
        self.error_codes = np.zeros(0)  #if error occurs, code will be assigned to this tag





    def getTandP(self):
        
        if self.Fluid == 1:
            self.Tout, self.Pout = self.u_H2O.interp_outlet_states(self.point)
        elif self.Fluid == 2:
            self.Tout, self.Pout = self.u_sCO2.interp_outlet_states(self.point)

        #Initial time correction (Correct production temperature and pressure at time 0 (the value at time 0 [=initial condition] is not a good representation for the first few months)
        self.Tout[0] = self.Tout[1]
        self.Pout[0] = self.Pout[1]
        
        #Extract Tout and Pout over lifetime
        self.InterpolatedTemperatureArray = self.Tout[0:self.indexclosestlifetime+1]-273.15
        self.InterpolatedPressureArray = self.Pout[0:self.indexclosestlifetime+1]
        
        
        
    def calculateLC(self):
        self.Linear_production_temperature = self.InterpolatedTemperatureArray
        self.Linear_production_pressure = self.InterpolatedPressureArray
        self.AveProductionTemperature = np.average(self.Linear_production_temperature)
        self.AveProductionPressure = np.average(self.Linear_production_pressure)/1e5  #[bar]
        self.Flow_rate = self.Flow_user #Total flow rate [kg/s]
        self.calculatedrillinglength()
        #print(self.Linear_production_temperature)
        if min(self.Linear_production_temperature) > self.T_in:
            self.calculateheatproduction()
            if self.End_use == 2:
                self.calculateelectricityproduction()
            self.calculatecapex()
            self.calculatopex()
            
            
            Discount_vector = 1./np.power(1+self.Discount_rate,np.linspace(0,self.Lifetime-1,self.Lifetime))
            # if self.End_use == 1:   #direct-use heating
            self.LCOH = (self.TotalCAPEX + np.sum(self.OPEX_Plant*Discount_vector))*1e6/np.sum(self.Annual_heat_production/1e3*Discount_vector) #$/MWh
            self.LCOH100 = (self.TotalCAPEX100 + np.sum(self.OPEX_Plant*Discount_vector))*1e6/np.sum(self.Annual_heat_production/1e3*Discount_vector) #$/MWh
            self.LCOH500 = (self.TotalCAPEX500 + np.sum(self.OPEX_Plant*Discount_vector))*1e6/np.sum(self.Annual_heat_production/1e3*Discount_vector) #$/MWh
            self.LCOH1500 = (self.TotalCAPEX1500 + np.sum(self.OPEX_Plant*Discount_vector))*1e6/np.sum(self.Annual_heat_production/1e3*Discount_vector) #$/MWh
            if self.LCOH<0:
                self.LCOH = 9999
                self.error_codes = np.append(self.error_codes,5000)
            # elif self.End_use == 2: #electricity production
            if self.Average_electricity_production == 0:
                self.LCOE = 9999
                self.LCOE100 = float("nan")
                self.LCOE500 = float("nan")
                self.LCOE1500 = float("nan")
                self.error_codes = np.append(self.error_codes,6000)
            else:
                self.LCOE = (self.TotalCAPEX + np.sum(self.OPEX_Plant*Discount_vector))*1e6/np.sum((self.Annual_electricity_production-self.Annual_pumping_power)/1e3*Discount_vector) #$/MWh
                self.LCOE100 = (self.TotalCAPEX100 + np.sum(self.OPEX_Plant*Discount_vector))*1e6/np.sum((self.Annual_electricity_production-self.Annual_pumping_power)/1e3*Discount_vector) #$/MWh
                self.LCOE500 = (self.TotalCAPEX500 + np.sum(self.OPEX_Plant*Discount_vector))*1e6/np.sum((self.Annual_electricity_production-self.Annual_pumping_power)/1e3*Discount_vector) #$/MWh
                self.LCOE1500 = (self.TotalCAPEX1500 + np.sum(self.OPEX_Plant*Discount_vector))*1e6/np.sum((self.Annual_electricity_production-self.Annual_pumping_power)/1e3*Discount_vector) #$/MWh
            if self.LCOE<0:
                self.LCOE = 9999
                self.LCOE100 = float("nan")
                self.LCOE500 = float("nan")
                self.LCOE1500 = float("nan")                    
                self.error_codes = np.append(self.error_codes,7000)
            
        else:  #Production temperature went below injection temperature
            self.LCOE = 9999
            self.LCOH = 9999
            self.error_codes = np.append(self.error_codes,1000)
    
 
    def calculatedrillinglength(self):
        if self.Configuration == 1:
            self.Drilling_length = self.Hor_length_user + 2*self.Depth_user  #Total drilling depth of both wells and lateral in U-loop [m]
        elif self.Configuration == 2:
            self.Drilling_length = self.Hor_length_user + self.Depth_user  #Total drilling depth of well and lateral in co-axial case [m]        
    
    def calculateheatproduction(self):
        #Calculate instantaneous heat production
        self.Average_fluid_density = interpn((self.Pvector,self.Tvector),self.density,np.dstack((0.5*self.P_in + 0.5*self.Linear_production_pressure,0.5*self.T_in + 0.5*self.Linear_production_temperature+273.15))[0])
        self.hprod = interpn((self.Pvector,self.Tvector),self.enthalpy,np.dstack((self.Linear_production_pressure,self.Linear_production_temperature+273.15))[0])
        self.hinj = interpn((self.Pvector,self.Tvector),self.enthalpy,np.array([self.P_in,self.T_in+273.15]))
        self.Instantaneous_heat_production = self.Flow_rate*(self.hprod - self.hinj)/1000 #Heat production based on produced minus injected enthalpy [kW]
        
        #Calculate annual heat production (kWh)
        self.Annual_heat_production = 8760/5*(self.Instantaneous_heat_production[0::4][0:-1]+self.Instantaneous_heat_production[1::4]+self.Instantaneous_heat_production[2::4]+self.Instantaneous_heat_production[3::4]+self.Instantaneous_heat_production[4::4])
      
        #Calculate average heat production
        self.AveAnnualHeatProduction = np.average(self.Annual_heat_production) #kWh
        self.AveInstHeatProduction = np.average(self.Instantaneous_heat_production) #kWth
        
        #Calculate average heat production and first year heat production
        self.Average_heat_production = np.average(self.Instantaneous_heat_production) #[kW]
        #Average_production_temperature = np.average(Linear_production_temperature) #[deg.C]
        self.FirstYearHeatProduction = self.Annual_heat_production[0] #kWh
        
        self.calculatepumpingpower()
        
        
    def calculateelectricityproduction(self):
        
        #Calculate instantaneous exergy production, exergy extraction, and electricity generation (MW) and annual electricity generation [kWh]
        self.h_prod = self.hprod #produced enthalpy [J/kg]
        self.h_inj = self.hinj #injected enthalpy [J/kg]
        self.s_prod = interpn((self.Pvector,self.Tvector),self.entropy,np.dstack((self.Linear_production_pressure,self.Linear_production_temperature+273.15))[0]) #produced entropy [J/kg/K]
        self.s_inj = interpn((self.Pvector,self.Tvector),self.entropy,np.array([self.P_in,self.T_in+273.15])) #injected entropy [J/kg/K]
            
        self.Instantaneous_exergy_production = (self.Flow_rate*(self.h_prod-self.h_0 - self.T0*(self.s_prod-self.s_0)))/1000 #[kW]
        self.Instantaneous_exergy_extraction = (self.Flow_rate*(self.h_prod-self.h_inj - self.T0*(self.s_prod-self.s_inj)))/1000 #[kW]     
            
        self.AverageInstNetExergyProduction = np.average(self.Instantaneous_exergy_production) #[kW]
        self.AverageInstNetExergyExtraction = np.average(self.Instantaneous_exergy_extraction) #[kW]
            
        if self.Fluid == 1:
            
            if self.T_in >= 50 and min(self.Linear_production_temperature) >= 100 and max(self.Linear_production_temperature) <= 385:
                self.Instantaneous_utilization_efficiency_method_1 = np.interp(self.Linear_production_temperature,self.Utilization_efficiency_correlation_temperatures,self.Utilization_efficiency_correlation_conversion,left = 0) #Utilization efficiency based on conversion of produced exergy to electricity
                self.Instantaneous_electricity_production_method_1 = self.Instantaneous_exergy_production*self.Instantaneous_utilization_efficiency_method_1 #[kW]
                self.Instantaneous_themal_efficiency = np.interp(self.Linear_production_temperature,self.Heat_to_power_efficiency_correlation_temperatures,self.Heat_to_power_efficiency_correlation_conversion,left = 0) #Utilization efficiency based on conversion of produced exergy to electricity
                self.Instantaneous_electricity_production_method_3 = self.Instantaneous_heat_production*self.Instantaneous_themal_efficiency #[kW]
                
            else: #Water injection temperature and/or production tempeature fall outside the range used in the correlations
                self.error_codes = np.append(self.error_codes,2000)
                self.Instantaneous_utilization_efficiency_method_1 = np.zeros(len(self.Time_array))
                self.Instantaneous_electricity_production_method_1 = np.zeros(len(self.Time_array))
                self.Instantaneous_themal_efficiency = np.zeros(len(self.Time_array))
                self.Instantaneous_electricity_production_method_3 = np.zeros(len(self.Time_array))
                    
            #based on method 1 for now (could be 50-50)
            self.Annual_electricity_production = 8760/5*(self.Instantaneous_electricity_production_method_1[0::4][0:-1]+self.Instantaneous_electricity_production_method_1[1::4]+self.Instantaneous_electricity_production_method_1[2::4]+self.Instantaneous_electricity_production_method_1[3::4]+self.Instantaneous_electricity_production_method_1[4::4])
            self.Inst_electricity_production = self.Instantaneous_electricity_production_method_1 #[kW]
            self.AveInstElectricityProduction = np.average(self.Instantaneous_electricity_production_method_1) #[kW]
            
    
        elif self.Fluid == 2:
            T_prod = self.Linear_production_temperature #Production temperature [deg.C]
            P_prod = self.Linear_production_pressure    #Production pressure [Pa]
    
            h_turbine_out_ideal = interpn((self.Pvector_ap,self.svector_ap),self.hPs,np.dstack((np.ones(self.TNOP)*self.Turbine_outlet_pressure*1e5,self.s_prod))[0]) 
            self.Instantaneous_turbine_power = self.Flow_rate*(self.h_prod-h_turbine_out_ideal)*self.Turbine_isentropic_efficiency/1000 #Turbine output [kW]
            h_turbine_out_actual = self.h_prod-self.Instantaneous_turbine_power/self.Flow_rate*1000 #Actual fluid enthalpy at turbine outlet [J/kg]
            self.T_turbine_out_actual = interpn((self.Pvector_ap,self.hvector_ap),self.TPh,np.dstack((np.ones(self.TNOP)*self.Turbine_outlet_pressure*1e5,h_turbine_out_actual))[0])-273.15 
                
            if min(self.T_turbine_out_actual) > 37 and self.T_in > 32:
                
                self.Pre_cooling_temperature = min(self.T_turbine_out_actual) - self.Pre_Cooling_Delta_T
                Post_cooling = 2000
                valuefound = 0
                lastrun = 0
                #print('y')
                while valuefound==0:
                    #print('b')
                    Pre_compressor_h = interpn((self.Pvector,self.Tvector),self.enthalpy,np.array([self.Turbine_outlet_pressure*1e5,self.Pre_cooling_temperature+273.15])) 
                    
                    Pre_cooling = self.Flow_rate*(h_turbine_out_actual - Pre_compressor_h)/1e3 #Pre-compressor cooling [kWth]
                    Pre_compressor_s = interpn((self.Pvector,self.Tvector),self.entropy,np.array([self.Turbine_outlet_pressure*1e5,self.Pre_cooling_temperature+273.15])) 
                    
                    
                    Post_compressor_h_ideal = interpn((self.Pvector_ap,self.svector_ap),self.hPs,np.array([self.P_in,Pre_compressor_s[0]])) 
                    Post_compressor_h_actual = Pre_compressor_h + (Post_compressor_h_ideal-Pre_compressor_h)/self.Compressor_isentropic_efficiency #Actual fluid enthalpy at compressor outlet [J/kg]
                    self.Post_compressor_T_actual = interpn((self.Pvector_ap,self.hvector_ap),self.TPh,np.array([self.P_in,Post_compressor_h_actual[0]])) - 273.15
                    Compressor_Work = self.Flow_rate*(Post_compressor_h_actual - Pre_compressor_h)/1e3 #[kWe]
                    Post_cooling = self.Flow_rate*(Post_compressor_h_actual - self.h_inj)/1e3 #Fluid cooling after compression [kWth]
                    #print(str(Post_cooling))
                    #print(str(self.Pre_cooling_temperature))
                    
                    if lastrun == 0:
                        if self.Pre_cooling_temperature < 32:
                            lastrun = 1                        
                            self.Pre_cooling_temperature = self.Pre_cooling_temperature + 0.5
                        elif Post_cooling < 0:
                            self.Pre_cooling_temperature = self.Pre_cooling_temperature + 0.5
                            lastrun = 1
                        elif Post_cooling >0:
                            self.Pre_cooling_temperature = self.Pre_cooling_temperature - 0.5
                    
                    elif lastrun == 1:
                        valuefound = 1
                        print(self.Pre_cooling_temperature - min(self.T_turbine_out_actual))

                    #print(min(self.T_turbine_out_actual))
                    #print(self.Pre_cooling_temperature)
                    

                    
                if Post_cooling<0:
                    ResistiveHeating = -Post_cooling
                    Post_cooling = 0
                else:
                    ResistiveHeating = 0
                   
                Total_cooling = Pre_cooling + Post_cooling #Total CO2 cooling requirements [kWth]
                
                T_air_in_pre_cooler = self.T0-273.15
                T_air_out_pre_cooler = (self.T_turbine_out_actual+self.Pre_cooling_temperature)/2 #Air outlet temperature in pre-cooler [deg.C]
                cp_air = np.interp(0.5*T_air_in_pre_cooler+0.5*T_air_out_pre_cooler,self.Tair_for_cp_array,self.cp_air_array) #Air specific heat capacity in pre-cooler [J/kg/K]
                m_air_pre_cooler = Pre_cooling*1000/(cp_air*(T_air_out_pre_cooler - T_air_in_pre_cooler)) #Air flow rate in pre-cooler [kg/s]
               
                T_air_in_post_cooler = self.T0-273.15
                T_air_out_post_cooler = (self.Post_compressor_T_actual+self.T_in)/2 #Air outlet temperature in post-cooler [deg.C]
                cp_air = np.interp(0.5*T_air_in_post_cooler+0.5*T_air_out_post_cooler,self.Tair_for_cp_array,self.cp_air_array) #Air specific heat capacity in post-cooler [J/kg/K]
                    
                m_air_post_cooler = Post_cooling*1000/(cp_air*(T_air_out_post_cooler - T_air_in_post_cooler)) #Air flow rate in post-cooler [kg/s]
                
                Air_cooling_power = (m_air_pre_cooler+m_air_post_cooler)*0.25  #Electricity for air-cooling, assuming 0.25 kWe per kg/s [kWe] 
                #print(self.Instantaneous_turbine_power)
                #print(Compressor_Work)
                self.Instantaneous_electricity_production_method_4 = self.Instantaneous_turbine_power*self.Generator_efficiency-Compressor_Work-Air_cooling_power-ResistiveHeating # Net electricity using CO2 direct turbine expansion cycle [kWe]
                self.Inst_electricity_production = self.Instantaneous_electricity_production_method_4 #[kW]
                self.Annual_electricity_production = 8760/5*(self.Instantaneous_electricity_production_method_4[0::4][0:-1]+self.Instantaneous_electricity_production_method_4[1::4]+self.Instantaneous_electricity_production_method_4[2::4]+self.Instantaneous_electricity_production_method_4[3::4]+self.Instantaneous_electricity_production_method_4[4::4])
                self.AveInstElectricityProduction = np.average(self.Instantaneous_electricity_production_method_4) #[kW]
                #check if negative
                if min(self.Instantaneous_electricity_production_method_4)<0:
                    self.error_codes = np.append(self.error_codes,5500) #Calculated electricity generation is negative
                    
                    self.Annual_electricity_production = np.zeros(self.Lifetime)
                    self.Inst_electricity_production = np.zeros(self.TNOP)
                    self.AveInstElectricityProduction = 0
            else: #turbine outlet or reinjection temperature too low
                if (self.T_in <= 32):
                    self.error_codes = np.append(self.error_codes,3000)
                    
                if (min(self.T_turbine_out_actual)<=37):
                    self.error_codes = np.append(self.error_codes,4000)
                    
                self.Annual_electricity_production = np.zeros(self.Lifetime)
                self.Inst_electricity_production = np.zeros(self.TNOP)
                self.AveInstElectricityProduction = 0
        
        self.calculatepumpingpower()
        
        self.Average_electricity_production = np.average(self.Annual_electricity_production)/8760 #[kW]
        self.AveAnnualElectricityProduction = np.average(self.Annual_electricity_production) #[kWh]
        self.AveInstNetElectricityProduction = self.AveInstElectricityProduction - np.average(self.PumpingPower) #[kW]
        self.AveAnnualNetElectricityProduction = self.AveAnnualElectricityProduction - np.average(self.Annual_pumping_power) #kWh
        self.FirstYearElectricityProduction = self.Annual_electricity_production[0] #kWh
        self.Inst_Net_Electricity_production = self.Inst_electricity_production-self.PumpingPower #[kW]
        
        if self.AveInstNetElectricityProduction < 0:
            self.AveInstNetElectricityProduction = 0

    def calculatepumpingpower(self):
        #Calculate pumping power
        self.PumpingPower = (self.P_in-self.Linear_production_pressure)*self.Flow_rate/self.Average_fluid_density/self.Pump_efficiency/1e3 #Pumping power [kW]
        self.PumpingPower[self.PumpingPower<0] = 0 #Set negative values to zero (if the production pressure is above the injection pressure, we throttle the fluid)
        self.Annual_pumping_power = 8760/5*(self.PumpingPower[0::4][0:-1]+self.PumpingPower[1::4]+self.PumpingPower[2::4]+self.PumpingPower[3::4]+self.PumpingPower[4::4]) #kWh
            
    def calculatecapex(self):
        self.CAPEX_Drilling = self.Drilling_length*self.Drilling_cost_per_m/1e6 #Drilling capital cost [M$]
        if self.End_use == 1:   #direct-use heating
            self.CAPEX_Surface_Plant = np.max(self.Instantaneous_heat_production)*self.Direct_use_heat_cost_per_kWth/1e6 #[M$]
        elif self.End_use == 2: #electricity production
            if self.Fluid == 1:
                self.CAPEX_Surface_Plant = np.max(self.Instantaneous_electricity_production_method_1)*self.Power_plant_cost_per_kWe/1e6 #[M$]
            elif self.Fluid == 2:
                self.CAPEX_Surface_Plant = np.max(self.Instantaneous_electricity_production_method_4)*self.Power_plant_cost_per_kWe/1e6 #[M$]
        
        self.TotalCAPEX = self.CAPEX_Drilling + self.CAPEX_Surface_Plant           #Total system capital cost (only includes drilling and surface plant cost) [M$]
        self.TotalCAPEX100 = self.Drilling_length*100/1e6 + self.CAPEX_Surface_Plant           #Total system capital cost (only includes drilling and surface plant cost) [M$]
        self.TotalCAPEX500 = self.Drilling_length*500/1e6 + self.CAPEX_Surface_Plant           #Total system capital cost (only includes drilling and surface plant cost) [M$]
        self.TotalCAPEX1500 = self.Drilling_length*1500/1e6 + self.CAPEX_Surface_Plant           #Total system capital cost (only includes drilling and surface plant cost) [M$]
        
    def calculatopex(self):
        #Calculate OPEX
        if self.End_use == 1: #direct-use heating
            self.OPEX_Plant = self.O_and_M_cost_plant*self.CAPEX_Surface_Plant + self.Annual_pumping_power*self.Electricity_rate/1e6  #Annual plant O&M cost [M$/year]
        elif self.End_use == 2: #electricity production
            self.OPEX_Plant = self.O_and_M_cost_plant*self.CAPEX_Surface_Plant  #Annual plant O&M cost [M$/year]
        self.AverageOPEX_Plant = np.average(self.OPEX_Plant)
    
    def printresults(self):
        ### Print results to screen
        print('##################')
        print('Simulation Results')
        print('##################')
        print('Number of cases in database = ' + str(self.numberofcases))
        print(" ")
        print('### Configuration ###')
        if self.End_use == 1:
            print('End-Use = Direct-Use')
        elif self.End_use == 2:    
            print('End-Use = Electricity')
        if self.Fluid == 1:
            print('Fluid = water')
        elif self.Fluid == 2:
            print('Fluid = sCO2')     
        if self.Configuration == 1:
            print('Design = U-loop')
        elif self.Configuration == 2:
            print('Design = Co-axial')
            
        #Print conditions
        print("Flow rate = " + "{0:.1f}".format((self.Flow_user)) +" kg/s")
        print("Lateral Length = " + str(round(self.Hor_length_user)) +" m")
        print("Vertical Depth = " + str(round(self.Depth_user)) +" m")
        print("Geothermal Gradient = " + "{0:.1f}".format((self.Gradient_user*1000)) +" deg.C/km")
        print("Wellbore Diameter = " + "{0:.4f}".format((self.Diameter_user)) +" m")
        print("Injection Temperature = " + "{0:.1f}".format((self.Tin_user-273.15)) +" deg.C")
        print("Thermal Conductivity = " + "{0:.2f}".format((self.krock_user)) +" W/m/K")
        
        #check if error occured
        if len(self.error_codes)>0:
            print(" ")
            if np.in1d(1000,self.error_codes): #plot the temperature and pressure for these
                print("Error: production temperature drops below injection temperature. Simulation terminated.\n")
                
            if np.in1d(2000,self.error_codes): #plot the temperature and pressure for these
                print("Error: Water injection temperature and/or production tempeature fall outside the range of the ORC correlations. These correlations require injection temperature larger than 50 deg.C and production temperature in the range of 100 to 385 deg.C. Electricity production set to 0. \n")
                
            if np.in1d(3000,self.error_codes): #CO2 injection temperature cannot be below 32 degrees C (CO2 must be supercritical)
                print("Error: Too low CO2 reinjection temperature. CO2 must remain supercritical.\n")
                
            if np.in1d(4000,self.error_codes): #Turbine outlet CO2 temperature dropped below 37 degrees C
                print("Error: Too low CO2 turbine outlet temperature. Turbine outlet temperature must be above 37 degrees C.\n")    
                
            if np.in1d(5000,self.error_codes): #Calculated LCOH was negative, set to $9999/MWh
                print("Error: Calculated LCOH was negative and has been reset to $9,999/MWh.\n")     
                
            if np.in1d(5500,self.error_codes): #Negative electricity calculated with CO2 cycle
                print("Error: Calculated net electricity generation is negative and reset to 0.\n")
                
            if np.in1d(6000,self.error_codes): #Zero electricity production. LCOE set to $9999/MWh
                print("Error: Calculated net electricity production was 0. LCOE reset to $9,999/MWh.\n")   
        
            if np.in1d(7000,self.error_codes): #Calculated LCOE was negative, set to $9999/MWh
                print("Error: Calculated LCOE was negative and has been reset to $9,999/MWh.\n")  
                
        
        #Print results for heating
        if self.End_use==1:
            print(" ")
            print('### Reservoir Simulation Results ###')
            print("Average Production Temperature = " + "{0:.1f}".format((self.AveProductionTemperature)) + " deg.C")
            print("Average Production Pressure = " + "{0:.1f}".format((self.AveProductionPressure)) + " bar")
            if np.in1d(1000,self.error_codes) == False:
                print("Average Heat Production = " + "{0:.1f}".format((self.AveInstHeatProduction)) +" kWth" )    
                print("First Year Heat Production = " + "{0:.1f}".format((self.FirstYearHeatProduction/1e3)) + " MWh")    
                print(" ")
                print('### Cost Results ###')
                print("Total CAPEX = " + "{0:.1f}".format((self.TotalCAPEX)) + " M$")
                print("Drilling Cost = " + "{0:.1f}".format((self.CAPEX_Drilling)) + " M$")
                print("Surface Plant Cost = " + "{0:.1f}".format((self.CAPEX_Surface_Plant)) + " M$")
                print("OPEX = " + "{0:.1f}".format((self.AverageOPEX_Plant*1000)) + " k$/year")
                print("LCOH = " + "{0:.1f}".format((self.LCOH)) +" $/MWh")
            
        if self.End_use == 2:
            print(" ")
            print('### Reservoir Simulation Results ###')
            print("Average Production Temperature = " + "{0:.1f}".format((self.AveProductionTemperature)) + " deg.C")
            print("Average Production Pressure = " + "{0:.1f}".format((self.AveProductionPressure)) + " bar")
            if np.in1d(1000,self.error_codes) == False:
                print("Average Heat Production = " + "{0:.1f}".format((self.AveInstHeatProduction)) +" kWth" )    
                print("Average Net Electricity Production = " + "{0:.1f}".format((self.AveInstNetElectricityProduction)) +" kWe" )    
                print("First Year Heat Production = " + "{0:.1f}".format((self.FirstYearHeatProduction/1e3)) + " MWh")    
                print("First Year Electricity Production = " + "{0:.1f}".format((self.FirstYearElectricityProduction/1e3)) + " MWh")
                print(" ")
                print('### Cost Results ###')
                print("Total CAPEX = " + "{0:.1f}".format((self.TotalCAPEX)) + " M$")
                print("Drilling Cost = " + "{0:.1f}".format((self.CAPEX_Drilling)) + " M$")
                print("Surface Plant Cost = " + "{0:.1f}".format((self.CAPEX_Surface_Plant)) + " M$")
                print("OPEX = " + "{0:.1f}".format((self.AverageOPEX_Plant*1000)) + " k$/year")
                print("LCOE = " + "{0:.1f}".format((self.LCOE)) +" $/MWh")
                
                
    def plotresults(self):
        #Plot injection and production temperature
        plt.figure()
        plt.title("Injection and Production Temperature") 
        plt.xlabel("Time [year]") 
        plt.ylabel("Temperature [$^\circ$C]") 
        plt.plot([0, self.Lifetime],[self.T_in, self.T_in],'b-')
        plt.plot(self.Linear_time_distribution,self.Linear_production_temperature,'r-')
        plt.legend(['Injection Temperature', 'Production Temperature'])  
        plt.axis([0, self.Lifetime,self.T_in-10,max(self.Linear_production_temperature)+10])          
        plt.xticks(np.arange(0,self.Lifetime+1,2),np.arange(0,self.Lifetime+1,2))
        plt.show()

        #Plot injection and production pressure
        plt.figure()
        plt.title("Injection and Production Pressure") 
        plt.xlabel("Time [year]") 
        plt.ylabel("Pressure [bar]") 
        plt.plot([0, self.Lifetime],[self.P_in/1e5, self.P_in/1e5],'b-')
        plt.plot(self.Linear_time_distribution,self.Linear_production_pressure/1e5,'r-')
        plt.legend(['Injection Pressure', 'Production Pressure'])  
        maxpressure = np.max([self.P_in/1e5,np.max(self.Linear_production_pressure)/1e5])
        minpressure = np.min([self.P_in/1e5,np.min(self.Linear_production_pressure)/1e5])
        plt.axis([0,self.Lifetime,minpressure-10,maxpressure+10])
        plt.xticks(np.arange(0,self.Lifetime+1,2),np.arange(0,self.Lifetime+1,2))
        plt.show()

        #Plot heat production
        plt.figure()
        plt.title("Heat Production") 
        plt.xlabel("Time [year]") 
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)
        plt.ylabel("Heat [$MW_{t}$]") 
        plt.plot(self.Linear_time_distribution,self.Instantaneous_heat_production/1e3,'r-')
        plt.axis([0, self.Lifetime,0,max(self.Instantaneous_heat_production/1e3)+1])    
        plt.xticks(np.arange(0,self.Lifetime+1,2),np.arange(0,self.Lifetime+1,2))
        plt.show()
        
        plt.figure()
        plt.bar(np.arange(1,self.Lifetime+1),self.Annual_heat_production/1e6)
        plt.xlabel("Time [year]")
        plt.ylabel("Heat Production [GWh]")
        plt.title("Annual Heat Production")
        plt.xticks(np.arange(0,self.Lifetime+1,2),np.arange(0,self.Lifetime+1,2))
        plt.show()
        
        #plot electricity over time if end-use is 2
        if self.End_use == 2:
            plt.figure()
            plt.title("Electricity Production") 
            plt.xlabel("Time [year]") 
            params = {'mathtext.default': 'regular' }          
            plt.rcParams.update(params)
            plt.ylabel("Electricity [$MW_{e}$]") 
            plt.plot(self.Linear_time_distribution,self.Inst_Net_Electricity_production/1e3,'r-')
            plt.axis([0, self.Lifetime,0,max(self.Inst_Net_Electricity_production/1e3)+1])    
            plt.xticks(np.arange(0,self.Lifetime+1,2),np.arange(0,self.Lifetime+1,2))
            plt.show()
            
            plt.figure()
            plt.bar(np.arange(1,self.Lifetime+1),self.Annual_electricity_production/1e6)
            plt.xlabel("Time [year]")
            plt.ylabel("Electricity Production [GWh]")
            plt.title("Annual Electricity Production")
            plt.xticks(np.arange(0,self.Lifetime+1,2),np.arange(0,self.Lifetime+1,2))
            plt.show()
        
            #create Ts-diagram in case of CO2 power production
            if self.Fluid == 2 and np.in1d(3000,self.error_codes) == False and np.in1d(4000,self.error_codes) == False:
                #CO2 isobaric lines for Ts diagram (do not change data below)
                pvectorforTS = np.array([40,50,60,70,80,90,100,120,150,190,240,300]) #bar
                svectorforTS = np.arange(1000,2000,20) #J/kg/K                              
                tmatrix = np.array([[273.58779987, 275.73964358, 277.82952322, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 278.44972407, 279.16739097, 281.98680218, 285.11251959, 288.53183612, 292.23286854, 296.20421415, 300.43497368, 304.91492276, 309.63463723], \
                [274.42077418, 276.63242586, 278.79031756, 280.88470759, 282.90420093, 284.8353605,  286.66228321, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.43392381, 287.96651004, 290.06555059, 292.48128297, 295.20562474, 298.23009373, 301.54561859, 305.14278777, 309.01219991, 313.14471796, 317.53155667, 322.16425682, 327.03466505], \
                [275.23288781, 277.50029097, 279.72101728, 281.88672793, 283.98779516, 286.01303489, 287.94946975, 289.78218424, 291.4942541 , 293.06653014, 294.47679734, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.12790099, 295.9354915,  297.35537749, 299.07212335, 301.08795261, 303.40261519, 306.01350043, 308.91613741, 312.10478208, 315.57290651, 319.31351482, 323.31931035, 327.58280441, 332.09645076, 336.85281943, 341.84475813], \
                [276.02607997, 278.34583338, 280.62513674, 282.85677401, 285.03249573, 287.14282112, 289.17689654, 291.12248905, 292.96614008, 294.69336153, 296.28858454, 297.73458483, 299.011625  , 300.09758466, 300.97073532, 301.61423442, 301.8325153 , 301.8325153 , 301.8325153 , 301.8325153, 301.8325153,  301.8325153 , 301.8325153 , 301.8325153 , 301.8325153, 301.8325153,  301.8325153 , 301.8325153 , 301.8325153 , 301.8325153, 302.09542982, 302.58879462, 303.2793647 , 304.18893716, 305.33653776, 306.73773627, 308.40431208, 310.34433077, 312.56253551, 315.06089701, 317.83917492, 320.89539075, 324.2261797 , 327.82705562, 331.69265609, 335.81701249, 340.19383373, 344.81675258, 349.67949196, 354.77594888], \
                [276.80189842, 279.17107055, 281.50533372, 283.79838944, 286.04309117, 288.23128859, 290.3537384 , 292.40016137, 294.35947993, 296.22015087, 297.97036237, 299.59784329, 301.08934154, 302.43049789, 303.60731254, 304.60958484, 305.43472959, 306.09016372, 306.59420747, 306.97336927, 307.25532028, 307.47069845, 307.65403034, 307.82936031, 308.0076719, 308.20853737, 308.45497452, 308.76827506, 309.16916382, 309.67981016, 310.32386748, 311.12522831, 312.10669659, 313.28900162, 314.69018476, 316.32529927, 318.20635339, 320.34242373, 322.73986477, 325.4025539, 328.33214361, 331.52832983, 334.98916153, 338.71139868, 342.69088804, 346.92290167, 351.40239399, 356.12416833, 361.08297566, 366.27357671], \
                [277.56162433, 279.97764204, 282.36372714, 284.71434274, 287.02324203, 289.28333988, 291.48665471, 293.6243954 , 295.68722534, 297.66562966, 299.55018853, 301.33153232, 302.9999648 , 304.54520748, 305.95710135, 307.22771897, 308.35404538, 309.33970039, 310.19516516, 310.935894, 311.57873857, 312.143772  , 312.65796099, 313.14933519, 313.64039782, 314.1467086 , 314.6901326 , 315.29582886, 315.98700684, 316.78469962, 317.70901078, 318.78001191, 320.01762104, 321.44086311, 323.06709059, 324.91143497, 326.98650416, 329.30226115, 331.86603447, 334.68264398, 337.75463976, 341.08263885, 344.66571751, 348.50179783, 352.58797567, 356.92076582, 361.49627286, 366.31031192, 371.3585025 , 376.63634824], \
                [278.30634695, 280.76692289, 283.20206879, 285.60688807, 287.97587967, 290.30282905, 292.58077049, 294.80208911, 296.95878792, 299.04284976, 301.04652031, 302.96231809, 304.78273895, 306.49997116, 308.10624416, 309.59524733, 310.96417741, 312.21517992, 313.35532245, 314.39511557, 315.34669749, 316.22485477, 317.04972335, 317.84500607, 318.63352456, 319.43547086, 320.26866588, 321.15491544, 322.11728356, 323.17735523, 324.3552376 , 325.67016149, 327.14083776, 328.78527246, 330.6202218, 332.6605961 , 334.91901704, 337.40559839, 340.12794395, 343.09132309, 346.29896643, 349.7524136 , 353.45185087, 357.39639806, 361.58433378, 366.01326948, 370.68028972, 375.58207254, 380.71499591, 386.07523169], \
                [279.75444778, 282.29817944, 284.82433455, 287.32902789, 289.80791774, 292.25612472, 294.66822243, 297.0383542 , 299.36048536, 301.62872141, 303.83755132, 305.98187696, 308.05680937, 310.05743003, 311.97889683, 313.81720263, 315.57044276, 317.23986013, 318.82984388, 320.3468679, 321.79853724, 323.19410279, 324.5458906 , 325.8693588 , 327.18153051, 328.49985457, 329.84235722, 331.22807789, 332.67693979, 334.20933551, 335.84483513, 337.60187632, 339.49779226, 341.54879485, 343.76981952, 346.17429027, 348.77391483, 351.57858366, 354.59638447, 357.83370121, 361.29535632, 364.98476646, 368.9040978 , 373.05441794, 377.43584441, 382.04768885, 386.88859397, 391.9566601 , 397.24955922, 402.76463545], \
                [281.83443065, 284.49054627, 287.13869075, 289.77604486, 292.3994679, 295.00544884, 297.59012963, 300.14943366, 302.67928723, 305.17586241, 307.63573556, 310.05588235, 312.43352359, 314.76595749, 317.05058504, 319.28527574, 321.46901173, 323.60249084, 325.68827882, 327.73038804, 329.73373947, 331.70423537, 333.64955593, 335.5798509 , 337.50755558, 339.44657727, 341.4116138 , 343.41789639, 345.48108684, 347.61704308, 349.84146682, 352.169554  , 354.61583061, 357.19410582, 359.91740391, 362.79786554, 365.84662864, 369.07371357, 372.48793503, 376.09685442, 379.90677609, 383.92278397, 388.14881086, 392.58773063, 397.24146404, 402.11109043, 407.19695909, 412.49879649, 418.01580673, 423.74676402], \
                [284.46010614, 287.24808249, 290.03809343, 292.82824725, 295.61643858, 298.40033711, 301.17744368, 303.94522119, 306.70126758, 309.44346417, 312.1700312 , 314.87946451, 317.57039544, 320.24147407, 322.89138567, 325.51905575, 328.12400395, 330.70672155, 333.2689251 , 335.81360214, 338.34489257, 340.86796442, 343.38903245, 345.91549948, 348.45602389, 351.02034174, 353.61888872, 356.26243929, 358.96193526, 361.72849507, 364.5734711 , 367.50843626, 370.54506747, 373.69496089, 376.9694292, 380.37931828, 383.93485464, 387.64552617, 391.51999386, 395.56603159, 399.79049255, 404.19930077, 408.79746559, 413.58911618, 418.57755178, 423.76530373, 429.15420499, 434.74546357, 440.53973699, 446.53720545], \
                [287.54370173, 290.47477034, 293.41743992, 296.37055692, 299.33284669, 302.30293816, 305.27943869, 308.26104366, 311.24664098, 314.23536353, 317.22656351, 320.21972203, 323.21434555, 326.20991425, 329.20592964, 332.20206785, 335.19840422, 338.19565227, 341.19536186, 344.20003798, 347.21316276, 350.23912622, 353.28309478, 356.35086517, 359.44874828, 362.58349711, 365.76225958, 368.99253217, 372.28211442, 375.63908334, 379.07179464, 382.58888913, 386.19926861, 389.9120183 , 393.7362794, 397.68109447, 401.75525243, 405.96715298, 410.32469917, 414.8352191, 419.50541343, 424.34132465, 429.34832416, 434.5311145 , 439.89374434, 445.43963403, 451.17161001, 457.09194579, 463.2024078 , 469.50430417], \
                [291.00408259, 294.08333456, 297.18293362, 300.3023086 , 303.44085231, 306.59796523, 309.77312115, 312.96593133, 316.17617866, 319.40380478, 322.64885629, 325.91141812, 329.19156995, 332.48939224, 335.80502725, 339.13878159, 342.4912472 , 345.86342063, 349.25680717, 352.67350208, 356.11624126, 359.58841294, 363.09402366, 366.63762115, 370.22418891, 373.85903781, 377.54771937, 381.29597333, 385.10970762, 388.99499966, 392.95810832, 397.00548825, 401.14379812, 405.37989264, 409.72078858, 414.17360117, 418.74545536, 423.4433829 , 428.2742184 , 433.24450557, 438.36042029, 443.62771334, 449.05167197, 454.63709839, 460.38830223,466.30910437, 472.40284992, 478.67242829, 485.12029884, 491.74852062]])
                
                
                #dome data for Ts diagram
                sdome = np.array([1001.37898699, 1046.06882698, 1088.21416035, 1128.86481841, 1169.09596369, 1210.24526707, 1254.66210399, 1309.32653659, 1323.58218729, 1340.74189756, 1364.61655928, 1400.86587985, 1468.75018408, 1516.27446567, 1546.66849166, 1567.63482054, 1584.41961907, 1643.56442582, 1686.22125974, 1722.06785103, 1754.37017763, 1784.81415615, 1814.51645442, 1844.39848637])
                Tdome = np.array([273.31081538, 278.44972407, 283.13044128, 287.43392381, 291.41872472, 295.12790099, 298.59249754, 301.8325153,  302.45422394, 303.06687319, 303.6699029,  304.08533585, 304.08533585, 303.6699029,  303.06687319, 302.45422394, 301.8325153,  298.59249754, 295.12790099, 291.41872472, 287.43392381, 283.13044128, 278.44972407, 273.31081538])
                
                #Ts at turbine inlet
                turbinletT = self.Linear_production_temperature[-1]
                turbinletS = self.s_prod[-1]
            
                #Ts at turbine outlet
                turbineoutletT = self.T_turbine_out_actual[-1]
                turbineoutletS = interpn((self.Pvector,self.Tvector),self.entropy,np.array([self.Turbine_outlet_pressure*1e5,turbineoutletT+273.15]))[0]
            
                #Ts at compressor inlet
                compressorinletT = self.Pre_cooling_temperature
                compressorinletS = interpn((self.Pvector,self.Tvector),self.entropy,np.array([self.Turbine_outlet_pressure*1e5,compressorinletT+273.15]))[0]
            
                #Ts at compressor outlet
                compressoroutletT = self.Post_compressor_T_actual[0]
                compressoroutletS = interpn((self.Pvector,self.Tvector),self.entropy,np.array([self.P_in,compressoroutletT+273.15]))[0]
            
                #Ts at re-injection
                reinjectionT = self.T_in
                reinjectionS = interpn((self.Pvector,self.Tvector),self.entropy,np.array([self.P_in,reinjectionT+273.15]))[0]
            
                cycleTarray = np.array([turbinletT,turbineoutletT,compressorinletT,compressoroutletT,reinjectionT,turbinletT])
                cycleSarray = np.array([turbinletS,turbineoutletS,compressorinletS,compressoroutletS,reinjectionS,turbinletS])
            
                #
                plt.figure()
                plt.title("CO2 TS-diagram") 
                plt.xlabel("Entropy [J/kg/K]") 
                plt.ylabel("Temperature [K]") 
                for i in range (len(pvectorforTS)):
                    plt.plot(svectorforTS,tmatrix[i,:],'b-')
                    plt.plot(sdome,Tdome,'r-')    
                    plt.plot(cycleSarray,cycleTarray+273.15,'k-+')
                #plt.axis([0, 200,0,50])          
                plt.show()
                
    def lineplot(self,parameterrange,parameterindex):
        print(" ")
        print('### Line Plot Calculation ###')
        
        #create arrays to store result
        lcoerange = np.zeros(len(parameterrange))
        avekwthrange = np.zeros(len(parameterrange))
        avekwerange = np.zeros(len(parameterrange))
        
        # 0 = Flow rate; 1 =  Horizontal length; 2 = Depth; 3 = Gradient; 4 = Diameter; 5 = Injection temperature; 6 = Rock thermal conductivity
        #store originals
        flow_original = self.Flow_user
        hor_length_original = self.Hor_length_user
        depth_original = self.Depth_user
        gradient_original = self.Gradient_user
        diameter_original = self.Diameter_user
        Tin_original = self.Tin_user
        krock_original = self.krock_user
        
        #verify input
        if parameterindex == 0:
            if min(parameterrange)<5 or max(parameterrange) > 100:
                print('Error: Flow rate outise allowable range [5 to 100 kg/s]')
                sys.exit()
        elif parameterindex == 1:
            if min(parameterrange)<1000 or max(parameterrange) > 20000:
                print('Error: Horizontal length outise allowable range [1,000 to 20,000 m]')
                sys.exit()
        elif parameterindex == 2:
            if min(parameterrange)<1000 or max(parameterrange) > 5000:
                print('Error: Depth outise allowable range [1,000 to 5,000 m]')
                sys.exit()
        elif parameterindex == 3:
            if min(parameterrange)<0.03 or max(parameterrange) > 0.07:
                print('Error: Gradient outside allowable range [0.03 to 0.07 deg.C/m]')
                sys.exit()
        elif parameterindex == 4:
            if min(parameterrange)<0.2159 or max(parameterrange) > 0.4445:
                print('Error: Diameter outise allowable range [0.2159 to 0.4445 m]')
                sys.exit()
        elif parameterindex == 5:
            if min(parameterrange)<303.15 or max(parameterrange) > 333.15:
                print('Error: Injection temperature outside allowable range [303.15 to 333.15 K]')
                sys.exit()
        elif parameterindex == 6:
            if min(parameterrange)<1.5 or max(parameterrange) > 4.5:
                print('Error: Rock thermal conductivity outise allowable range [1.5 to 4.5 W/m-K]')
                sys.exit()
    

                
        #cycle through array to update object and recalculated TEA results, then store them in arrays
        for i in range(len(parameterrange)):
            print('Line plot calculation: '+str(i+1)+'/'+str(len(parameterrange)))
            if parameterindex == 0:
                self.Flow_user = parameterrange[i]
                xlabel = 'Flow rate [kg/s]'
            elif parameterindex == 1:
                self.Hor_length_user = parameterrange[i]
                xlabel = 'Horizontal length [m]'
            elif parameterindex == 2:
                self.Depth_user = parameterrange[i]
                xlabel = 'Depth [m]'
            elif parameterindex == 3:
                self.Gradient_user = parameterrange[i]
                xlabel = 'Geothermal gradient [deg.C/m]'
            elif parameterindex == 4:
                self.Diameter_user = parameterrange[i]
                xlabel = 'Diameter [m]'
            elif parameterindex == 5:
                self.Tin_user = parameterrange[i]
                self.T_in = self.Tin_user-273.15   #Injection temperature [deg.C]
                xlabel = 'Injection temperature [deg.C]'
            elif parameterindex == 6:
                self.krock_user = parameterrange[i]     
                xlabel = 'Rock thermal conductivity [W/m-K]'
               
            #update self.point
            self.point = (self.Flow_user, self.Hor_length_user, self.Depth_user, self.Gradient_user, self.Diameter_user, self.Tin_user, self.krock_user)
            
            self.getTandP()
            self.calculateLC()
            if np.in1d(1000,self.error_codes) == False:
                avekwthrange[i] = self.AveInstHeatProduction
                if self.End_use==1:
                    lcoerange[i] = self.LCOH
                else:
                    lcoerange[i] = self.LCOE
                    avekwerange[i] = self.AveInstNetElectricityProduction
            else:
                lcoerange[i] = 9999
            
                
        #restore object to its original condition
        self.Flow_user = flow_original
        self.Hor_length_user = hor_length_original
        self.Depth_user = depth_original
        self.Gradient_user = gradient_original
        self.Diameter_user = diameter_original
        self.Tin_user = Tin_original
        self.T_in = self.Tin_user-273.15
        self.krock_user = krock_original
        self.point = (self.Flow_user, self.Hor_length_user, self.Depth_user, self.Gradient_user, self.Diameter_user, self.Tin_user, self.krock_user)
        
        #create 1D line plots
        plt.figure()
        plt.plot(parameterrange, lcoerange, color = 'blue', linestyle = '-', linewidth='2')
        plt.grid()
        plt.xlim([parameterrange[0], parameterrange[-1]])
        plt.ylim([0, 1.05 * np.max(lcoerange)])
        plt.xlabel(xlabel)
        if self.End_use == 1:
            plt.ylabel('Levelized cost of heat [$/MWh]')
        else:
            plt.ylabel('Levelized cost of electricity [$/MWh]')
        plt.show()
        
        plt.figure()
        plt.plot(parameterrange, avekwthrange/1000, color = 'blue', linestyle = '-', linewidth='2')
        plt.grid()
        plt.xlim([parameterrange[0], parameterrange[-1]])
        plt.ylim([0, 1.05 * np.max(avekwthrange/1000)])
        plt.xlabel(xlabel)
        plt.ylabel('Average heat production [$MW_{t}$]')
        plt.show()
        
        if self.End_use == 2:
            plt.figure()
            plt.plot(parameterrange, avekwerange, color = 'blue', linestyle = '-', linewidth='2')
            plt.grid()
            plt.xlim([parameterrange[0], parameterrange[-1]])
            plt.ylim([0, 1.05 * np.max(avekwerange)])
            plt.xlabel(xlabel)
            plt.ylabel('Average electricity production [$kW_{e}$]')
            plt.show()
            
    def contourplot(self,parameter1range,parameter1index,parameter2range,parameter2index,newtitle):
        print(" ")
        print('### Contour Plot Calculation ###')       
        
        #create arrays to store result
        lcoerange = np.zeros((len(parameter1range),len(parameter2range)))
        lcoerange100 = np.zeros((len(parameter1range),len(parameter2range)))
        lcoerange500 = np.zeros((len(parameter1range),len(parameter2range)))
        lcoerange1500 = np.zeros((len(parameter1range),len(parameter2range)))
        avekwthrange = np.zeros((len(parameter1range),len(parameter2range)))
        avekwerange = np.zeros((len(parameter1range),len(parameter2range)))
        
        # 0 = Flow rate; 1 =  Horizontal length; 2 = Depth; 3 = Gradient; 4 = Diameter; 5 = Injection temperature; 6 = Rock thermal conductivity
        #store originals
        flow_original = self.Flow_user
        hor_length_original = self.Hor_length_user
        depth_original = self.Depth_user
        gradient_original = self.Gradient_user
        diameter_original = self.Diameter_user
        Tin_original = self.Tin_user
        krock_original = self.krock_user
        
        #verify input
        if parameter1index == 0:
            if min(parameter1range)<5 or max(parameter1range) > 100:
                print('Error: Flow rate outise allowable range [5 to 100 kg/s]')
                sys.exit()
        elif parameter1index == 1:
            if min(parameter1range)<1000 or max(parameter1range) > 20000:
                print('Error: Horizontal length outise allowable range [1,000 to 20,000 m]')
                sys.exit()
        elif parameter1index == 2:
            if min(parameter1range)<1000 or max(parameter1range) > 5000:
                print('Error: Depth outise allowable range [1,000 to 5,000 m]')
                sys.exit()
        elif parameter1index == 3:
            if min(parameter1range)<0.03 or max(parameter1range) > 0.07:
                print('Error: Gradient outside allowable range [0.03 to 0.07 deg.C/m]')
                sys.exit()
        elif parameter1index == 4:
            if min(parameter1range)<0.2159 or max(parameter1range) > 0.4445:
                print('Error: Diameter outise allowable range [0.2159 to 0.4445 m]')
                sys.exit()
        elif parameter1index == 5:
            if min(parameter1range)<303.15 or max(parameter1range) > 333.15:
                print('Error: Injection temperature outside allowable range [303.15 to 333.15 K]')
                sys.exit()
        elif parameter1index == 6:
            if min(parameter1range)<1.5 or max(parameter1range) > 4.5:
                print('Error: Rock thermal conductivity outise allowable range [1.5 to 4.5 W/m-K]')
                sys.exit()
        if parameter2index == 0:
            if min(parameter2range)<5 or max(parameter2range) > 100:
                print('Error: Flow rate outise allowable range [5 to 100 kg/s]')
                sys.exit()
        elif parameter2index == 1:
            if min(parameter2range)<1000 or max(parameter2range) > 20000:
                print('Error: Horizontal length outise allowable range [1,000 to 20,000 m]')
                sys.exit()
        elif parameter2index == 2:
            if min(parameter2range)<1000 or max(parameter2range) > 5000:
                print('Error: Depth outise allowable range [1,000 to 5,000 m]')
                sys.exit()
        elif parameter2index == 3:
            if min(parameter2range)<0.03 or max(parameter2range) > 0.07:
                print('Error: Gradient outside allowable range [0.03 to 0.07 deg.C/m]')
                sys.exit()
        elif parameter2index == 4:
            if min(parameter2range)<0.2159 or max(parameter2range) > 0.4445:
                print('Error: Diameter outise allowable range [0.2159 to 0.4445 m]')
                sys.exit()
        elif parameter2index == 5:
            if min(parameter2range)<303.15 or max(parameter2range) > 333.15:
                print('Error: Injection temperature outside allowable range [303.15 to 333.15 K]')
                sys.exit()
        elif parameter2index == 6:
            if min(parameter2range)<1.5 or max(parameter2range) > 4.5:
                print('Error: Rock thermal conductivity outise allowable range [1.5 to 4.5 W/m-K]')
                sys.exit()        
        
        #cycle through double array to update object and recalculated TEA results, then store them in arrays
        for i in range(len(parameter1range)):
            if parameter1index == 0:
                self.Flow_user = parameter1range[i]
                xlabel = 'Flow rate [kg/s]'
            elif parameter1index == 1:
                self.Hor_length_user = parameter1range[i]
                xlabel = 'Horizontal length [m]'
            elif parameter1index == 2:
                self.Depth_user = parameter1range[i]
                xlabel = 'Depth [m]'
            elif parameter1index == 3:
                self.Gradient_user = parameter1range[i]
                xlabel = 'Geothermal gradient [deg.C/m]'
            elif parameter1index == 4:
                self.Diameter_user = parameter1range[i]
                xlabel = 'Diameter [m]'
            elif parameter1index == 5:
                self.Tin_user = parameter1range[i]
                self.T_in = self.Tin_user-273.15   #Injection temperature [deg.C]
                xlabel = 'Injection temperature [deg.C]'
            elif parameter1index == 6:
                self.krock_user = parameter1range[i]  
                
            for j in range(len(parameter2range)):
                print('Contour plot calculation: '+str(j+1+i*len(parameter2range))+'/'+str(len(parameter1range)*len(parameter2range)))        
                if parameter2index == 0:
                    self.Flow_user = parameter2range[j]
                    ylabel = 'Flow rate [kg/s]'
                elif parameter2index == 1:
                    self.Hor_length_user = parameter2range[j]
                    ylabel = 'Horizontal length [m]'
                elif parameter2index == 2:
                    self.Depth_user = parameter2range[j]
                    ylabel = 'Depth [m]'
                elif parameter2index == 3:
                    self.Gradient_user = parameter2range[j]
                    ylabel = 'Geothermal gradient [deg.C/m]'
                elif parameter2index == 4:
                    self.Diameter_user = parameter2range[j]
                    ylabel = 'Diameter [m]'
                elif parameter2index == 5:
                    self.Tin_user = parameter2range[j]
                    self.T_in = self.Tin_user-273.15   #Injection temperature [deg.C]
                    ylabel = 'Injection temperature [deg.C]'
                elif parameter2index == 6:
                    self.krock_user = parameter2range[j]                  
   
    
   
                #update self.point
                self.point = (self.Flow_user, self.Hor_length_user, self.Depth_user, self.Gradient_user, self.Diameter_user, self.Tin_user, self.krock_user)
                self.error_codes = np.zeros(0)
                self.getTandP()
                self.calculateLC()
                if np.in1d(1000,self.error_codes) == False:
                    avekwthrange[(i,j)] = self.AveInstHeatProduction
                        
                    if self.End_use==1:
                        lcoerange[(i,j)] = self.LCOH
                        lcoerange100[(i,j)] = self.LCOH100
                        lcoerange500[(i,j)] = self.LCOH500
                        lcoerange1500[(i,j)] = self.LCOH1500
                    else:
                        lcoerange[(i,j)] = self.LCOE
                        lcoerange100[(i,j)] = self.LCOE100
                        lcoerange500[(i,j)] = self.LCOE500
                        lcoerange1500[(i,j)] = self.LCOE1500
                        avekwerange[(i,j)] = self.AveInstNetElectricityProduction
                else:
                    lcoerange[(i,j)] = float("nan")
                    lcoerange100[(i,j)] = float("nan")
                    lcoerange500[(i,j)] = float("nan")
                    lcoerange1500[(i,j)] = float("nan")

        lcoerange100[lcoerange100>1000] = float("nan")
        lcoerange500[lcoerange500>2000] = float("nan")
        lcoerange1500[lcoerange1500>5000] = float("nan")
    
        
        self.avekwthrange = avekwthrange
        self.lcoerange = lcoerange
        self.lcoerange100 = lcoerange100
        self.lcoerange500 = lcoerange500
        self.lcoerange1500 = lcoerange1500
                
        #restore object to its original condition
        self.Flow_user = flow_original
        self.Hor_length_user = hor_length_original
        self.Depth_user = depth_original
        self.Gradient_user = gradient_original
        self.Diameter_user = diameter_original
        self.Tin_user = Tin_original
        self.T_in = self.Tin_user-273.15
        self.krock_user = krock_original
        self.point = (self.Flow_user, self.Hor_length_user, self.Depth_user, self.Gradient_user, self.Diameter_user, self.Tin_user, self.krock_user)
        
        #make plots
        X, Y = np.meshgrid(parameter1range, parameter2range)
        fig, ax = plt.subplots()
        contour = plt.contourf(X,Y, np.transpose(avekwthrange), levels = 32, cmap=cm.coolwarm)   
        ax.set(xlim=(X.min(), X.max()),ylim=(Y.min(), Y.max()))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.title('Average Heat Production')
        cbar = plt.colorbar(contour)
        cbar.set_label('Average Heat Production [kWth]')
        plt.show()
        
        newtitle100 = newtitle +str(100)+'/m'
        newtitle500 = newtitle +str(500)+'/m'
        newtitle1500 = newtitle +str(1500)+'/m'
        
        #fig, ax = plt.subplots()
        #contour = plt.contourf(X,Y, np.transpose(lcoerange), levels = 32, cmap=cm.coolwarm)   
        #ax.set(xlim=(X.min(), X.max()),ylim=(Y.min(), Y.max()))
        #ax.set_xlabel(xlabel)
        #ax.set_ylabel(ylabel)
        #plt.title(newtitle)
        #cbar = plt.colorbar(contour)
        #if self.End_use == 1:
        #    cbar.set_label('LCOH [$/MWh]')
        #elif self.End_use == 2:
        #    cbar.set_label('LCOE [$/MWh]')
        #plt.show()
        
        fig, ax = plt.subplots()
        contour = plt.contourf(X,Y, np.transpose(lcoerange100), levels = 32, cmap=cm.coolwarm)   
        ax.set(xlim=(X.min(), X.max()),ylim=(Y.min(), Y.max()))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #if self.End_use == 1:
        #    newtile = 'U-loop | water | $'+str(self.Drilling_cost_per_m)+'/m'
        #    plt.title(newtile)
        #elif self.End_use == 2:
        plt.title(newtitle100)
        cbar = plt.colorbar(contour)
        if self.End_use == 1:
            cbar.set_label('LCOH [$/MWh]')
        elif self.End_use == 2:
            cbar.set_label('LCOE [$/MWh]')
        plt.show()        

        fig, ax = plt.subplots()
        contour = plt.contourf(X,Y, np.transpose(lcoerange500), levels = 32, cmap=cm.coolwarm)   
        ax.set(xlim=(X.min(), X.max()),ylim=(Y.min(), Y.max()))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #if self.End_use == 1:
        #    newtile = 'U-loop | water | $'+str(self.Drilling_cost_per_m)+'/m'
        #    plt.title(newtile)
        #elif self.End_use == 2:
        plt.title(newtitle500)
        cbar = plt.colorbar(contour)
        if self.End_use == 1:
            cbar.set_label('LCOH [$/MWh]')
        elif self.End_use == 2:
            cbar.set_label('LCOE [$/MWh]')
        plt.show()   
        
        fig, ax = plt.subplots()
        contour = plt.contourf(X,Y, np.transpose(lcoerange1500), levels = 32, cmap=cm.coolwarm)   
        ax.set(xlim=(X.min(), X.max()),ylim=(Y.min(), Y.max()))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #if self.End_use == 1:
        #    newtile = 'U-loop | water | $'+str(self.Drilling_cost_per_m)+'/m'
        #    plt.title(newtile)
        #elif self.End_use == 2:
        
        plt.title(newtitle1500)
        cbar = plt.colorbar(contour)
        if self.End_use == 1:
            cbar.set_label('LCOH [$/MWh]')
        elif self.End_use == 2:
            cbar.set_label('LCOE [$/MWh]')
        plt.show()   
        
        if self.End_use == 2:
            fig, ax = plt.subplots()
            contour = plt.contourf(X,Y, np.transpose(avekwerange), levels = 32, cmap=cm.coolwarm)   
            ax.set(xlim=(X.min(), X.max()),ylim=(Y.min(), Y.max()))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.title('Average Electricity Production')
            cbar = plt.colorbar(contour)
            cbar.set_label('Average Electricity Production [kWe]')
            plt.show()

    def optimalLCOH(self):
        #parameters to vary: FlowRateVector,HorizontalLengthVector,DepthVector, , TinVector, KrockVector

        
        #store originals
        flow_original = self.Flow_user
        hor_length_original = self.Hor_length_user
        depth_original = self.Depth_user
        gradient_original = self.Gradient_user
        diameter_original = self.Diameter_user
        Tin_original = self.Tin_user
        krock_original = self.krock_user
        
        #create optimum vectors
        flowratevectorforoptimum = self.FlowRateVector 
        horizontallengthvectorforoptimum = self.HorizontalLengthVector
        depthvectorforoptimum = self.DepthVector
        tinvectorforoptimum = self.TinVector
        
        #create zero vectors to store results
        avekwthrange = np.zeros([len(flowratevectorforoptimum),len(horizontallengthvectorforoptimum),len(depthvectorforoptimum),len(tinvectorforoptimum)])
        lcoerange = np.zeros([len(flowratevectorforoptimum),len(horizontallengthvectorforoptimum),len(depthvectorforoptimum),len(tinvectorforoptimum)])
        avekwerange = np.zeros([len(flowratevectorforoptimum),len(horizontallengthvectorforoptimum),len(depthvectorforoptimum),len(tinvectorforoptimum)])
        
        #cycle through array to update object and recalculated TEA results, then store them in arrays
        for i in range(len(flowratevectorforoptimum)):
            print('Optimum calculation: '+str(i+1)+'/'+str(len(flowratevectorforoptimum)))
            self.Flow_user = flowratevectorforoptimum[i]

            for j in range(len(horizontallengthvectorforoptimum)):
                self.Hor_length_user = horizontallengthvectorforoptimum[j]
                
                for k in range(len(depthvectorforoptimum)):
                    self.Depth_user = depthvectorforoptimum[k]
                    
                    for l in range(len(tinvectorforoptimum)):
                        self.Tin_user = tinvectorforoptimum[l]
                        self.T_in = self.Tin_user-273.15
                        
    #            elif parameterindex == 2:
    #                self.Depth_user = parameterrange[i]
    #                xlabel = 'Depth [m]'
    #            elif parameterindex == 3:
    #                self.Gradient_user = parameterrange[i]
    #                xlabel = 'Geothermal gradient [deg.C/m]'
    #            elif parameterindex == 4:
    #                self.Diameter_user = parameterrange[i]
    #                xlabel = 'Diameter [m]'
    #            elif parameterindex == 5:
    #                self.Tin_user = parameterrange[i]
    #                self.T_in = self.Tin_user-273.15   #Injection temperature [deg.C]
    #                xlabel = 'Injection temperature [deg.C]'
    #            elif parameterindex == 6:
    #                self.krock_user = parameterrange[i]     
    #                xlabel = 'Rock thermal conductivity [W/m-K]'
               
                    #update self.point
                        self.point = (self.Flow_user, self.Hor_length_user, self.Depth_user, self.Gradient_user, self.Diameter_user, self.Tin_user, self.krock_user)
                        
                        self.getTandP()
                        self.calculateLC()
                                               
                        if np.in1d(1000,self.error_codes) == False:
                            avekwthrange[i,j,k,l] = self.AveInstHeatProduction
                            if self.End_use==1:
                                lcoerange[i,j,k,l] = self.LCOH
                            else:
                                lcoerange[i,j,k,l] = self.LCOE
                                avekwerange[i,j,k,l] = self.AveInstNetElectricityProduction
                        else:
                            lcoerange[i,j,k,l] = 9999
                
                        
                            

        #store results
        self.avekwthrange = avekwthrange
        self.lcoerange = lcoerange
        self.avekwerange = avekwerange
        
        
        #restore object to its original condition
        self.Flow_user = flow_original
        self.Hor_length_user = hor_length_original
        self.Depth_user = depth_original
        self.Gradient_user = gradient_original
        self.Diameter_user = diameter_original
        self.Tin_user = Tin_original
        self.T_in = self.Tin_user-273.15
        self.krock_user = krock_original
        self.point = (self.Flow_user, self.Hor_length_user, self.Depth_user, self.Gradient_user, self.Diameter_user, self.Tin_user, self.krock_user)
        
        
    def optimalLCOH2(self):
        #parameters to vary: FlowRateVector,HorizontalLengthVector,DepthVector, , TinVector, KrockVector

        
        #store originals
        flow_original = self.Flow_user
        hor_length_original = self.Hor_length_user
        depth_original = self.Depth_user
        gradient_original = self.Gradient_user
        diameter_original = self.Diameter_user
        Tin_original = self.Tin_user
        krock_original = self.krock_user
        
        #create optimum vectors
        flowratevectorforoptimum = self.FlowRateVector 
        horizontallengthvectorforoptimum = self.HorizontalLengthVector
        tinvectorforoptimum = self.TinVector
        
        #create zero vectors to store results
        avekwthrange = np.zeros([len(flowratevectorforoptimum),len(horizontallengthvectorforoptimum),len(tinvectorforoptimum)])
        lcoerange = np.zeros([len(flowratevectorforoptimum),len(horizontallengthvectorforoptimum),len(tinvectorforoptimum)])
        avekwerange = np.zeros([len(flowratevectorforoptimum),len(horizontallengthvectorforoptimum),len(tinvectorforoptimum)])
        
        #cycle through array to update object and recalculated TEA results, then store them in arrays
        for i in range(len(flowratevectorforoptimum)):
            print('Optimum calculation: '+str(i+1)+'/'+str(len(flowratevectorforoptimum)))
            self.Flow_user = flowratevectorforoptimum[i]

            for j in range(len(horizontallengthvectorforoptimum)):
                self.Hor_length_user = horizontallengthvectorforoptimum[j]
                
                    
                for l in range(len(tinvectorforoptimum)):
                    self.Tin_user = tinvectorforoptimum[l]
                    self.T_in = self.Tin_user-273.15
                    
#            elif parameterindex == 2:
#                self.Depth_user = parameterrange[i]
#                xlabel = 'Depth [m]'
#            elif parameterindex == 3:
#                self.Gradient_user = parameterrange[i]
#                xlabel = 'Geothermal gradient [deg.C/m]'
#            elif parameterindex == 4:
#                self.Diameter_user = parameterrange[i]
#                xlabel = 'Diameter [m]'
#            elif parameterindex == 5:
#                self.Tin_user = parameterrange[i]
#                self.T_in = self.Tin_user-273.15   #Injection temperature [deg.C]
#                xlabel = 'Injection temperature [deg.C]'
#            elif parameterindex == 6:
#                self.krock_user = parameterrange[i]     
#                xlabel = 'Rock thermal conductivity [W/m-K]'
           
                #update self.point
                    self.point = (self.Flow_user, self.Hor_length_user, self.Depth_user, self.Gradient_user, self.Diameter_user, self.Tin_user, self.krock_user)
                    
                    self.getTandP()
                    self.calculateLC()
                                           
                    if np.in1d(1000,self.error_codes) == False:
                        avekwthrange[i,j,l] = self.AveInstHeatProduction
                        if self.End_use==1:
                            lcoerange[i,j,l] = self.LCOH
                        else:
                            lcoerange[i,j,l] = self.LCOE
                            avekwerange[i,j,l] = self.AveInstNetElectricityProduction
                    else:
                        lcoerange[i,j,l] = 9999
            
                        
                            

        #store results
        self.avekwthrange = avekwthrange
        self.lcoerange = lcoerange
        self.avekwerange = avekwerange
        
        
        #restore object to its original condition
        self.Flow_user = flow_original
        self.Hor_length_user = hor_length_original
        self.Depth_user = depth_original
        self.Gradient_user = gradient_original
        self.Diameter_user = diameter_original
        self.Tin_user = Tin_original
        self.T_in = self.Tin_user-273.15
        self.krock_user = krock_original
        self.point = (self.Flow_user, self.Hor_length_user, self.Depth_user, self.Gradient_user, self.Diameter_user, self.Tin_user, self.krock_user)        