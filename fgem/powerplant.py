import numpy as np
import pdb 
import math
import os
from .utils.utils import *
import pickle
from scipy.spatial import cKDTree
from datetime import timedelta, datetime
from pathlib import Path
parent_path = Path(__file__).parent

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

class BasePowerPlant(object):
    
    """Base class for defining a power plant."""
    
    def __init__(self, ppc, Tres, cf,
                 timestep=timedelta(hours=1)):
        """Define attributes for the BasePowerPlant class.

        Args:
            ppc (float): power plant nameplate capacity in MW.
            Tres (float): reservoir temperature in deg C.
            cf (float): capacity factor.
            timestep (datetime, optional): simulation timestep size. Defaults to timedelta(hours=1).
        """
        self.powerplant_capacity = ppc
        self.cf = cf

        #default starting point
        self.power_output_MWh_kg = 1e-5
        self.power_output_MWe = ppc
        self.power_generation_MWh = ppc * 1
        self.T_mix = Tres
        self.T_inj = Tres/2
        self.timestep = timestep

    def compute_geofluid_consumption(self, T, T_amb, m_turbine):
        """Compute geofluid consumption.

        Args:
            T (float): power plant inlet temperature in deg C.
            T_amb (float): ambient temperautre in deg C.

        Raises:
            NotImplementedError: you must implemented this.
        """
        raise NotImplementedError
    
    def compute_injection_temp(self, T, T_amb, m_turbine):
        """Compute injection temperature.

        Args:
            T (float): power plant inlet temperature in deg C.
            T_amb (float): ambient temperautre in deg C.

        Raises:
            NotImplementedError: you must implement this.
        """
        raise NotImplementedError

    def compute_cplant(self, MaxProducedTemperature, min_cost=0):
        """Compute power plant capex.

        Args:
            MaxProducedTemperature (float): maximum produced geofluid temperature in deg C.

        Raises:
            NotImplementedError: you must implement this.
        """
        raise NotImplementedError

    def compute_thermalexergy(self, T, T_amb):
        """Compute power plant thermal exergy for an inflowing geofluid at a given ambient temperature.

        Args:
            T (float): power plant inlet temperature in deg C.
            T_amb (float): ambient temperature in deg C.

        Returns:
            float: thermal exergy in MJ/kg.
        """
        
        A = 4.041650
        B = -1.204E-2
        C = 1.60500E-5
        T_amb_k = T_amb + 273.15 #deg K
        T_k = T + 273.15 #deg K
        thermal_exergy = ((A-B*T_amb_k)*(T_k-T_amb_k)+(B-C*T_amb_k)/2.0*(T_k**2-T_amb_k**2)+C/3.0*(T_k**3-T_amb_k**3)-A*T_amb_k*np.log(T_k/T_amb_k))*2.2046/947.83 #MJ/kg

        return thermal_exergy
    
    def compute_power_output(self, m_turbine):
        return self.cf * min(self.powerplant_capacity, self.power_output_MWh_kg * (m_turbine * 3600))
    
    def step(self,
            m_turbine,
            T_prd_wh,
            T_amb,
            m_tes_out=0,
            T_tes_out=100,):

        """Step and update power plant.

        Args:
            m_turbine (float): total mass flowing into the turbine in kg/s (including flow from both producers and thermal energy storage tank).
            T_prd_wh (float): producer wellheat temperature in deg C.
            T_amb (float): ambient temperature in deg C.
            m_tes_out (float, optional): mass flow rate from the thermal energy storage tank in kg/s. Defaults to 0.
            T_tes_out (float, optional): temperature of water of the thermal energy storage tank. Defaults to 100.

        """

        # If we are using all geofluid to charge the tank, then there is no output
        if m_turbine == 0:
            self.power_output_MWe = 0
            self.power_generation_MWh = 0
            self.T_inj = self.compute_injection_temp(self.T_mix, T_amb, m_turbine)
            return 0.0, 0.0, 0.0, 0.0, self.T_inj
        else:
            if m_tes_out>0:
                m_wh_to_turbine = m_turbine - m_tes_out
                self.T_mix = (m_wh_to_turbine*heatcapacitywater(T_prd_wh)*T_prd_wh + m_tes_out*heatcapacitywater(T_tes_out)*T_tes_out)/(m_wh_to_turbine*heatcapacitywater(T_prd_wh)+m_tes_out*heatcapacitywater(T_tes_out)+1e-3)
            else:
                self.T_mix = T_prd_wh
                
            self.power_output_MWh_kg = self.compute_geofluid_consumption(self.T_mix, T_amb, m_turbine)
            self.power_output_MWe = self.compute_power_output(m_turbine)
            self.power_generation_MWh = self.power_output_MWe * (self.timestep.total_seconds()/3600)
            self.T_inj = self.compute_injection_temp(self.T_mix, T_amb, m_turbine)

class GEOPHIRESORCPowerPlant(BasePowerPlant):
    
    """ORC Binary power plant based on GEOPHIRES."""
    
    def __init__(self, ppc, Tres, cf=1.0):
        """Define attributes for the BasePowerPlant class.

        Args:
            ppc (float): power plant nameplate capacity in MW.
            Tres (float): reservoir temperature in deg C.
            cf (float): capacity factor. Defualts to 1.
        """
        super(GEOPHIRESORCPowerPlant, self).__init__(ppc, Tres, cf)

    def compute_cplant(self, MaxProducedTemperature, min_cost=0):
        """Compute power plant capex.

        Args:
            MaxProducedTemperature (float): maximum produced geofluid temperature in deg C.

        Returns:
            float: cost of power plant in USD/kW
        """
        if (MaxProducedTemperature < 150.):
            C3 = -1.458333E-3
            C2 = 7.6875E-1 
            C1 = -1.347917E2
            C0 = 1.0075E4
            CCAPP1 = C3*MaxProducedTemperature**3 + C2*MaxProducedTemperature**2 + C1*MaxProducedTemperature + C0
        else:
            CCAPP1 = 2231 - 2*(MaxProducedTemperature-150.)
        
        Cplantcorrelation = max(min_cost, CCAPP1*math.pow(self.powerplant_capacity/15.,-0.06)) * 1e-6 * self.powerplant_capacity * 1e3 #$MM
        return Cplantcorrelation
    
    def compute_geofluid_consumption(self, T, T_amb, m_turbine):
        """Compute geofluid consumption.

        Args:
            T (float): power plant inlet temperature in deg C.
            T_amb (float): ambient temperautre in deg C.

        Returns:
            float: power plant output in MWh/kg.
        """
        
        self.thermal_exergy = self.compute_thermalexergy(T, T_amb)
        if (T_amb < 15.):
            C1 = 2.746E-3
            C0 = -8.3806E-2
            D1 = 2.713E-3
            D0 = -9.1841E-2
            Tfraction = (T_amb-5.)/10.
        else:
            C1 = 2.713E-3
            C0 = -9.1841E-2
            D1 = 2.676E-3
            D0 = -1.012E-1
            Tfraction = (T_amb-15.)/10.
        etaull = C1*T + C0
        etauul = D1*T + D0
        self.etau = (1-Tfraction)*etaull + Tfraction*etauul
        
        power_output_MWh_kg = max(self.thermal_exergy*self.etau/3600, 0.0) #MWh/kg
        
        return power_output_MWh_kg #MWh/kg
    
    def compute_injection_temp(self, T, T_amb, m_turbine):
        """Compute injection temperature.

        Args:
            T (float): power plant inlet temperature in deg C.
            T_amb (float): ambient temperautre in deg C.

        Returns:
            float: power plant condenser outlet temperature in deg C.
        """

        """Compute the exiting (reinjection) water temperature of an ORC binary power plant."""
        
        if (T_amb < 15.):
            C1 = 0.0894
            C0 = 55.6
            D1 = 0.0894
            D0 = 62.6
            Tfraction = (T_amb-5.)/10.
        else:
            C1 = 0.0894
            C0 = 62.6
            D1 = 0.0894
            D0 = 69.6
            Tfraction = (T_amb-15.)/10.
        reinjtll = C1*T + C0
        reinjtul = D1*T + D0
        Tinj = max((1.-Tfraction)*reinjtll + Tfraction*reinjtul, 0.0) #deg C

        return Tinj #deg C
    
class GEOPHIRESFlashPowerPlant(BasePowerPlant):
    """Single Flash Binary power plant."""
    def __init__(self, ppc, Tres, cf=1.0):
        """Define attributes for the BasePowerPlant class.

        Args:
            ppc (float): power plant nameplate capacity in MW.
            Tres (float): reservoir temperature in deg C.
            cf (float): capacity factor. Defualts to 1.
        """
        
        super(GEOPHIRESFlashPowerPlant, self).__init__(ppc, Tres, cf)
        
    def compute_cplant(self, MaxProducedTemperature, min_cost=0):
        """Compute power plant capex.

        Args:
            MaxProducedTemperature (float): maximum produced geofluid temperature in deg C.

        Returns:
            float: cost of power plant in USD/kW
        """

        if self.powerplant_capacity < 10:
            C2 = 4.8472E-2 
            C1 = -35.2186
            C0 = 8.4474E3
            D2 = 4.0604E-2 
            D1 = -29.3817
            D0 = 6.9911E3
            PLL = 5.
            PRL = 10.
        elif self.powerplant_capacity < 25:
            C2 = 4.0604E-2 
            C1 = -29.3817
            C0 = 6.9911E3	  
            D2 = 3.2773E-2 
            D1 = -23.5519
            D0 = 5.5263E3        
            PLL = 10.
            PRL = 25.
        elif self.powerplant_capacity < 50:
            C2 = 3.2773E-2 
            C1 = -23.5519
            C0 = 5.5263E3
            D2 = 3.4716E-2 
            D1 = -23.8139
            D0 = 5.1787E3	          
            PLL = 25.
            PRL = 50.
        elif self.powerplant_capacity < 75:
            C2 = 3.4716E-2 
            C1 = -23.8139
            C0 = 5.1787E3	
            D2 = 3.5271E-2 
            D1 = -24.3962
            D0 = 5.1972E3          
            PLL = 50.
            PRL = 75.
        else:
            C2 = 3.5271E-2 
            C1 = -24.3962
            C0 = 5.1972E3	
            D2 = 3.3908E-2 
            D1 = -23.4890
            D0 = 5.0238E3          
            PLL = 75.
            PRL = 100.
        CCAPPLL = C2*MaxProducedTemperature**2 + C1*MaxProducedTemperature + C0
        CCAPPRL = D2*MaxProducedTemperature**2 + D1*MaxProducedTemperature + D0  
        b = math.log(CCAPPRL/CCAPPLL)/math.log(PRL/PLL)
        a = CCAPPRL/PRL**b
        Cplantcorrelation = max(min_cost, 0.8*a*math.pow(self.powerplant_capacity,b))*self.powerplant_capacity*1000./1e6 #factor 0.75 to make double flash 25% more expansive than single flash
    
        return Cplantcorrelation
    
    def compute_geofluid_consumption(self, T, T_amb, m_turbine):
        """Compute geofluid consumption.

        Args:
            T (float): power plant inlet temperature in deg C.
            T_amb (float): ambient temperautre in deg C.

        Returns:
            float: power plant output in MWh/kg.
        """

        self.thermal_exergy = self.compute_thermalexergy(T, T_amb)
        if (T_amb < 15.):
            C2 = -4.27318E-7
            C1 = 8.65629E-4
            C0 = 1.78931E-1
            D2 = -5.85412E-7
            D1 = 9.68352E-4
            D0 = 1.58056E-1
            Tfraction = (T_amb-5.)/10.
        else:
            C2 = -5.85412E-7
            C1 = 9.68352E-4
            C0 = 1.58056E-1
            D2 = -7.78996E-7
            D1 = 1.09230E-3
            D0 = 1.33708E-1
            Tfraction = (T_amb-15.)/10.
        etaull = C2*T**2 + C1*T + C0
        etauul = D2*T**2 + D1*T + D0
        self.etau = (1.-Tfraction)*etaull + Tfraction*etauul
        
        power_output_MWh_kg = max(self.thermal_exergy*self.etau/3600, 0.0) #MWh/kg
    
        return power_output_MWh_kg #MWh/kg
    
    def compute_injection_temp(self, T, T_amb, m_turbine):
        """Compute injection temperature.

        Args:
            T (float): power plant inlet temperature in deg C.
            T_amb (float): ambient temperautre in deg C.

        Returns:
            float: power plant condenser outlet temperature in deg C.
        """

        """Compute the exiting (reinjection) water temperature of an ORC binary power plant."""
        
        if (T_amb < 15.):
            C2 = -1.11519E-3
            C1 = 7.79126E-1
            C0 = -10.2242
            D2 = -1.10232E-3
            D1 = 7.83893E-1
            D0 = -5.17039
            Tfraction = (T_amb-5.)/10.
        else:
            C2 = -1.10232E-3
            C1 = 7.83893E-1
            C0 = -5.17039
            D2 = -1.08914E-3
            D1 = 7.88562E-1
            D0 = -1.89707E-1
            Tfraction = (T_amb-15.)/10.
        reinjtll = C2*T**2 + C1*T + C0
        reinjtul = D2*T**2 + D1*T + D0
        Tinj = max((1.-Tfraction)*reinjtll + Tfraction*reinjtul, 0.0)
        
        return Tinj #deg C

class HighEnthalpyCLGWGPowerPlant(BasePowerPlant):
    
    """On/Off-Design Air-cooled ORC Binary power plant developed in Python by Aljubran et al. (2024)."""
    
    def __init__(self, Tres, Tamb, m_prd, num_prd, cf=1.0, high_resolution=False, k=2):
        """Define attributes for the BasePowerPlant class.

        Args:
            ppc (float): power plant nameplate capacity in MW.
            Tres (float): reservoir temperature in deg C.
            m_prd (float): mass flow rate of a single producer in the project in kg/s.
            cf (float): capacity factor. Defualts to 1.
            high_resolution (bool): if True, a higher resolution power plant model is loaded, but it is slower
        """
        self.Tres_design = Tres
        self.Tamb_design = Tamb
        self.m_prd_design = m_prd
        self.num_prd = num_prd
        self.k = k 
        self.model = CustomUnpickler(open(os.path.join(parent_path, "data/powerplants", "HighEnthalpyCLGWGPowerPlant.pkl"), 'rb')).load()

        model_input = np.array([[self.Tamb_design, self.Tres_design, self.m_prd_design]])
        model_output = self.model.predict(model_input)[0]

        ppc = model_output[0] * num_prd # MWe power plant capacity for all wells
        super(HighEnthalpyCLGWGPowerPlant, self).__init__(ppc, Tres, cf)

        self.power_output_MWh_kg  =  model_output[0]/self.m_prd_design / 3600 # MWh/kg
        self.T_inj = model_output[1] # deg C

    def compute_cplant(self, MaxProducedTemperature, min_cost=0):
        """Compute power plant capex.

        Args:
            MaxProducedTemperature (float): maximum produced geofluid temperature in deg C.

        Returns:
            float: cost of power plant in USD/kW
        """
        if (MaxProducedTemperature < 150.):
            C3 = -1.458333E-3
            C2 = 7.6875E-1 
            C1 = -1.347917E2
            C0 = 1.0075E4
            CCAPP1 = C3*MaxProducedTemperature**3 + C2*MaxProducedTemperature**2 + C1*MaxProducedTemperature + C0
        else:
            CCAPP1 = 2231 - 2*(MaxProducedTemperature-150.)
        Cplantcorrelation = max(min_cost, CCAPP1*math.pow(self.powerplant_capacity/15.,-0.06)) * 1e-6 * self.powerplant_capacity * 1e3 #$MM
        return Cplantcorrelation
    
    def compute_geofluid_consumption(self, T, T_amb, m_turbine):
        """Compute geofluid consumption.

        Args:
            T (float): power plant inlet temperature in deg C.
            T_amb (float): ambient temperautre in deg C.

        Returns:
            float: power plant output in MWh/kg.
        """

        m_prd = m_turbine/self.num_prd # single well flow rate

        model_input = np.array([[T_amb, T, m_prd]])
        model_output = self.model.predict(model_input)[0]

        self.power_output_MWh_kg  =  max(model_output[0] / m_prd / 3600, 0) # MWh/kg
        if self.power_output_MWh_kg == 0:
            self.T_inj = T
        else:
            self.T_inj = model_output[1] # deg C

        return self.power_output_MWh_kg #MWh/kg
    
    def compute_injection_temp(self, T, T_amb, m_turbine):
        """Compute injection temperature.

        Args:
            T (float): power plant inlet temperature in deg C.
            T_amb (float): ambient temperautre in deg C.

        Returns:
            float: power plant condenser outlet temperature in deg C.
        """

        """Compute the exiting (reinjection) water temperature of an ORC binary power plant."""
        
        return self.T_inj #deg C

class ORCPowerPlant(BasePowerPlant):
    
    """On/Off-Design Air-cooled ORC Binary power plant developed in Python by Aljubran et al. (2024)."""
    
    def __init__(self, Tres, Tamb, m_prd, num_prd=None, ppc=None, cf=1.0, k=2):
        """Define attributes for the BasePowerPlant class.

        Args:
            Tres (float): reservoir temperature in deg C.
            Tamb (float): ambient temperature in deg C.
            m_prd (float): mass flow rate of a single producer in the project in kg/s.
            num_prd (int): number of producers (Note: only used if power plant capacity 'ppc' is not provided)
            ppc (float): power plant nameplate capacity in MW.
            cf (float): capacity factor. Defualts to 1.
        """

        self.Tres_design = Tres
        self.Tamb_design = Tamb
        self.m_prd_design = m_prd
        self.num_prd = num_prd
        self.k = k 
        self.model = CustomUnpickler(open(os.path.join(parent_path, "data/powerplants", "ORCPowerPlant.pkl"), 'rb')).load()

        self.keys = np.array(list(self.model.keys()))
        tree = cKDTree(self.keys)
        distances, indices = tree.query(np.array([[self.Tamb_design, self.Tres_design, self.m_prd_design]]), k=self.k)
        self.distances, self.indices = distances[0], indices[0]
        if isinstance(self.distances, float):
            self.distances, self.indices = np.array([self.distances]), np.array([self.indices])
        self.inv_distances = 1/(self.distances + 1e-3)**2 # important choice on how much to interpolate ... more impactful when simualted data is sparse ...
        self.inv_distances = self.inv_distances/self.inv_distances.sum()
        self.inv_distances = self.inv_distances[:,None]
        self.model_list = [self.model[tuple(self.keys[self.indices][i])] for i in range(self.k)]

        model_input = np.array([[self.Tamb_design, self.Tres_design, self.m_prd_design]])
        preds = np.vstack([m.predict(model_input) for m in self.model_list])
        model_output = (preds * self.inv_distances).sum(axis=0)

        if ppc is None:
            ppc = max(model_output[0] * num_prd, 1) # MWe power plant capacity for all wells
        super(ORCPowerPlant, self).__init__(ppc, Tres, cf)

        self.power_output_MWh_kg  =  np.clip(model_output[0]/self.m_prd_design / 3600, a_min=0.0, a_max=2.5e-4)  # MWh/kg
        self.T_inj = np.clip(model_output[1], a_min=self.Tamb_design, a_max=self.Tres_design) # deg C

        if self.num_prd is None:
            m_g = ppc / nonzero(self.power_output_MWh_kg) / 3600 #kg/s
            self.num_prd = np.ceil(m_g / m_prd).astype(int)

    def compute_cplant(self, MaxProducedTemperature, min_cost=0):
        """Compute power plant capex.

        Args:
            MaxProducedTemperature (float): maximum produced geofluid temperature in deg C.

        Returns:
            float: cost of power plant in USD/kW
        """

        if (MaxProducedTemperature < 150.):
            C3 = -1.458333E-3
            C2 = 7.6875E-1 
            C1 = -1.347917E2
            C0 = 1.0075E4
            CCAPP1 = C3*MaxProducedTemperature**3 + C2*MaxProducedTemperature**2 + C1*MaxProducedTemperature + C0
        else:
            CCAPP1 = 2231 - 2*(MaxProducedTemperature-150.)
        Cplantcorrelation = max(min_cost, CCAPP1*math.pow(self.powerplant_capacity/15.,-0.06)) * 1e-6 * self.powerplant_capacity * 1e3 #$MM
        return Cplantcorrelation
    
    def compute_geofluid_consumption(self, T, T_amb, m_turbine):
        """Compute geofluid consumption.

        Args:
            T (float): power plant inlet temperature in deg C.
            T_amb (float): ambient temperautre in deg C.

        Returns:
            float: power plant output in MWh/kg.
        """

        m_prd = m_turbine/self.num_prd # single well flow rate
        model_input = np.array([[T_amb, T, m_prd]])
        
        preds = np.vstack([m.predict(model_input) for m in self.model_list])
        model_output = (preds * self.inv_distances).sum(axis=0)
        self.power_output_MWh_kg  =  np.clip(model_output[0] / m_prd / 3600, a_min=0.0, a_max=2.5e-4) # MWh/kg
        self.T_inj = np.clip(model_output[1], a_min=T_amb, a_max=T) # deg C

        return self.power_output_MWh_kg #MWh/kg
    
    def compute_injection_temp(self, T, T_amb, m_turbine):
        """Compute injection temperature.

        Args:
            T (float): power plant inlet temperature in deg C.
            T_amb (float): ambient temperautre in deg C.

        Returns:
            float: power plant condenser outlet temperature in deg C.
        """

        """Compute the exiting (reinjection) water temperature of an ORC binary power plant."""
        
        return self.T_inj #deg C