import numpy as np
import pdb 
import math 

from fgem.utils.utils import heatcapacitywater

class BasePowerPlant(object):
    
    """Base class for defining a power plant."""
    
    def __init__(self, ppc, Tres, cf):
        """Define attributes for the BasePowerPlant class."""
        self.ppc = ppc
        self.cf = cf
        self.T_mix = Tres

    def extend_ppc(self, added_ppc):
        """Increase power plant capacity."""
        self.ppc += added_ppc

    def compute_power_output(self, T, T0):
        raise NotImplementedError
    
    def compute_injection_temp(self, T, T0):
        raise NotImplementedError

    def compute_cplant(self, MaxProducedTemperature):
        raise NotImplementedError

    def compute_thermalexergy(self, T, T0):
        
        """Compute power plant thermal exergy for an inflowing geofluid at a given ambient temperature."""
        
        A = 4.041650
        B = -1.204E-2
        C = 1.60500E-5
        T0_k = T0 + 273.15 #deg K
        T_k = T + 273.15 #deg K
        thermal_exergy = ((A-B*T0_k)*(T_k-T0_k)+(B-C*T0_k)/2.0*(T_k**2-T0_k**2)+C/3.0*(T_k**3-T0_k**3)-A*T0_k*np.log(T_k/T0_k))*2.2046/947.83 #MJ/kg
        return thermal_exergy
    
    def power_plant_outputs(self,
                            timestep,
                            m_turbine,
                            m_wh_to_turbine,
                            m_tes_out,
                            T_wh,
                            T_tes_out,
                            T0):
        
        """Get power plant outputs for a timestep."""

        # If we are using all geofluid to charge the tank, then there is no output
        if m_turbine == 0:
            self.T_inj = self.compute_injection_temp(self.T_mix, T0)
            return 0.0, 0.0, 0.0, 0.0, self.T_inj
        else:
            self.T_mix = (m_wh_to_turbine*heatcapacitywater(T_wh)*T_wh + m_tes_out*heatcapacitywater(T_tes_out)*T_tes_out)/(m_wh_to_turbine*heatcapacitywater(T_wh)+m_tes_out*heatcapacitywater(T_tes_out)+1e-3)
            power_output_MWh_kg = self.compute_power_output(self.T_mix, T0)
            power_output_MWe = self.cf * min(self.ppc, power_output_MWh_kg * (m_turbine * 3600))
            power_generation_MWh = power_output_MWe * (timestep.total_seconds()/3600)
            self.T_inj = self.compute_injection_temp(self.T_mix, T0)

        return power_output_MWh_kg, power_output_MWe, power_generation_MWh, self.T_mix, self.T_inj
    
class ORCPowerPlant(BasePowerPlant):
    
    """ORC Binary power plant."""
    
    def __init__(self, ppc, Tres, cf=1.0):
        """Define attributes for the ORCPowerPlant class."""
        super(ORCPowerPlant, self).__init__(ppc, Tres, cf)
    
    def compute_cplant(self, MaxProducedTemperature):
        """Compute cost of an ORC binary power plant."""
        if (MaxProducedTemperature < 150.):
            C3 = -1.458333E-3
            C2 = 7.6875E-1 
            C1 = -1.347917E2
            C0 = 1.0075E4
            CCAPP1 = C3*MaxProducedTemperature**3 + C2*MaxProducedTemperature**2 + C1*MaxProducedTemperature + C0
        else:
            CCAPP1 = 2231 - 2*(MaxProducedTemperature-150.)
        Cplantcorrelation = CCAPP1*math.pow(self.ppc/15.,-0.06) * 1e-6 * self.ppc * 1e3 #$MM
        return Cplantcorrelation
    
    def compute_power_output(self, T, T0):
        """Compute the power output in MWh electricity per kg geofluid of an ORC binary power plant."""
        thermal_exergy = self.compute_thermalexergy(T, T0)
        if (T0 < 15.):
            C1 = 2.746E-3
            C0 = -8.3806E-2
            D1 = 2.713E-3
            D0 = -9.1841E-2
            Tfraction = (T0-5.)/10.
        else:
            C1 = 2.713E-3
            C0 = -9.1841E-2
            D1 = 2.676E-3
            D0 = -1.012E-1
            Tfraction = (T0-15.)/10.
        etaull = C1*T + C0
        etauul = D1*T + D0
        etau = (1-Tfraction)*etaull + Tfraction*etauul
        
        power_output = max(thermal_exergy*etau/3600, 0.0) #MWh/kg
    
        return power_output #MWh/kg
    
    def compute_injection_temp(self, T, T0):
        """Compute the exiting (reinjection) water temperature of an ORC binary power plant."""
        if (T0 < 15.):
            C1 = 0.0894
            C0 = 55.6
            D1 = 0.0894
            D0 = 62.6
            Tfraction = (T0-5.)/10.
        else:
            C1 = 0.0894
            C0 = 62.6
            D1 = 0.0894
            D0 = 69.6
            Tfraction = (T0-15.)/10.
        reinjtll = C1*T + C0
        reinjtul = D1*T + D0
        Tinj = max((1.-Tfraction)*reinjtll + Tfraction*reinjtul, 0.0) #deg C

        return Tinj #deg C
    
class FlashPowerPlant(BasePowerPlant):
    """Single Flash Binary power plant."""
    def __init__(self, ppc, Tres, cf=1.0):
        """Define attributes for the FlashPowerPlant class."""
        super(FlashPowerPlant, self).__init__(ppc, Tres, cf)
    
    def compute_cplant(self, MaxProducedTemperature):
        """Compute cost of a single flash binary power plant."""
        if self.ppc < 10:
            C2 = 4.8472E-2 
            C1 = -35.2186
            C0 = 8.4474E3
            D2 = 4.0604E-2 
            D1 = -29.3817
            D0 = 6.9911E3
            PLL = 5.
            PRL = 10.
        elif self.ppc < 25:
            C2 = 4.0604E-2 
            C1 = -29.3817
            C0 = 6.9911E3	  
            D2 = 3.2773E-2 
            D1 = -23.5519
            D0 = 5.5263E3        
            PLL = 10.
            PRL = 25.
        elif self.ppc < 50:
            C2 = 3.2773E-2 
            C1 = -23.5519
            C0 = 5.5263E3
            D2 = 3.4716E-2 
            D1 = -23.8139
            D0 = 5.1787E3	          
            PLL = 25.
            PRL = 50.
        elif self.ppc < 75:
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
        Cplantcorrelation = 0.8*a*math.pow(self.ppc,b)*self.ppc*1000./1e6 #factor 0.75 to make double flash 25% more expansive than single flash
    
        return Cplantcorrelation
    
    def compute_power_output(self, T, T0):
        """Compute the power output in MWh electricity per kg geofluid of a single flash binary power plant."""
        thermal_exergy = self.compute_thermalexergy(T, T0)
        if (T0 < 15.):
            C2 = -4.27318E-7
            C1 = 8.65629E-4
            C0 = 1.78931E-1
            D2 = -5.85412E-7
            D1 = 9.68352E-4
            D0 = 1.58056E-1
            Tfraction = (T0-5.)/10.
        else:
            C2 = -5.85412E-7
            C1 = 9.68352E-4
            C0 = 1.58056E-1
            D2 = -7.78996E-7
            D1 = 1.09230E-3
            D0 = 1.33708E-1
            Tfraction = (T0-15.)/10.
        etaull = C2*T**2 + C1*T + C0
        etauul = D2*T**2 + D1*T + D0
        etau = (1.-Tfraction)*etaull + Tfraction*etauul
        
        power_output = max(thermal_exergy*etau/3600, 0.0) #MWh/kg
    
        return power_output #MWh/kg
    
    def compute_injection_temp(self, T, T0):
        """Compute the exiting (reinjection) water temperature of a single flash power plant."""
        if (T0 < 15.):
            C2 = -1.11519E-3
            C1 = 7.79126E-1
            C0 = -10.2242
            D2 = -1.10232E-3
            D1 = 7.83893E-1
            D0 = -5.17039
            Tfraction = (T0-5.)/10.
        else:
            C2 = -1.10232E-3
            C1 = 7.83893E-1
            C0 = -5.17039
            D2 = -1.08914E-3
            D1 = 7.88562E-1
            D0 = -1.89707E-1
            Tfraction = (T0-15.)/10.
        reinjtll = C2*T**2 + C1*T + C0
        reinjtul = D2*T**2 + D1*T + D0
        Tinj = max((1.-Tfraction)*reinjtll + Tfraction*reinjtul, 0.0)
        
        return Tinj #deg C
