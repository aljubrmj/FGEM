import math
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from fgem.utils.utils import compute_drilling_cost
from fgem.subsurface import *
from fgem.powerplant import *
from fgem.markets import *
from fgem.weather import *
from fgem.storage import *
from pyXSteam.XSteam import XSteam

class World:
    
    """High-level class to define a project involving upstream, midstream, and downstream components."""
    
    def __init__(self, config):
        
        """Defining attributes for the World class."""
        
        # Record major input variables
        self.config = config
        self.config_to_placeholders(self.config)

        self.data_dir = os.path.join(self.base_dir, self.data_dir)
        self.market_filepath = os.path.join(self.data_dir, self.market_dir, self.market_filename)
        self.capacity_filepath = os.path.join(self.data_dir, self.market_dir, self.capacity_filename)
        self.weather_filepath = os.path.join(self.data_dir, self.market_dir, self.weather_filename)
        self.battery_costs_filepath = os.path.join(self.data_dir, self.market_dir, self.battery_costs_filename)
        self.time_init = pd.to_datetime(self.time_init)
        self.start_year = self.time_init.year
        self.end_year = self.start_year + self.L
        self.height = 2 * self.diameter
        self.num_inj = int(np.floor(self.inj_prd_ratio * self.num_prd)) # underestimate the need for injectors since devlopers would often prefer to drill more later if needed
        self.effective_ppc = self.ppc * (self.num_prd > 0) # zero if either no pp or no wells
        self.turbine_power_output_MWe = 0
        self.turbine_power_generation_MWh = 0
        self.m_market = 0
        self.m_bypass = 0
        self.m_turbine = 0
        self.T_inj = 0
        self.steamtable = XSteam(XSteam.UNIT_SYSTEM_MKS) # m/kg/sec/°C/bar/W
        self.df_records = pd.DataFrame()
        self.pp_type_thresh = 175 # if reservoir temperature is greater, then use a flash power plant
        if not self.power_plant_type:
            self.power_plant_type = "Binary" if self.Tres_init < self.pp_type_thresh else "Flash"
        #make sure battery design is appropriate
        for i in range(len(self.battery_duration)):
            if min(self.battery_duration[i], self.battery_power_capacity[i]) == 0:
                self.battery_duration[i] = 0.0
                self.battery_power_capacity[i] = 0.0
        
        self._reset()

    def step(self,
             timestep,
             data,
             m_prd,
             m_inj,
             m_tes_in=0,
             m_tes_out=0,
             p_bat_ppin=0,
             p_bat_gridin=0,
             p_bat_out=0,
             m_bypass=0):
        
        """Stepping the project in time."""
        
        # Record important quantities
        self.timestep = timestep
        self.timestep_hrs = self.timestep.total_seconds() / 3600
        self.time_curr += timestep
        self.T_amb = data["T0"] #self.weather.amb_temp(self.time_curr) Rather, we will get it externally
        self.m_prd = m_prd
        self.m_inj = m_inj
        self.m_tes_in = m_tes_in
        self.m_tes_out = m_tes_out
        self.p_bat_ppin = p_bat_ppin
        self.p_bat_gridin = p_bat_gridin
        self.p_bat_in = self.p_bat_ppin + self.p_bat_gridin
        self.p_bat_out = p_bat_out
        self.m_bypass = m_bypass
        self.m_g = self.m_prd.sum()
        self.price = data["price"]
        self.price_raw = data["price_raw"]
        self.capacity_price = data["capacity_price"]
        self.recs_price = data["recs_price"]
        self.battery_elcc = data["battery_elcc"] if self.battery else 0.0

        # Step TES, if required
        if self.st:
            self.m_tes_in, self.m_tes_out = self.st.step(timestep, self.T_amb, self.m_tes_in, self.m_tes_out, self.T_prd_wh)
            self.T_tes_out = self.st.Tw
        else:
            self.T_tes_out = 100.0

        if self.num_prd > 0:
            # Mass going to turbine
            self.T_prd_wh = self.reservoir.T_prd_wh.mean()
            self.m_wh_to_turbine = self.m_g - self.m_tes_in
            self.m_turbine = self.m_wh_to_turbine + self.m_tes_out
            self.power_output_MWh_kg = self.pp.compute_power_output(self.T_prd_wh, self.T_amb)

        # Mass used to charge battery
        if self.battery:
            violation = self.battery.step(timestep, self.p_bat_in, self.p_bat_out)
            if violation:
                self.p_bat_out, self.p_bat_in, self.p_bat_ppin, self.p_bat_gridin = 0.0, 0.0, 0.0, 0.0
                
            self.m_battery = min(self.p_bat_ppin / self.power_output_MWh_kg / 3600, self.m_turbine) if (self.num_prd > 0) else 0.0
            self.battery_power_output_MWe = self.battery.roundtrip_eff * self.p_bat_out
            self.battery_power_generation_MWh = self.battery_power_output_MWe * self.timestep_hrs
        else:
            self.m_battery = 0.0
            self.battery_power_output_MWe = 0.0
            self.battery_power_generation_MWh = 0.0
        
        if self.num_prd > 0:
            # Check how much the turbine can send to market and compute m_market accordingly
            ##### COMMENTED OUT #####
            self.m_market = min((self.ppc - self.p_bat_ppin) / self.power_output_MWh_kg / 3600,
                                self.m_turbine - self.m_battery) #kg/s

            self.m_excess = self.m_turbine - self.m_market - self.m_battery
            # Turn on for flexible wellhead control
            # self.m_prd = np.maximum(np.zeros(self.num_prd), self.m_prd - self.m_excess/self.num_prd)
            # self.m_inj = np.maximum(np.zeros(self.num_inj), self.m_inj - self.m_excess/self.num_inj)
            
            #####
            ##### COMMENTED OUT #####
            # self.m_market = (self.ppc - self.p_bat_ppin) / self.power_output_MWh_kg / 3600
            # self.m_turbine = self.market + self.m_battery
            # self.m_wh_to_turbine = 
            # self.m_excess = 0
            
            # Mass used to generate power that is directly sold to market
            # self.m_market = self.m_turbine - self.m_battery
            # Bypass if needed
            if self.bypass and self.price < -np.abs(self.recs_price):
                self.m_bypass += self.m_market
                self.m_market = 0.0
            
            # Calculate powerplant outputs
            _, self.turbine_power_output_MWe, self.turbine_power_generation_MWh, _, self.T_inj = \
                self.pp.power_plant_outputs(timestep, self.m_market, self.m_wh_to_turbine, self.m_tes_out, self.T_prd_wh, self.T_tes_out, self.T_amb)

            # Absolute geothermal capacity potential for capacity revenue calculation
            _, self.effective_ppc, _, _, _ = \
                self.pp.power_plant_outputs(timestep, self.m_g, self.m_g, 0.0, self.T_prd_wh, self.T_tes_out, self.T_amb)

            # Step reservoir
            self.reservoir.step(timestep, self.m_prd, self.m_inj, self.T_inj)
            self.PumpingPower = self.reservoir.PumpingPower_ideal
            self.PumpingPowerInj = self.reservoir.PumpingPowerInj
            self.PumpingPowerProd = self.reservoir.PumpingPowerProd
            
        self.net_power_output_MWe = self.turbine_power_output_MWe + self.battery_power_output_MWe - self.PumpingPower
        self.net_power_generation_MWh = self.turbine_power_generation_MWh + self.battery_power_generation_MWh - self.PumpingPower * self.timestep_hrs
        
        return self.net_power_output_MWe, self.net_power_generation_MWh, self.T_inj

    def record_step(self):
        
        """Recording information about the most recent information in the project."""
        
        self.records["World Time"].append(self.time_curr)
        self.records["Year"].append(self.time_curr.year)
        self.records["Month"].append(self.time_curr.month)
        self.records["Day"].append(self.time_curr.day)
        self.records["Hour"].append(self.time_curr.hour)
        self.records["Minute"].append(self.time_curr.minute)
        self.records["DayOfYear"].append(self.time_curr.dayofyear)
        self.records["Net Power Output [MWe]"].append(self.net_power_output_MWe)
        self.records["Turbine Output [MWe]"].append(self.turbine_power_output_MWe)
        self.records["Battery Output [MWe]"].append(self.battery_power_output_MWe)
        self.records["Net Power Generation [MWhe]"].append(self.net_power_generation_MWh)
        self.records["Atm Temp [deg C]"].append(self.T_amb)
        self.records["LMP [$/MWh]"].append(self.price)
        self.records["Raw LMP [$/MWh]"].append(self.price_raw)
        self.records["RECs Value [$/MWh]"].append(self.recs_price)
        self.records["Capacity Value [$/MW-hour]"].append(self.capacity_price)
        self.records["PP Wholesale Revenue [$MM]"].append(self.price * self.turbine_power_output_MWe / 1e6)
        self.records["Battery Wholesale Net Revenue [$MM]"].append(self.price * (self.battery_power_output_MWe - self.p_bat_gridin) / 1e6)
        self.records["Wholesale Revenue [$MM]"].append(self.records["PP Wholesale Revenue [$MM]"][-1] + self.records["Battery Wholesale Net Revenue [$MM]"][-1])
        self.records["RECs Revenue [$MM]"].append(self.recs_price * (self.turbine_power_output_MWe + self.p_bat_ppin) / 1e6) # RECs revenue comes out of power from turbine to (1) market and (2) battery
        self.records["PP Capacity Revenue [$MM]"].append(self.effective_ppc * self.timestep_hrs * self.capacity_price  / 1e6)
        self.records["Battery Capacity Revenue [$MM]"].append(self.battery_elcc * self.roundtrip_eff * self.battery.power_capacity * self.timestep_hrs * self.capacity_price / 1e6 if self.battery else 0.0)
        self.records["Capacity Revenue [$MM]"].append(self.records["PP Capacity Revenue [$MM]"][-1] + self.records["Battery Capacity Revenue [$MM]"][-1])
        self.records["Revenue [$MM]"].append(self.records["Wholesale Revenue [$MM]"][-1] + self.records["RECs Revenue [$MM]"][-1] + self.records["Capacity Revenue [$MM]"][-1])
        self.records["M_Bypass [kg/s]"].append(self.m_bypass)
        self.records["M_Market [kg/s]"].append(self.m_market)
        self.records["M_Battery [kg/s]"].append(self.m_battery)
        self.records["M_Turbine [kg/s]"].append(self.m_turbine)
        self.records["M_Produced [kg/s]"].append(self.m_prd.sum())
        self.records["M_Injected [kg/s]"].append(self.m_inj.sum())
        self.records["Battery ELCC"].append(self.battery_elcc)

        if self.num_prd > 0:
            self.records["Res Temp [deg C]"].append(self.reservoir.Tres)
            self.records["WH Temp [deg C]"].append(self.T_prd_wh)
            self.records["Inj Temp [deg C]"].append(self.T_inj)
            self.records["Field Production [kg]"].append(self.m_prd.sum() * self.timestep.total_seconds())
            self.records["Pumping Power [MWe]"].append(self.PumpingPower)
            self.records["Production Pumping Power [MWe]"].append(self.PumpingPowerProd)
            self.records["Injection Pumping Power [MWe]"].append(self.PumpingPowerInj)
        if self.st:
            self.records["TES M_in [kg/s]"].append(self.m_tes_in)
            self.records["TES M_out [kg/s]"].append(self.m_tes_out)
            self.records["TES Water Vol [m3]"].append(self.st.Vl)
            self.records["TES Steam Vol [m3]"].append(self.st.Va)
            self.records["TES Temp [deg C]"].append(self.st.Tw)
            self.records["TES Steam Quality"].append(self.st.x)
            self.records["TES Max Discharge [kg/s]"].append(self.st.mass_max_discharge/self.timestep.total_seconds())
        if self.battery:
            self.records["Bat Charge From PP [MWe]"].append(self.p_bat_ppin)
            self.records["Bat Charge From Grid [MWe]"].append(self.p_bat_gridin)
            self.records["Bat Charge [MWe]"].append(self.p_bat_in)
            self.records["Bat Discharge [MWe]"].append(self.roundtrip_eff * self.p_bat_out)
            self.records["SOC [%]"].append(self.battery.SOC)
            self.records["Bat Energy Content [MWh]"].append(self.battery.energy_content)

    def compute_economics(self):
        
        """Compute the project capex/opex economics."""
        
        self.df_records = pd.DataFrame.from_dict(self.records).set_index("World Time")
        self.capex = {}
        self.opex = {}
        self.inflation_factor = (1+self.inflation)**(self.start_year - self.baseline_year)
        
        # Power plant and upstream costs
        self.Cplant = self.pp.compute_cplant(np.max(self.records["WH Temp [deg C]"][0]))
        Cinterconnection = self.powerplant_interconnection * 1e-6 * self.ppc * 1e3
        self.powerplant_capex = self.Cplant
        self.interconnection_capex = Cinterconnection #$MM
        self.capex["Power Plant"] = np.array([self.powerplant_capex] + [0 for _ in range(self.L-1)])
        self.capex["Interconnection"] = np.array([self.interconnection_capex] + [0 for _ in range(self.L-1)])
        
        # Drilling and exploration costs
        # self.CInjwell = (0.2818*self.well_depth**2 + 1275.5213*self.well_depth + 632315.)*1e-6 #$MM based on GEOPHIRES large wellbore
        # self.CInjwell = (0.3021*self.well_depth**2 + 584.9112*self.well_depth + 751368.)*1e-6 #$MM based on GEOPHIRES small wellbore
        # self.CPrdwell = (0.3021*self.well_depth**2 + 584.9112*self.well_depth + 751368.)*1e-6 #$MM based on GEOPHIRES small wellbore
        self.CInjwell = compute_drilling_cost(self.well_depth, self.inj_well_diam)
        self.CPrdwell = compute_drilling_cost(self.well_depth, self.prd_well_diam)
        
        self.expl_cost = (1. + self.CPrdwell*0.6) #MM
        self.capex["Exploration"] = np.array([self.expl_cost] + [0 for _ in range(self.L-1)])
        self.drilling_capex = self.CPrdwell * self.num_prd / self.DSR + self.CInjwell * self.num_inj #$MM
        self.capex["Drilling"] = np.array([self.drilling_capex] + [0 for _ in range(self.L-1)])
        
        # Pumping Cost
        injpumphp = np.stack(self.records["Injection Pumping Power [MWe]"]).sum(axis=1).max()*1341
        numberofinjpumps = np.ceil(injpumphp/2000) #pump can be maximum 2,000 hp
        if numberofinjpumps > 0:
            injpumphpcorrected = injpumphp/numberofinjpumps
            Cpumpsinj = numberofinjpumps*1.5*(1750*(injpumphpcorrected)**0.7)*3*(injpumphpcorrected)**(-0.11)
        else:
            Cpumpsinj = 0.0
        self.injectionpump_capex = Cpumpsinj/1e6 #$MM
        self.capex["Injection Pumps"] = np.array([self.injectionpump_capex] + [0 for _ in range(self.L-1)])
        Cpumping = Cpumpsinj
        
        # Production pumps are considered in ORC cases with low-med enthalpy resources
        # if "binary" in self.power_plant_type.lower():
        prodpumphp = np.stack(self.records["Production Pumping Power [MWe]"]).max(axis=0)*1341 #np.max(PumpingPowerProd)/nprod*1341
        Cpumpsprod = np.sum(1.5*(1750*(prodpumphp)**0.7 + 5750*(prodpumphp)**0.2  + 10000 + self.reservoir.pumpdepth*50*3.281)) #see page 46 in user's manual asusming rental of rig for 1 day.
        self.prodpump_capex = Cpumpsprod/1e6 #$MM
        self.capex["Production Pumps"] = np.array([self.prodpump_capex] + [0 for _ in range(self.L-1)])
        Cpumping += Cpumpsprod
        # else:
        #     self.capex["Production Pumps"] = np.zeros(self.L)
        
        self.injection_stimulation_capex = self.num_inj*2.5 if self.SSR>0.0 else 0.0 #$MM
        self.capex["Injection Stimulation"] = np.array([self.injection_stimulation_capex] + [0 for _ in range(self.L-1)])
        
        # Gathering system and piplines
        self.Cgath = ((self.num_prd+self.num_inj) * 750 * 500 + Cpumping)/1e6 #$MM
        self.capex["Gathering System"] = np.array([self.Cgath] + [0 for _ in range(self.L-1)])
        self.Cpipe = 750/1000*self.pipinglength
        self.capex["Pipelines"] = np.array([self.Cpipe] + [0 for _ in range(self.L-1)])
        # self.opex["wells & Power Plant"] = self.upstream_opex * (1+self.opex_escalation)**np.arange(self.L) * ((self.ppc > 0) + (self.num_prd > 0))/2 #$MM/year escalated (levelized O&M cost including pumping Requirements)
        
        # Upstream OPEX
        self.Claborcorrelation = 1.1*(589.*math.log(self.ppc)-304.)/1e3 #$MM/year
        self.opex["Power Plant"] = np.array(self.L * [1.5/100.*self.Cplant + 0.75*self.Claborcorrelation])
        self.opex["Wellsite"] = np.array(self.L * [1./100.*(self.CPrdwell + self.Cgath) + 0.25*self.Claborcorrelation])
        self.opex["Makeup Water"] = self.df_records.groupby('Year')["Field Production [kg]"].sum().values*self.waterloss/1000 * 264.172 * (0.00613 if "flash" in self.power_plant_type.lower() else 0.00092) /1e6 # (ref: GETEM page 30)
        
        # TES cost
        self.VTank = math.pi/4*self.diameter**2*self.height
        self.tes_capex = (self.tank_capex_rate * self.VTank)
        self.capex["TES"] = np.array([self.tes_capex] + [0 for _ in range(self.L-1)])

        # Lithium-ion battery cost
        df_bat_costs = pd.read_csv(self.battery_costs_filepath)
        df_bat_costs = df_bat_costs[df_bat_costs.Year.isin(range(self.start_year, self.start_year+self.L))].copy()
        self.capex["Battery"] = np.zeros(self.L)
        self.opex["Battery"] = np.zeros(self.L)

        for i, (battery_duration, battery_power_capacity) in enumerate(zip(self.battery_duration, self.battery_power_capacity)):
            # itc = self.itc # if i == 0 else 0.0 # no ITC for battery-2 in the future?
            
            installation_year = min(i * self.battery_lifetime, self.L)
            retirement_year = min((i+1) * self.battery_lifetime, self.L)
            
            #case where L is inputted to be smaller than battery_lifetime, then the second battery does not get purchased
            if installation_year == retirement_year:
                continue
            
            if installation_year < len(df_bat_costs):
                battery_energy_cost, battery_power_cost, battery_interconnection_cost = df_bat_costs["energy cost"].iloc[installation_year], df_bat_costs["power cost"].iloc[installation_year], df_bat_costs["interconnection cost"].iloc[installation_year]
                unit_capex = (battery_energy_cost*battery_duration + battery_power_cost) * battery_power_capacity*1e3 / 1e6
                if i == 0:
                    unit_capex += battery_interconnection_cost * battery_power_capacity * 1e3 / 1e6
                else:
                    additional_interconnection = max(battery_power_capacity - self.battery_power_capacity[i-1], 0)
                    unit_capex += additional_interconnection * battery_power_capacity * 1e3 / 1e6
                capex_ratio = (retirement_year - installation_year) / self.battery_lifetime
                self.capex["Battery"][installation_year] = unit_capex * capex_ratio #$MM
                
                df_temp = df_bat_costs.iloc[installation_year:retirement_year].copy()
                df_temp[["FOM", "power augmentation", "energy augmentation"]] = df_temp[["FOM", "power augmentation", "energy augmentation"]].mul((1+self.opex_escalation)**np.arange(len(df_temp)), axis=0)
                bat_fom = df_temp["FOM"].values * battery_power_capacity*1e3 / 1e6 # $MM/year
                bat_power_aug = df_temp["power augmentation"].values * battery_power_capacity*1e3 / 1e6 # $MM/year
                bat_energy_aug = df_temp["energy augmentation"].values * (battery_power_capacity*battery_duration*1e3) / 1e6 # $MM/year
                self.opex["Battery"][installation_year:retirement_year] = bat_fom + bat_power_aug + bat_energy_aug # $MM/year
        
        self.capex["Contingency"] = 0.15 * np.sum(list(self.capex.values()), axis=0) # Based on GETEM page 21
        
        self.present_capex_per_unit = {}
        for k, v in self.capex.items():
            self.present_capex_per_unit[k] = np.sum(v/(1+self.d)**np.arange(self.L))
            
        self.present_opex_per_unit = {}
        for k, v in self.opex.items():
            self.present_opex_per_unit[k] = np.sum(v/(1+self.d)**np.arange(self.L))

        # Total CAPEX/OPEX
        self.capex_total = np.sum(np.array([v for v in self.capex.values()]), axis=0) * self.inflation_factor * (1 - self.itc)
        self.opex_total = np.sum(np.array([v for v in self.opex.values()]), axis=0) # $MM/year

        self.cashout = self.capex_total + self.opex_total ##
        
    def _reset(self):
        
        """Reseting the project to its initial state."""
        
        self.time_curr = self.time_init
        self.records = defaultdict(list)

        # Create market
        self.market = TabularPowerMarket()
        self.market.create_market(filepath=self.market_filepath,
                                           resample=self.resample, 
                                           fat_factor=self.fat_factor)
        self.market.create_capacity_market(self.capacity_filepath, convert_to_usd_per_mwh=True)
        self.market.create_elcc_forecast(self.battery_costs_filepath)
        
        # Create weather
        self.weather = Weather()
        self.weather.create_weather_model(filepath=self.weather_filepath, resample=self.resample)

        # Create reservoir
        self.reservoir = Subsurface(Tres_init=self.Tres_init, geothermal_gradient=self.geothermal_gradient, surface_temp=self.surface_temp, 
                                    L=self.L, time_init=self.time_init, well_depth=self.well_depth, 
                                    prd_well_diam=self.prd_well_diam, inj_well_diam=self.inj_well_diam, num_prd=self.num_prd, 
                                    num_inj=self.num_inj, waterloss=self.waterloss,
                                    power_plant_type=self.power_plant_type, pumpeff=self.pumpeff, PI=self.PI, II=self.II, SSR=self.SSR)
        self.reservoir.create_percentage_model()

        # Create TES storage tank object, if specified
        self.st = TES(time_init=self.time_init, d=self.diameter, H=self.height) if self.diameter > 0 else None

        # Create Lithium-ion storage object, if specified
        self.battery = LiIonBattery(time_init=self.time_init, duration=self.battery_duration, power_capacity=self.battery_power_capacity, \
                                    roundtrip_eff=self.roundtrip_eff, lifetime=self.battery_lifetime) if max(self.battery_power_capacity) > 0 else None

        # Create power plant
        if "binary" in self.power_plant_type.lower():
            self.pp = ORCPowerPlant(ppc=self.ppc, Tres=self.reservoir.Tres_init, cf=self.cf)
        else:
            self.pp = FlashPowerPlant(ppc=self.ppc, Tres=self.reservoir.Tres_init, cf=self.cf)
        
        # self.compute_economics()
        
    def plot_price_distribution(self):
        
        """Plotting power wholesale market price distribution."""
        
        fig = plt.figure(figsize=(8, 5))
        self.df.price.hist(histtype='step', bins=200)
        plt.xlim([self.df.price.min(), self.df.price.max()])
       
    def config_to_placeholders(self, config):
        
        """Create attributes for all keys and values in a nested configuration dictionary."""
        
        for top_val in config.values():
            for key1, val1 in top_val.items():
                exec("self." + key1 + '=val1')
                if isinstance(val1, dict):
                    for key2, val2 in val1.items():
                        exec("self." + key2 + '=val2')

if __name__ == '__main__':

    world = World()
