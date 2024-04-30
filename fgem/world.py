import math
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import sys
import warnings
from datetime import timedelta, datetime
from fgem.utils.utils import compute_drilling_cost, compute_npv, plot_cols
from fgem.utils.constants import SMALL_NUM, SMALLER_NUM
from fgem.subsurface import *
from fgem.powerplant import *
from fgem.markets import *
from fgem.weather import *
from fgem.storage import *
from pyXSteam.XSteam import XSteam
colors = 24*sns.color_palette()

class World:

    """High-level class to define a project involving upstream, midstream, and downstream components."""
    
    def __init__(self, config):
        """Defining attributes for the World class.

        Args:
            config (dict): json object with project configuration.
        """
        
        # Set default values at first
        self.set_defaults()
        
        # Record major input variables

        self.config = config
        self.config_to_placeholders(self.config)

        self.data_dir = os.path.join(self.base_dir, self.data_dir)
        self.energy_filepath = os.path.join(self.data_dir, self.market_dir, self.energy_market_filename) if self.energy_market_filename else None
        self.capacity_filepath = os.path.join(self.data_dir, self.market_dir, self.capacity_market_filename) if self.capacity_market_filename else None
        self.weather_filepath = os.path.join(self.data_dir, self.market_dir, self.weather_filename) if self.weather_filename else None
        self.battery_costs_filepath = os.path.join(self.data_dir, self.market_dir, self.battery_costs_filename) if isinstance(self.battery_costs_filename, str) else None
        self.time_init = pd.to_datetime(self.time_init) if self.time_init else pd.to_datetime('today')
        self.start_year = self.time_init.year
        self.end_year = self.start_year + self.L
        self.num_inj = max(int(np.floor(self.inj_prd_ratio * self.num_prd)), 1) # underestimate the need for injectors since devlopers would often prefer to drill more later if needed
        self.effective_ppc = self.powerplant_capacity * (self.num_prd > 0) # zero if either no power plant or no wells installed
        self.turbine_power_output_MWe = 0
        self.turbine_power_generation_MWh = 0
        self.m_market = 0
        self.m_bypass = 0
        self.m_turbine = 0
        self.T_inj = 0
        self.T_tes_out = 100.0
        self.m_battery = 0.0
        self.battery_power_output_MWe = 0.0
        self.battery_power_generation_MWh = 0.0
        self.PumpingPower = 0
        self.steamtable = XSteam(XSteam.UNIT_SYSTEM_MKS) # m/kg/sec/°C/bar/W
        self.df_records = pd.DataFrame()
        self.pp_type_thresh = 175 # if reservoir temperature is greater, then use a flash power plant
        self.half_lateral_length = self.lateral_length/2

        if not self.powerplant_type:
            self.powerplant_type = "Binary" if self.Tres_init < self.powerplant_type_thresh else "Flash"
        #make sure battery design is appropriate
        for i in range(len(self.battery_duration)):
            if min(self.battery_duration[i], self.battery_power_capacity[i]) == 0:
                self.battery_duration[i] = 0.0
                self.battery_power_capacity[i] = 0.0
        
        self._reset()

    def step_update_record(self,
                            m_prd=80,
                            m_inj=None,
                            m_tes_in=0,
                            m_tes_out=0,
                            p_bat_ppin=0,
                            p_bat_gridin=0,
                            p_bat_out=0,
                            m_bypass=0,
                            keep_records=True):
        """One function to step, update, and record the project in one go.

        Args:
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s. Defaults to 80.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s. Defaults to None.
            m_tes_in (float, optional): mass flow into thermal energy storage tank in kg/s. Defaults to 0.
            m_tes_out (float, optional): mass flow out of thermal energy storage tank in kg/s. Defaults to 0.
            p_bat_ppin (float, optional): power rate at which battery is charged directly from the geothermal power plant in MW. Defaults to 0.
            p_bat_gridin (float, optional): power rate at which battery is charged directly from the grid in MW. Defaults to 0.
            p_bat_out (float, optional): power rate at which battery is discharged to the grid in MW. Defaults to 0.
            m_bypass (Union[ndarray,float], optional): mass flow rates to be bypassed away from the power plant or turbine in kg/s. Defaults to 0.
            keep_records (bool, optional): whether or not to store records at each simulation timestep. Defaults to True.
        """
                
        self.update_state(m_prd, m_inj, 
                          m_tes_in, m_tes_out, 
                          p_bat_ppin, p_bat_gridin, p_bat_out, 
                          m_bypass)
        self.step()
        self.record_step()

    def step(self):
        """Stepping the project in time.
        """
        self.power_output_MWh_kg = self.powerplant.compute_geofluid_consumption(self.T_prd_wh, self.T_amb)

        # Step TES, if required
        if self.st:
            self.m_tes_in, self.m_tes_out, self.st_violation = self.st.step(self.T_amb, self.m_tes_in, self.m_tes_out, self.T_prd_wh)
            self.T_tes_out = self.st.Tw

        # Mass used to charge battery
        if self.battery:
            self.p_bat_out, self.p_bat_in, self.battery_violation = self.battery.step(self.p_bat_in, self.p_bat_out)
            if self.battery_violation:
                self.p_bat_ppin, self.p_bat_gridin = 0.0, 0.0
            
            self.m_battery = min(self.p_bat_ppin / self.power_output_MWh_kg / 3600, self.m_turbine) if (self.num_prd > 0) else 0.0
            self.battery_power_output_MWe = self.battery.battery_roundtrip_eff * self.p_bat_out
            self.battery_power_generation_MWh = self.battery_power_output_MWe * self.timestep_hrs
        
        if self.num_prd > 0:
            # Check how much the turbine can send to market and compute m_market accordingly
            self.m_market = min((self.powerplant_capacity - self.p_bat_ppin) / nonzero(self.power_output_MWh_kg) / 3600,
                                self.m_turbine - self.m_battery) #kg/s

            self.m_excess = self.m_turbine - self.m_market - self.m_battery
            
            # Bypass if needed
            if self.bypass:
                if self.price < -np.abs(self.recs_price):
                    self.m_bypass += self.m_market
                    self.m_market = 0.0
                elif self.m_excess > SMALL_NUM:
                    self.m_bypass += self.m_excess
                    self.m_excess = 0.0
            
            # Calculate powerplant outputs
            self.powerplant.step(m_turbine=self.m_market, 
                                 T_prd_wh=self.T_prd_wh,
                                 T_amb=self.T_amb,
                                 m_tes_out=self.m_tes_out,
                                 T_tes_out=self.T_tes_out)

            # Correct injection temperauture if you are bypassing:
            if self.bypass and (self.m_bypass > 0):
                self.T_inj = (self.m_turbine * self.T_inj + self.m_bypass * self.T_prd_wh)/(self.m_turbine + self.m_bypass)
            
            # Step reservoir
            self.reservoir.step(m_prd=self.m_prd, 
                                m_inj=self.m_inj,
                                T_inj=self.T_inj,
                                T_amb=self.T_amb)

    def update_state(self, 
                    m_prd=None,
                    m_inj=None,
                    m_tes_in=0,
                    m_tes_out=0,
                    p_bat_ppin=0,
                    p_bat_gridin=0,
                    p_bat_out=0,
                    m_bypass=0,
                    timestep=None):
        """Update the project state.

        Args:
            m_prd (Union[ndarray,float], optional): producer mass flow rates in kg/s. Defaults to 80.
            m_inj (Union[ndarray,float], optional): injector mass flow rates in kg/s. Defaults to None.
            m_tes_in (float, optional): mass flow into thermal energy storage tank in kg/s. Defaults to 0.
            m_tes_out (float, optional): mass flow out of thermal energy storage tank in kg/s. Defaults to 0.
            p_bat_ppin (float, optional): power rate at which battery is charged directly from the geothermal power plant in MW. Defaults to 0.
            p_bat_gridin (float, optional): power rate at which battery is charged directly from the grid in MW. Defaults to 0.
            p_bat_out (float, optional): power rate at which battery is discharged to the grid in MW. Defaults to 0.
            m_bypass (Union[ndarray,float], optional): mass flow rates to be bypassed away from the power plant or turbine in kg/s. Defaults to 0.
            timestep (datetime.timedelta, optional): simulation timestep size. Defaults to None.
        """
        # Update project timestep if indicated by timestep argument
        self.update_timestep_for_all_components(timestep)
        self.time_curr += self.timestep
        self.step_idx += 1
        self.state = self.df_market.iloc[self.step_idx]
        self.T_amb = self.state["T0"]
        self.m_tes_in = m_tes_in
        self.m_tes_out = m_tes_out
        self.p_bat_ppin = p_bat_ppin
        self.p_bat_gridin = p_bat_gridin
        self.p_bat_in = self.p_bat_ppin + self.p_bat_gridin
        self.p_bat_out = p_bat_out
        
        if m_prd is not None:
            self.m_prd = m_prd if isinstance(m_prd, np.ndarray) else np.array(self.num_prd * [m_prd])
        else:
            self.m_prd = self.m_prd if isinstance(self.m_prd, np.ndarray) else np.array(self.num_prd * [self.m_prd])
        
        self.m_g = self.m_prd.sum()

        if m_inj is not None:
            self.m_inj = m_inj if isinstance(m_inj, np.ndarray) else np.array(self.num_inj * [m_inj])
        else:
            self.m_inj = np.array(self.num_inj * [self.m_g/self.num_inj])

        self.m_bypass = m_bypass
        self.m_excess = 0
        self.m_turbine = self.m_g + self.m_tes_out - self.m_tes_in - self.m_bypass
        self.m_market = self.m_turbine
        self.price = self.state["price"]
        self.price_raw = self.state["price_raw"]
        self.capacity_price = self.state["capacity_price"]
        self.recs_price = self.state["recs_price"]
        self.battery_elcc = self.state["battery_elcc"] if self.battery else 0.0
        self.T_prd_wh = (self.reservoir.T_prd_wh * (self.m_prd+SMALL_NUM) / (self.m_prd+SMALL_NUM).sum()).sum() # add small number 0.1 to m_prd to account for the case where all wells are shut-off
        self.turbine_power_output_MWe = self.powerplant.power_output_MWe
        self.turbine_power_generation_MWh = self.powerplant.power_generation_MWh
        self.T_inj = self.powerplant.T_inj

    def record_step(self):
        """Recording information about the most recent information in the project.
        """
        
        self.PumpingPower = self.reservoir.PumpingPower_ideal
        self.PumpingPowerInj = self.reservoir.PumpingPowerInj
        self.PumpingPowerProd = self.reservoir.PumpingPowerProd

        self.net_power_output_MWe = self.turbine_power_output_MWe + self.battery_power_output_MWe - self.PumpingPower
        self.net_power_generation_MWh = self.turbine_power_generation_MWh + self.battery_power_generation_MWh - self.PumpingPower * self.timestep_hrs

        # Absolute geothermal capacity potential for capacity revenue calculation
        self.effective_ppc= self.powerplant.compute_power_output(self.m_g)

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
        self.records["PP Wholesale Revenue [$MM]"].append(self.price * self.turbine_power_output_MWe * self.timestep_hrs / 1e6)
        self.records["Battery Wholesale Net Revenue [$MM]"].append(self.price * (self.battery_power_output_MWe - self.p_bat_gridin) * self.timestep_hrs / 1e6)
        self.records["Wholesale Revenue [$MM]"].append(self.records["PP Wholesale Revenue [$MM]"][-1] + self.records["Battery Wholesale Net Revenue [$MM]"][-1])
        self.records["RECs Revenue [$MM]"].append(self.recs_price * (self.turbine_power_output_MWe + self.p_bat_ppin) * self.timestep_hrs / 1e6) # RECs revenue comes out of power from turbine to (1) market and (2) battery
        self.records["PP Capacity Revenue [$MM]"].append(self.effective_ppc * self.timestep_hrs * self.capacity_price  / 1e6)
        self.records["Battery Capacity Revenue [$MM]"].append(self.battery_elcc * self.battery_roundtrip_eff * self.battery.power_capacity * self.timestep_hrs * self.capacity_price / 1e6 if self.battery else 0.0)
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
            self.records["Producer Wellhead Pressure [bar]"].append(0.01 * self.reservoir.WHP_Prod)
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
            self.records["Bat Discharge [MWe]"].append(self.battery_roundtrip_eff * self.p_bat_out)
            self.records["SOC [%]"].append(self.battery.SOC)
            self.records["Bat Energy Content [MWh]"].append(self.battery.energy_content)

    def compute_economics(self):
        """Compute the project capex/opex economics.
        """
        
        self.df_records = pd.DataFrame.from_dict(self.records).set_index("World Time")
        self.df_records.iloc[0] = self.df_records.iloc[1] # overwrite the zero step as it is based on default params before the simulation actually starts

        # Reduce precision of float64 columns for plotting purposes
        float64_cols = list(self.df_records.select_dtypes(include='float64'))
        self.df_records[float64_cols] = self.df_records[float64_cols].astype('float32')

        self.capex = {}
        self.opex = {}
        
        # Power plant and upstream costs
        self.Cplant = self.powerplant.compute_cplant(np.max(self.records["WH Temp [deg C]"][0]))
        Cinterconnection = self.powerplant_interconnection_cost * 1e-6 * self.powerplant_capacity * 1e3
        self.powerplant_capex = self.Cplant
        self.interconnection_capex = Cinterconnection #$MM
        self.capex["Power Plant"] = np.array([self.powerplant_capex] + [0 for _ in range(self.L-1)])
        self.capex["Interconnection"] = np.array([self.interconnection_capex] + [0 for _ in range(self.L-1)])
        
        # Drilling and exploration costs
        self.CInjwell = compute_drilling_cost(self.well_depth, self.inj_well_diam, self.half_lateral_length if self.reservoir_type == "uloop" else self.lateral_length, 
                                              self.numberoflaterals, self.inj_total_drilling_length, self.drilling_cost)
        self.CPrdwell = compute_drilling_cost(self.well_depth, self.prd_well_diam, self.half_lateral_length if self.reservoir_type == "uloop" else self.lateral_length, 
                                              self.numberoflaterals, self.prd_total_drilling_length, self.drilling_cost)
        
        self.expl_cost = (1. + self.CPrdwell*0.6) #MM
        self.capex["Exploration"] = np.array([self.expl_cost] + [0 for _ in range(self.L-1)])
        self.drilling_cost = self.CPrdwell * self.num_prd / self.DSR + self.CInjwell * self.num_inj #$MM
        self.capex["Drilling"] = np.array([self.drilling_cost] + [0 for _ in range(self.L-1)])
        
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
        prodpumphp = np.stack(self.records["Production Pumping Power [MWe]"]).max(axis=0)*1341 #np.max(PumpingPowerProd)/nprod*1341
        Cpumpsprod = np.sum((1.5*(1750*(prodpumphp)**0.7 + 5750*(prodpumphp)**0.2  + 10000 + self.reservoir.pumpdepth*50*3.281)) * np.where(prodpumphp>0, 1, 0)) #see page 46 in user's manual assuming rental of rig for 1 day.
        self.prodpump_capex = Cpumpsprod/1e6 #$MM
        self.capex["Production Pumps"] = np.array([self.prodpump_capex] + [0 for _ in range(self.L-1)])
        Cpumping += Cpumpsprod
        
        self.injection_stimulation_capex = self.num_inj*2.5 if self.SSR>0.0 else 0.0 #$MM
        self.capex["Injection Stimulation"] = np.array([self.injection_stimulation_capex] + [0 for _ in range(self.L-1)])
        
        # Gathering system and piplines
        self.Cgath = ((self.num_prd+self.num_inj) * 750 * 500)/1e6 #$MM
        self.capex["Gathering System"] = np.array([self.Cgath] + [0 for _ in range(self.L-1)])
        self.Cpipe = (750*self.pipinglength)/1e6
        self.capex["Pipelines"] = np.array([self.Cpipe] + [0 for _ in range(self.L-1)])
        
        # Upstream OPEX
        self.Claborcorrelation = max(1.1*(589.*math.log(self.powerplant_capacity)-304.)/1e3, 0) #$MM/year
        self.opex["Power Plant"] = np.array(self.L * [1.5/100.*self.Cplant + 0.75*self.Claborcorrelation])
        self.opex["Wellsite"] = np.array(self.L * [1./100.*(self.CPrdwell + self.Cgath) + 0.25*self.Claborcorrelation])
        self.opex["Makeup Water"] = self.df_records.groupby('Year')["Field Production [kg]"].sum().values*self.waterloss/1000 * 264.172 * (0.00613 if "flash" in self.powerplant_type.lower() else 0.00092) /1e6 # (ref: GETEM page 30)
        
        # TES cost
        self.VTank = math.pi/4*self.tank_diameter**2*self.tank_height
        self.tes_capex = (self.tank_cost * self.VTank)
        self.capex["TES"] = np.array([self.tes_capex] + [0 for _ in range(self.L-1)])

        # Lithium-ion battery cost
        if self.battery_costs_filepath:
            df_bat_costs = pd.read_csv(self.battery_costs_filepath)
            df_bat_costs = df_bat_costs[df_bat_costs.Year.isin(range(self.start_year, self.start_year+self.L))].copy()

        else:
            df_bat_costs = pd.DataFrame()
            df_bat_costs["Year"] = range(self.start_year, self.start_year+self.L)
            df_bat_costs["elcc"] = self.battery_elcc
            df_bat_costs["energy cost"] = self.battery_energy_cost
            df_bat_costs["power cost"] = self.battery_power_cost
            df_bat_costs["interconnection cost"] = self.battery_interconnection_cost
            df_bat_costs["FOM"] = self.battery_fom
            df_bat_costs["power augmentation"] = self.battery_power_augmentation
            df_bat_costs["energy augmentation"] = self.battery_energy_augmentation
        
        self.capex["Battery"] = np.zeros(self.L)
        self.opex["Battery"] = np.zeros(self.L)

        for i, (battery_duration, battery_power_capacity) in enumerate(zip(self.battery_duration, self.battery_power_capacity)):
            #TODO: itc = self.itc # if i == 0 else 0.0 # no ITC for battery-2 in the future?
            
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
                    #TODO: consider inflating the cost of the second battery unit as it is only installed later in the project lifetime.
                    additional_interconnection = max(battery_power_capacity - self.battery_power_capacity[i-1], 0)
                    unit_capex += additional_interconnection * battery_power_capacity * 1e3 / 1e6
                capex_ratio = (retirement_year - installation_year) / self.battery_lifetime
                self.capex["Battery"][installation_year] = unit_capex * capex_ratio #$MM
                
                df_temp = df_bat_costs.iloc[installation_year:retirement_year].copy()
                bat_fom = df_temp["FOM"].values * battery_power_capacity*1e3 / 1e6 # $MM/year
                bat_power_aug = df_temp["power augmentation"].values * battery_power_capacity*1e3 / 1e6 # $MM/year
                bat_energy_aug = df_temp["energy augmentation"].values * (battery_power_capacity*battery_duration*1e3) / 1e6 # $MM/year
                self.opex["Battery"][installation_year:retirement_year] = bat_fom + bat_power_aug + bat_energy_aug # $MM/year
        
        self.capex["Contingency"] = 0.15 * np.sum(list(self.capex.values()), axis=0) # Based on GETEM page 21
        
        # Make sure they all have shape [0, L] with zero-padding (shorter arrays can happen due to leap year)
        for k in self.opex.keys():
            self.opex[k] = np.pad(self.opex[k], (0, self.L - len(self.opex[k])), 'constant')
        
        for k in self.capex.keys():
            self.capex[k] = np.pad(self.capex[k], (0, self.L - len(self.capex[k])), 'constant')

        self.present_capex_per_unit = {}
        for k, v in self.capex.items():
            self.present_capex_per_unit[k] = np.sum(v/(1+self.d)**np.arange(len(v)))
            
        self.present_opex_per_unit = {}
        for k, v in self.opex.items():
            self.present_opex_per_unit[k] = np.sum(v/(1+self.d)**np.arange(len(v)))

        # Total CAPEX/OPEX
        self.capex_total = np.sum(np.array([v for v in self.capex.values()]), axis=0) * (1 - self.itc)
        self.opex_total = np.sum(np.array([v for v in self.opex.values()]), axis=0) # $MM/year

        self.cashout = self.capex_total + self.opex_total ##
    
    def update_timestep_for_all_components(self, timestep):
        """Update the simulation timestep size across all project components.

        Args:
            timestep (datetime.timedelta): simulation timestep size
        """
        if timestep:
            self.timestep = timestep
            self.timestep_hrs = self.timestep.total_seconds() / 3600
            for component in [self.reservoir, self.powerplant, self.st, self.battery]:
                if component:
                    component.timestep = self.timestep
    def _reset(self):
        """Reseting the project to its initial state.
        """
        
        self.records = defaultdict(list)

        # Create market
        self.market = TabularPowerMarket()
        self.market.create_energy_market(filepath=self.energy_filepath,
                                         resample=self.resample, 
                                         fat_factor=self.fat_factor,
                                         energy_price=self.energy_price,
                                         recs_price=self.recs_price,
                                         L=self.L,
                                         time_init=self.time_init)
        
        self.market.df = self.market.df[(self.market.df.year >= self.start_year) & (self.market.df.year < self.end_year)].copy()

        self.market.create_capacity_market(filepath=self.capacity_filepath,
                                           capacity_price=self.capacity_price,
                                           convert_to_usd_per_mwh=True)

        self.market.create_elcc_forecast(filepath=self.battery_costs_filepath,
                                        battery_elcc=self.battery_elcc,
                                        start_year=self.start_year)

        self.df_market = self.market.df.copy()

        # Create weather
        self.weather = Weather()
        if self.weather_filepath:
            self.weather.create_weather_model(filepath=self.weather_filepath, resample=False)
            self.df_market = pd.merge(self.df_market, 
                                      self.weather.df[['month', 'day', 'hour', "T0"]],
                                      how='left', on=['month', 'day', 'hour'])
        else:
            self.df_market["T0"] = self.ambient_temperature

        self.df_market = self.df_market.bfill()

        # Configure simulation time
        self.step_idx = -1
        self.state = self.df_market.iloc[0]
        self.max_simulation_steps = len(self.df_market)
        self.timestep = self.df_market.TimeDiff.median()
        self.timestep_hrs = self.timestep.total_seconds() / 3600
        self.times_arr = self.df_market.TimeDiff_seconds.values
        self.time_curr = self.time_init - self.timestep

        # Create TES storage tank object, if specified
        self.st = TES(time_init=self.time_init, d=self.tank_diameter, H=self.tank_height) if self.tank_diameter > 0 else None

        # Create Lithium-ion storage object, if specified
        self.battery = LiIonBattery(time_init=self.time_init, duration=self.battery_duration, power_capacity=self.battery_power_capacity, \
                                    battery_roundtrip_eff=self.battery_roundtrip_eff, lifetime=self.battery_lifetime) if max(self.battery_power_capacity) > 0 else None

        # Create reservoir
        self.total_drilling_length, self.prd_total_drilling_length, self.inj_total_drilling_length = None, None, None

        if self.reservoir_type == "energy_decline":
            self.reservoir = EnergyDeclineReservoir(Tres_init=self.Tres_init, Pres_init=self.Pres_init, geothermal_gradient=self.geothermal_gradient, surface_temp=self.surface_temp, 
                                        L=self.L, time_init=self.time_init, well_depth=self.well_depth, 
                                        prd_well_diam=self.prd_well_diam, inj_well_diam=self.inj_well_diam, num_prd=self.num_prd, 
                                        num_inj=self.num_inj, waterloss=self.waterloss,
                                        powerplant_type=self.powerplant_type, pumpeff=self.pumpeff, PI=self.PI, II=self.II, SSR=self.SSR, V_res=self.V_res, phi_res=self.phi_res,
                                        rock_energy_recovery=self.rock_energy_recovery, reservoir_simulator_settings=self.reservoir_simulator_settings, ramey=self.ramey, PumpingModel=self.PumpingModel)
        elif self.reservoir_type == "diffusion_convection":                
            self.reservoir = DiffusionConvection(Tres_init=self.Tres_init, geothermal_gradient=self.geothermal_gradient, surface_temp=self.surface_temp,
                                        L=self.L, time_init=self.time_init, well_depth=self.well_depth, 
                                        prd_well_diam=self.prd_well_diam, inj_well_diam=self.inj_well_diam, num_prd=self.num_prd, 
                                        num_inj=self.num_inj, waterloss=self.waterloss, 
                                        powerplant_type=self.powerplant_type, pumpeff=self.pumpeff, PI=self.PI, II=self.II, SSR=self.SSR, 
                                        V_res=self.V_res, phi_res=self.phi_res, lateral_length=self.lateral_length, res_thickness=self.res_thickness, 
                                        krock=self.krock, reservoir_simulator_settings=self.reservoir_simulator_settings, PumpingModel=self.PumpingModel, ramey=self.ramey)
        elif self.reservoir_type == "uloop":
            self.reservoir = ULoopSBT(Tres_init=self.Tres_init, Pres_init=self.Pres_init, surface_temp=self.surface_temp, geothermal_gradient=self.geothermal_gradient,
                         prd_well_diam=self.prd_well_diam, inj_well_diam=self.inj_well_diam, lateral_diam=self.lateral_diam, 
                         well_depth = self.well_depth, numberoflaterals=self.numberoflaterals, half_lateral_length = self.half_lateral_length,
                         lateral_spacing=self.lateral_spacing, L=self.L, time_init=self.time_init, num_prd=self.num_prd, num_inj=self.num_inj, 
                         waterloss=self.waterloss, powerplant_type=self.powerplant_type, pumpeff=self.pumpeff, PI=self.PI, II=self.II,
                         times_arr=np.linspace(0, self.max_simulation_steps, self.max_simulation_steps), reservoir_simulator_settings=self.reservoir_simulator_settings, PumpingModel=self.PumpingModel,
                         closedloop_design=self.closedloop_design, ramey=self.ramey, dx=self.dx
                        )
            self.total_drilling_length, self.prd_total_drilling_length, self.inj_total_drilling_length = self.reservoir.total_drilling_length, self.reservoir.total_drilling_length/2, self.reservoir.total_drilling_length/2
        else:
            # Default to percentage drawdown model
            self.reservoir = PercentageReservoir(Tres_init=self.Tres_init, geothermal_gradient=self.geothermal_gradient, surface_temp=self.surface_temp, 
                            L=self.L, time_init=self.time_init, well_depth=self.well_depth, 
                            prd_well_diam=self.prd_well_diam, inj_well_diam=self.inj_well_diam, num_prd=self.num_prd, 
                            num_inj=self.num_inj, waterloss=self.waterloss,
                            powerplant_type=self.powerplant_type, pumpeff=self.pumpeff, PI=self.PI, II=self.II, SSR=self.SSR, 
                            drawdp=self.drawdp, plateau_length=self.plateau_length, reservoir_simulator_settings=self.reservoir_simulator_settings, ramey=self.ramey, PumpingModel=self.PumpingModel)
        
        # Create power plant
        if "binary" in self.powerplant_type.lower():
            self.powerplant = ORCPowerPlant(ppc=self.powerplant_capacity, Tres=self.reservoir.Tres_init, cf=self.cf)
        else:
            self.powerplant = FlashPowerPlant(ppc=self.powerplant_capacity, Tres=self.reservoir.Tres_init, cf=self.cf)

        self.update_timestep_for_all_components(self.timestep)

    def plot_price_distribution(self):
        """Plotting power wholesale market price distribution

        Returns:
            matplotlib.figure.Figure: figure
        """
        
        fig = plt.figure(figsize=(8, 5))
        self.df_market.price.hist(histtype='step', bins=200)
        plt.xlim([self.df_market.price.min(), self.df_market.price.max()])

        return fig
       
    def config_to_placeholders(self, config):
        
        """Create attributes for all keys and values in a nested configuration dictionary.

        Args:
            config (dict): project configuration
        """
        
        for top_key, top_val in config.items():
            if isinstance(top_val, dict):
                for key1, val1 in top_val.items():
                    exec("self." + key1 + '=val1')
                    if isinstance(val1, dict):
                        for key2, val2 in val1.items():
                            exec("self." + key2 + '=val2')
            else:
                exec("self." + top_key + '=top_val')

    def plot_economics(self, 
                       figsize=(10, 10),
                       dpi=150, 
                       fontsize=10, 
                       colors=colors):
        """Plot CAPEX and OPEX as pie charts.

        Args:
            figsize (tuple, optional): figure size. Defaults to (10, 10).
            dpi (int, optional): resolution dpi. Defaults to 150.
            fontsize (int, optional): font size. Defaults to 10.
            colors (list, optional): color of pie slices. Defaults to seaborn.color_palette().

        Returns:
            matplotlib.figure.Figure: figure
        """
        if not hasattr(self, "present_capex_per_unit"):
            warnings.warn("Warning: economics are computed based on the latest simulation timestep.")
            self.compute_economics()

        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

        def func(pct, allvals):
            absolute = int(np.round(pct/100.*np.sum(allvals)))
            return "{:.0f}%".format(pct, absolute)

        expenditures = {}

        include = ['Power Plant', 'Interconnection', 'Exploration', 
                'Drilling', 'Injection Stimulation', 'Gathering System', 'Pumps', 'TES', 'Battery']
        costs = {k:v for k,v in self.present_capex_per_unit.items()}
        costs["Pumps"] = costs["Production Pumps"] + costs["Injection Pumps"]
        costs_final = {k:costs[k] for k in include}
        expenditures["CAPEX"] = costs_final

        include = ['Power Plant', 'Wellsite', 'Makeup Water']
        costs = {k:v for k,v in self.present_opex_per_unit.items()}
        costs_final = {k:costs[k] for k in include}
        expenditures["OPEX"] = costs_final

        for i, (title, present_per_unit) in enumerate(expenditures.items()):
            ex = [v for _,v in present_per_unit.items() if v > 0]
            labels = [k for k,v in present_per_unit.items() if v > 0]
            wedges, _, _ = axes[i].pie(x=ex,
                                    pctdistance=0.8,
                                    labels=labels, 
                                    colors=colors, 
                                    autopct=lambda pct: func(pct, ex),
                                    textprops=dict(color="w", weight="bold", fontsize=fontsize))

            axes[i].legend(wedges, labels,
                        fontsize=10,
                    title=title,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.1, 0.0, 1.2),
                        ncols=2)
            
        plt.tight_layout()

        return fig

    def plot_operations(self, span=None):
        """Plot operational parameters.

        Args:
            span (Union[range, list], optional): range of timesteps to plot. Defaults to None.

        Returns:
            matplotlib.figure.Figure: figure
        """

        if not hasattr(self, "df_records"):
            warnings.warn("Warning: economics are computed based on the latest simulation timestep.")
            self.compute_economics()

        qdict = {
            "LMP [$/MWh]": "Electricity Price \n [$/MWh]",
          "Atm Temp [deg C]": "Ambient Temp. \n [$\degree C$]",
          "Res Temp [deg C]": "Reservoir Temp. \n [$\degree C$]",
          'Inj Temp [deg C]': "Intector Temp. \n [$\degree C$]",
          "Net Power Output [MWe]": "Net Generation \n [MWh]",
          'M_Produced [kg/s]': "Field Production \n [kg/s]",
          "Pumping Power [MWe]": "Pumping Power \n [MWe]",
          "WH Temp [deg C]": "Producer\nTemp. [$\degree C$]",
        }

        quantities = list(qdict.keys())
        ylabels = list(qdict.values())

        span = span if span else range(0, self.max_simulation_steps-1)
        fig = plot_cols({" ": self.df_records}, span, quantities, 
                                figsize=(10,12), ylabels=ylabels, legend_loc=False, dpi=100, 
                            formattime=False)
        
        return fig

    def set_defaults(self):
        """Set default parameters that are not specified by the user.
        """

        self.base_dir = "."
        self.data_dir = "data"
        self.market_dir = "market"
        self.time_init = "2025-01-01"
        self.L = 30
        self.d = 0.07
        self.inflation = 0.02
        
        self.drilling_cost = None

        self.battery_costs_filename = None
        self.battery_duration = [0, 0]
        self.battery_power_capacity = [0, 0]
        self.battery_interconnection_cost = 200
        self.battery_energy_cost = 200
        self.battery_power_cost = 300
        self.battery_fom = 10
        self.battery_energy_augmentation = 3
        self.battery_power_augmentation = 0.5
        self.battery_elcc = 1.0
        self.battery_roundtrip_eff = 0.85
        self.battery_lifetime = 15

        self.tank_diameter = 0
        self.tank_height = 0
        self.tank_cost = 0.00143453

        self.powerplant_capacity = 10
        self.pipinglength = 5
        self.powerplant_interconnection_cost = 130
        self.bypass = False

        self.weather_filename = None
        self.ambient_temperature = 20
        self.energy_price = 40
        self.recs_price = 10
        self.capacity_price = 100
        self.energy_market_filename = None
        self.capacity_market_filename = None
        self.fat_factor = 1
        self.resample = "1Y"

        self.reservoir_type = "diffusion_convection"
        self.Tres_init = 225
        self.Pres_init = 40
        self.V_res = 5
        self.phi_res = 0.1
        self.res_thickness = 300
        self.krock = 30
        self.drawdp = 0.005
        self.plateau_length = 3
        self.rock_energy_recovery = 1.0
        self.surface_temp = 20
        self.well_depth = 3000
        self.prd_well_diam = 0.3115
        self.int_well_diam = 0.3115
        self.numberoflaterals = 1
        self.num_prd = 4
        self.inj_prd_ratio = 1.0
        self.waterloss = 0.05
        self.pumpeff = 0.75
        self.DSR = 1.0
        self.SSR = 1.0
        self.PI = 20
        self.II = 20
        self.lateral_diam = 0.3115
        self.lateral_spacing = 100
        self.closedloop_design = "default"
        self.dx = None
        self.ramey = None
        self.reservoir_simulator_settings = {"fast_mode": False, "period": 31536000, 
        "accuracy": 1, "DynamicFluidProperties": True}
        self.geothermal_gradient = (self.Tres_init - self.surface_temp)/self.well_depth * 1000

        self.m_prd = 80

if __name__ == '__main__':
    pass