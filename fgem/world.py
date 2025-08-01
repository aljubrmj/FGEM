import math
import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import sys
import warnings
import numpy_financial as npf
from datetime import timedelta, datetime
from .utils.utils import compute_drilling_cost, plot_cols, FaissKNeighbors
from .utils.constants import SMALL_NUM, SMALLER_NUM, LATLON_CRS, UNIFIED_CRS
from .subsurface import *
from .powerplant import *
from .markets import *
from .weather import *
from .storage import *
from pyXSteam.XSteam import XSteam
from scipy.spatial import cKDTree
from shapely.geometry import Point

parent_path = Path(__file__).parent

colors = 24*sns.color_palette()

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

class World:

    """High-level class to define a project involving upstream, midstream, and downstream components."""
    
    def __init__(self, config, reset_market_weather=True):
        """Defining attributes for the World class.

        Args:
            config (dict): json object with project configuration.
        """
        
        # Set default values at first
        self.set_defaults()
        
        # Record major input variables

        self.config = config
        self.config_to_placeholders(self.config)

        self.reservoir_filepath = os.path.join(self.project_data_dir, self.reservoir_filename) if self.reservoir_filename else None
        self.energy_filepath = os.path.join(self.project_data_dir, self.energy_market_filename) if self.energy_market_filename else None
        self.capacity_filepath = os.path.join(self.project_data_dir, self.capacity_market_filename) if self.capacity_market_filename else None
        self.weather_filepath = os.path.join(self.project_data_dir, self.weather_filename) if self.weather_filename else None
        self.battery_costs_filepath = os.path.join(self.project_data_dir, self.battery_costs_filename) if isinstance(self.battery_costs_filename, str) else None
        self.time_init = pd.to_datetime(self.time_init) if self.time_init else pd.to_datetime('today')
        self.start_year = self.time_init.year
        self.end_year = self.start_year + self.L
        self.num_inj = max(int(np.floor(self.inj_prd_ratio * self.num_prd)), 1) if self.num_prd else None # underestimate the need for injectors since devlopers would often prefer to drill more later if needed
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
        self.Tres_init = self.surface_temp + self.geothermal_gradient/1000 * self.well_tvd
        self.auto_redrill = True if self.redrill_ratio else False
        self.auto_shutoff = True if self.shutoff_ratio else False

        if not hasattr(self, "powerplant_type"):
            self.powerplant_type = "Binary"
        #make sure battery design is appropriate
        for i in range(len(self.battery_duration)):
            if min(self.battery_duration[i], self.battery_power_capacity[i]) == 0:
                self.battery_duration[i] = 0.0
                self.battery_power_capacity[i] = 0.0
        
        # Query interconnection costs based on latlon, if location is specified:
        if (self.project_lat is None) or (self.project_long is None):
            pass
        else:
            self.df_trans = CustomUnpickler(open(os.path.join(parent_path, "data/interconnection", "interconnection_usd_kw.pkl"), 'rb')).load()
            northing_easting = gpd.GeoDataFrame(geometry=[Point(self.project_long, self.project_lat)], crs=LATLON_CRS).to_crs(UNIFIED_CRS).loc[0, "geometry"]
            easting, northing = northing_easting.xy[0][0], northing_easting.xy[1][0]
            loc = [easting, northing]
            A = cKDTree(self.df_trans[["Easting", "Northing"]].values)
            distances, indices = A.query(loc, k=5, workers=4, p=2)
            inv_distances = 1/nonzero(distances)
            inv_distances = inv_distances/inv_distances.sum()
            y_interp = (inv_distances * self.df_trans["interconnection_usd_kw"].values[indices]).sum()
            self.powerplant_interconnection_cost = y_interp
            self.battery_interconnection_cost = y_interp

        self._reset(reset_market_weather)

    def step_update_record(self,
                            m_prd=None,
                            m_inj=None,
                            T_inj=None,
                            T_amb=None,
                            m_tes_in=0,
                            m_tes_out=0,
                            p_bat_ppin=0,
                            p_bat_gridin=0,
                            p_bat_out=0,
                            m_bypass=0,
                            keep_records=True):
        """One function to step, update, and record the project in one go.

        Args:
            m_prd (Union[ndarray,list,float], optional): producer mass flow rates in kg/s. Defaults to 80.
            m_inj (Union[ndarray,list,float], optional): injector mass flow rates in kg/s. Defaults to None.
            m_tes_in (float, optional): mass flow into thermal energy storage tank in kg/s. Defaults to 0.
            m_tes_out (float, optional): mass flow out of thermal energy storage tank in kg/s. Defaults to 0.
            p_bat_ppin (float, optional): power rate at which battery is charged directly from the geothermal power plant in MW. Defaults to 0.
            p_bat_gridin (float, optional): power rate at which battery is charged directly from the grid in MW. Defaults to 0.
            p_bat_out (float, optional): power rate at which battery is discharged to the grid in MW. Defaults to 0.
            m_bypass (Union[ndarray,float], optional): mass flow rates to be bypassed away from the power plant or turbine in kg/s. Defaults to 0.
            keep_records (bool, optional): whether or not to store records at each simulation timestep. Defaults to True.
        """

        if self.reservoir_filename:
            m_prd, m_inj = np.zeros(self.num_res), np.zeros(self.num_res)
            for i, reservoir in enumerate(self.reservoirs):
                reservoir_time_passed_seconds = (reservoir.time_curr - reservoir.time_init).total_seconds()
                m_prd[i] = float(reservoir.m_prd_interpolator(reservoir_time_passed_seconds))
                m_inj[i] = float(reservoir.m_inj_interpolator(reservoir_time_passed_seconds))

        elif m_prd is None:
            m_prd = self.m_prd * np.ones(self.num_res)
            m_inj = self.m_inj * np.ones(self.num_res)

        else:
            if isinstance(m_prd, list):
                assert len(m_inj) == len(m_prd) == self.num_res, f'You should pass exactly {self.num_res} values for production mass flow rate.'
                m_prd = np.array(m_prd)
                m_inj = np.array(m_inj)
            else:
                m_prd = m_prd * np.ones(self.num_res)
                m_inj = m_inj * np.ones(self.num_res)

        self.update_state(m_prd, m_inj, T_inj, T_amb, 
                          m_tes_in, m_tes_out, 
                          p_bat_ppin, p_bat_gridin, p_bat_out, 
                          m_bypass)
        self.step()
        self.record_step()
    
    def update_state(self, 
                    m_prd=None,
                    m_inj=None,
                    T_inj=None,
                    T_amb=None,
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
        self.state = self.df_market.iloc[self.step_idx]
        self.step_idx += 1
        self.timestep = timestep if timestep else self.state["TimeDiff"]
        self.update_timestep_for_all_components(self.timestep)

        self.time_curr += self.timestep
        self.T_amb = T_amb if T_amb else self.state["T0"]
        self.m_tes_in = m_tes_in
        self.m_tes_out = m_tes_out
        self.p_bat_ppin = p_bat_ppin
        self.p_bat_gridin = p_bat_gridin
        self.p_bat_in = self.p_bat_ppin + self.p_bat_gridin
        self.p_bat_out = p_bat_out

        # self.T_prd_wh = self.reservoir.T_prd_wh.mean() # add small number 0.1 to m_prd to account for the case where all wells are shut-off
        self.m_prd = m_prd
        self.m_inj = m_inj
        self.res_shutoffs = np.array([r.shutoff for r in self.reservoirs])
        self.res_bh_temps = np.array([r.Tres for r in self.reservoirs])
        self.res_wh_temps = np.array([r.T_prd_wh for r in self.reservoirs])
        self.C_res = heatcapacitywater(self.res_wh_temps)
        self.res_total_prd = self.m_prd * self.num_res_prd * (1-self.res_shutoffs)
        self.res_total_inj = self.m_inj * self.num_res_inj * (1-self.res_shutoffs)
        self.T_prd_wh = (self.res_total_prd*self.C_res*self.res_wh_temps).sum()/nonzero((self.res_total_prd*self.C_res).sum())
        self.m_g = self.res_total_prd.sum()

        self.m_bypass = m_bypass
        self.m_excess = 0
        self.m_turbine = self.m_g + self.m_tes_out - self.m_tes_in - self.m_bypass
        self.m_market = self.m_turbine
        self.price = self.state["price"]
        self.price_raw = self.state["price_raw"]
        self.capacity_price = self.state["capacity_price"]
        self.recs_price = self.state["recs_price"]
        self.battery_elcc = self.state["battery_elcc"] if self.battery else 0.0
        self.turbine_power_output_MWe = self.powerplant.power_output_MWe
        self.turbine_power_generation_MWh = self.powerplant.power_generation_MWh
        self.T_inj = T_inj if T_inj else self.powerplant.T_inj

    def step(self):
        """Stepping the project in time.
        """
        self.power_output_MWh_kg = self.powerplant.compute_geofluid_consumption(self.T_prd_wh, self.T_amb, self.m_turbine)
        self.market = self.m_turbine

        # Step TES, if required
        if self.st:
            self.m_tes_in, self.m_tes_out, self.st_violation = self.st.step(self.T_amb, self.m_tes_in, self.m_tes_out, self.T_prd_wh)
            self.T_tes_out = self.st.Tw

        # Mass used to charge battery
        if self.battery:
            self.p_bat_out, self.p_bat_in, self.battery_violation = self.battery.step(self.p_bat_in, self.p_bat_out)
            if self.battery_violation:
                self.p_bat_ppin, self.p_bat_gridin = 0.0, 0.0
            
            self.m_battery = min(self.p_bat_ppin / nonzero(self.power_output_MWh_kg) / 3600, self.m_turbine) if (self.num_prd > 0) else 0.0
            self.battery_power_output_MWe = self.battery.battery_roundtrip_eff * self.p_bat_out
            self.battery_power_generation_MWh = self.battery_power_output_MWe * self.timestep_hrs
            self.m_market = min((self.powerplant.powerplant_capacity - self.p_bat_ppin) / nonzero(self.power_output_MWh_kg) / 3600,
                                self.m_turbine - self.m_battery) #kg/s
            
        if self.num_prd > 0:
            # Check how much the turbine can send to market and compute m_market accordingly
            self.m_excess = self.m_turbine - self.m_market - self.m_battery
            
            # Bypass if needed
            if self.bypass:
                if self.price < 0:
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
                self.T_inj = (self.m_turbine * self.T_inj + self.m_bypass * self.T_prd_wh)/nonzero(self.m_turbine + self.m_bypass)
            
            # Step reservoir
            for i, reservoir in enumerate(self.reservoirs):
                reservoir.step(m_prd=self.m_prd[i], 
                               m_inj=self.m_inj[i],
                               T_inj=self.T_inj,
                               T_amb=self.T_amb)
                if self.auto_shutoff:
                    if self.shutoff_ratio >= reservoir.Tres / reservoir.Tres_init:
                        self.shutoff_wells(reservoir)

            if self.auto_redrill:
                # time_passed_years = (self.time_curr - self.time_init).total_seconds()/8760/3600
                # reservoir_time_passed_years = (self.reservoir[-1].time_curr - self.reservoir.time_init).total_seconds()/8760/3600
                if (self.redrill_ratio >= self.powerplant.power_output_MWe / self.powerplant.powerplant_capacity):# & (time_passed_years<self.L-3):
                    self.shutoff_wells(self.reservoirs[-1])
                    self.redrill()

    def shutoff_wells(self, reservoir, num_prd=None, num_inj=None):
        if not reservoir.shutoff:
            num_prd = min(reservoir.num_prd if num_prd is None else num_prd, reservoir.num_prd)
            num_inj = min(reservoir.num_inj if num_inj is None else num_inj, reservoir.num_inj)
            reservoir.num_prd -= num_prd
            reservoir.num_inj -= num_inj
            self.num_prd_shutoff[self.time_curr.year - self.time_init.year] -= num_prd
            self.num_inj_shutoff[self.time_curr.year - self.time_init.year] -= num_inj
            self.num_res_prd = np.array([r.num_prd for r in self.reservoirs])
            self.num_res_inj = np.array([r.num_inj for r in self.reservoirs])

        if reservoir.num_prd == 0:
            reservoir.shutoff = True

    def redrill(self, num_prd=None, num_inj=None):
        num_prd = self.num_prd if num_prd is None else num_prd
        num_inj = self.num_inj if num_inj is None else num_inj

        new_reservoir = self.create_reservoir(num_prd, num_inj, self.time_curr)
        # new_reservoir.time_init = self.time_curr
        len_history = len(self.reservoirs[0].m_prd_arr)
        new_reservoir.time_passed_arr = len_history * [0.0]
        new_reservoir.m_prd_arr = len_history * [0.0]
        new_reservoir.m_inj_arr = len_history * [0.0]
        new_reservoir.T_prd_wh_arr = len_history * [np.nan]
        new_reservoir.T_res_arr = len_history * [np.nan]
        new_reservoir.T_inj_arr = len_history * [np.nan]
        new_reservoir.num_prd_arr = len_history * [0.0]
        new_reservoir.num_inj_arr = len_history * [0.0]
        new_reservoir.shutoff_arr = len_history * [True]
        self.reservoirs.append(new_reservoir)
        self.num_prd_drilled[self.time_curr.year - self.time_init.year] += new_reservoir.num_prd
        self.num_inj_drilled[self.time_curr.year - self.time_init.year] += new_reservoir.num_inj

        self.num_res = len(self.reservoirs)
        self.num_res_prd = np.array([r.num_prd for r in self.reservoirs])
        self.num_res_inj = np.array([r.num_inj for r in self.reservoirs])

    def record_step(self):
        """Recording information about the most recent information in the project.
        """

        # Absolute geothermal capacity potential for capacity revenue calculation
        self.effective_ppc= self.powerplant.powerplant_capacity #self.powerplant.compute_power_output(self.m_g)
        self.turbine_power_output_MWe = self.powerplant.power_output_MWe
        self.turbine_power_generation_MWh = self.powerplant.power_generation_MWh

        self.records["Time Passed"].append((self.time_curr - self.time_init).total_seconds())
        # self.records["Reservoir Time Passed"].append((self.reservoir.time_curr - self.reservoir.time_init).total_seconds())
        self.records["World Time"].append(self.time_curr)
        self.records["Year"].append(self.time_curr.year)
        self.records["Month"].append(self.time_curr.month)
        self.records["Day"].append(self.time_curr.day)
        self.records["Hour"].append(self.time_curr.hour)
        self.records["Minute"].append(self.time_curr.minute)
        self.records["DayOfYear"].append(self.time_curr.dayofyear)
        self.records["Specific Power Output [kWh/kg]"].append(self.powerplant.power_output_MWh_kg*1e3)
        self.records["Installed Power Plant Capacity [MWe]"].append(self.powerplant.powerplant_capacity)
        self.records["Turbine Output [MWe]"].append(self.turbine_power_output_MWe)
        self.records["Battery Output [MWe]"].append(self.battery_power_output_MWe)
        self.records["Atm Temp [deg C]"].append(self.T_amb)
        self.records["LMP [$/MWh]"].append(self.price)
        self.records["Raw LMP [$/MWh]"].append(self.price_raw)
        self.records["RECs Value [$/MWh]"].append(self.recs_price)
        self.records["Capacity Value [$/MW-hour]"].append(self.capacity_price)
        self.records["M_Bypass [kg/s]"].append(self.m_bypass)
        self.records["M_Market [kg/s]"].append(self.m_market)
        self.records["M_Battery [kg/s]"].append(self.m_battery)
        self.records["M_Turbine [kg/s]"].append(self.m_turbine)
        self.records["M_Produced [kg/s]"].append(self.res_total_prd.sum())
        self.records["M_Injected [kg/s]"].append(self.res_total_inj.sum())
        self.records["Battery Power Capacity [MWe]"].append(0.0)
        self.records["Battery ELCC"].append(self.battery_elcc)
        self.records["Bat Charge From PP [MWe]"].append(self.p_bat_ppin)
        self.records["Bat Charge From Grid [MWe]"].append(self.p_bat_gridin)
        self.records["Bat Charge [MWe]"].append(self.p_bat_in)
        self.records["Bat Discharge [MWe]"].append(self.battery_roundtrip_eff * self.p_bat_out)

        if self.num_prd > 0:
            self.records["Res Temp [deg C]"].append(self.res_bh_temps.mean())
            self.records["WH Temp [deg C]"].append(self.T_prd_wh)
            self.records["Inj Temp [deg C]"].append(self.T_inj)
            self.records["Field Production [kg]"].append(self.m_g * self.timestep.total_seconds())
        if self.st:
            self.records["TES M_in [kg/s]"].append(self.m_tes_in)
            self.records["TES M_out [kg/s]"].append(self.m_tes_out)
            self.records["TES Water Vol [m3]"].append(self.st.Vl)
            self.records["TES Steam Vol [m3]"].append(self.st.Va)
            self.records["TES Temp [deg C]"].append(self.st.Tw)
            self.records["TES Steam Quality"].append(self.st.x)
            self.records["TES Max Discharge [kg/s]"].append(self.st.mass_max_discharge/self.timestep.total_seconds())
        if self.battery:
            self.records["SOC [%]"].append(self.battery.SOC)
            self.records["Bat Energy Content [MWh]"].append(self.battery.energy_content)
            self.records["Battery Power Capacity [MWe]"][-1] = self.battery.power_capacity

    def compute_pumping(self):
        self.res_PumpingPowerInj = []
        self.res_PumpingPowerProd = []
        self.res_PumpingPower_ideal = []
        self.res_WHP_Prod = []

        for reservoir in self.reservoirs:
            self.well_tvd = reservoir.well_tvd
            self.well_md = reservoir.well_md
            m_prd = np.array(reservoir.m_prd_arr)
            m_inj = np.array(reservoir.m_inj_arr)
            T_prd_wh = np.array(reservoir.T_prd_wh_arr) 
            Tres = np.array(reservoir.T_res_arr) 
            T_inj = np.array(reservoir.T_inj_arr)
            num_prd = np.array(reservoir.num_prd_arr)
            num_inj = np.array(reservoir.num_inj_arr)

            if self.reservoir_filename:
                time_passed_seconds = reservoir.time_passed_arr
                shutoff_arr = np.array(reservoir.shutoff_arr)
                self.WHP_Prod = reservoir.WHP_prd_interpolator(time_passed_seconds) * (1-shutoff_arr)
                self.WHP_Inj = reservoir.WHP_inj_interpolator(time_passed_seconds) * (1-shutoff_arr)
                self.PumpingPowerProd = reservoir.pumping_prd_interpolator(time_passed_seconds)*num_prd * (1-shutoff_arr)
                self.PumpingPowerInj = reservoir.pumping_inj_interpolator(time_passed_seconds)*num_inj * (1-shutoff_arr)
                self.PumpingPower_ideal = np.maximum(self.PumpingPowerInj + self.PumpingPowerProd, 0.0)
                self.pumpdepth = np.array([0.0]) #no production pump at any depth

            elif "uloop" in self.reservoir_type.lower():
                T_inj = T_inj[:,None]
                diam = 2*self.reservoir.radiusvector[None]
                m = m_inj[:,None]
                dL = reservoir.dL[None]
                dz = reservoir.dz[None]
                T = reservoir.TwMatrix[:len(m)]

                rho = densitywater(T)
                mu = viscositywater(T)

                v = (m/reservoir.numberoflaterals)*(1.+reservoir.waterloss)/rho/(math.pi/4.*diam**2)
                Re = 4.*(m/reservoir.numberoflaterals)*(1.+reservoir.waterloss)/(mu*math.pi*diam)
                f = compute_f(Re, diam)

                # Necessary injection wellhead pressure [kPa]
                self.DPSurfaceplant = 68.95
                self.Pprodwellhead = 0.0

                # pressure drop in pipes in parallel is the same, so we average things out
                self.DP_flow = f*(rho*v**2/2)*(dL/diam)/1e3
                self.DP_flow = self.DP_flow[:,:reservoir.interconnections[1]+1].sum(axis=1, keepdims=True) + self.DP_flow[:,reservoir.interconnections[1]+1:].sum(axis=1, keepdims=True)/reservoir.numberoflaterals

                # hydrsotatic is counted once along depth, so we make sure we do not double count hydrostatic pressure build-up from different laterals
                self.DP_hydro = rho*9.81*dz/1e3
                self.DP_hydro = self.DP_hydro[:,:reservoir.interconnections[1]+1].sum(axis=1, keepdims=True) + self.DP_hydro[:,reservoir.interconnections[1]+1:].sum(axis=1, keepdims=True)/reservoir.numberoflaterals

                self.DP = self.Pprodwellhead + self.DP_flow + self.DP_hydro - self.DPSurfaceplant

                self.PumpingPowerInj = (self.DP*m/densitywater(T_inj)/self.pumpeff/1e3).squeeze()
                self.WHP_Prod = -self.DP.squeeze()
                self.PumpingPowerProd = np.zeros_like(self.WHP_Prod) # no pumps at producers in closed loop designs

                # Total pumping power
                self.PumpingPower_ideal = np.maximum(self.PumpingPowerInj + self.PumpingPowerProd, 0.0)
                self.pumpdepth = np.array([0.0]) #no production pump at any depth
            
            elif "coaxial" in self.reservoir_type.lower():
                if self.coaxialflowtype == 1: #CXA
                    self.dh_down = 2 * (reservoir.radius - reservoir.outerradiuscenterpipe) * np.ones(reservoir.N)[None] #hydraulid diameter: https://www.engineeringtoolbox.com/hydraulic-equivalent-diameter-d_458.html
                    self.dh_up = 2 * reservoir.radiuscenterpipe * np.ones(reservoir.N)[None] #hydraulid diameter: https://www.engineeringtoolbox.com/hydraulic-equivalent-diameter-d_458.html
                    self.A_down = math.pi * (reservoir.radius**2 - reservoir.outerradiuscenterpipe**2) * np.ones(reservoir.N)[None]
                    self.A_up = math.pi * reservoir.radiuscenterpipe**2 * np.ones(reservoir.N)[None]

                elif  self.coaxialflowtype == 2: #CXC
                    self.dh_up = 2 * (reservoir.radius - reservoir.outerradiuscenterpipe) * np.ones(reservoir.N)[None] #hydraulid diameter: https://www.engineeringtoolbox.com/hydraulic-equivalent-diameter-d_458.html
                    self.dh_down = 2 * reservoir.radiuscenterpipe * np.ones(reservoir.N)[None] #hydraulid diameter: https://www.engineeringtoolbox.com/hydraulic-equivalent-diameter-d_458.html
                    self.A_up = math.pi * (reservoir.radius**2 - reservoir.outerradiuscenterpipe**2) * np.ones(reservoir.N)[None]
                    self.A_down = math.pi * reservoir.radiuscenterpipe**2 * np.ones(reservoir.N)[None]

                self.dh = np.hstack((self.dh_down, self.dh_up)).reshape(1,-1)
                self.A = np.hstack((self.A_down, self.A_up)).reshape(1,-1)
                self.T = np.hstack((reservoir.Tw_down_Matrix, reservoir.Tw_up_Matrix))
                self.m = reservoir.mvector[:, None]
                self.dz = np.vstack((np.abs(np.diff(reservoir.z, axis=0)), -np.abs(np.diff(reservoir.z, axis=0)))).reshape(1, -1)
                self.dL = np.abs(self.dz)

                self.rho = densitywater(self.T)
                self.mu = viscositywater(self.T)

                self.v = (self.m/self.rho)/self.A # veclocity [m/s]
                self.Re = self.rho * self.v * self.dh / self.mu # Renolds number []: https://www.engineeringtoolbox.com/reynolds-number-d_237.html
                self.f = compute_f(self.Re, self.dh)

                # Necessary injection wellhead pressure [kPa]
                self.DPSurfaceplant = 68.95

                # pressure drop in pipes in parallel is the same, so we average things out
                self.DP_flow = (self.f * (self.dL/self.dh) * (self.rho * self.v**2/2) / 1e3).sum(axis=1, keepdims=True) # flow frictional losses [kPa]

                # hydrsotatic is counted once along depth, so we make sure we do not double count hydrostatic pressure build-up from different laterals
                self.DP_hydro = (self.rho * 9.81 * self.dz / 1e3).sum(axis=1, keepdims=True) # hydrostatic pressure change [kPa]

                self.DP = self.DP_flow - self.DP_hydro - self.DPSurfaceplant

                self.PumpingPowerInj = (self.DP*(self.m/self.rho[:,[0]])/self.pumpeff/1e3).squeeze() # pumping power [Mwe]
                self.WHP_Prod = -self.DP.squeeze()
                self.PumpingPowerProd = np.zeros_like(self.WHP_Prod) # no pumps at producers in closed loop designs

                # Total pumping power
                self.PumpingPower_ideal = np.maximum(self.PumpingPowerInj + self.PumpingPowerProd, 0.0)
                self.pumpdepth = np.array([0.0]) #no production pump at any depth

            else:
                Tavg = (3/4*Tres+1/4*T_prd_wh).values #most of temperature drop happens in upper section (because surrounding rock temperature is lowest in upper section)
                self.rhowaterprod = np.array([self.steamtable.rho_pt(reservoir.Phydrostatic/100/2, t) for t in Tavg]) #densitywater(Tavg)
                muwaterprod = np.array([self.steamtable.my_pt(reservoir.Phydrostatic/100/2, t) for t in Tavg]) #viscositywater(Tavg)
                self.vprod = (m_prd/reservoir.numberoflaterals)/self.rhowaterprod/(math.pi/4.*reservoir.prd_well_diam**2)
                self.Rewaterprod = 4.*(m_prd/reservoir.numberoflaterals)/(muwaterprod*math.pi*reservoir.prd_well_diam) #laminar or turbulent flow?
                self.f3 = compute_f(self.Rewaterprod, reservoir.prd_well_diam)

                Tavg = T_inj
                self.rhowaterinj = np.array([self.steamtable.rho_pt(reservoir.Phydrostatic/100/2, t) for t in Tavg])  #densitywater(Tavg)
                muwaterinj = np.array([self.steamtable.my_pt(reservoir.Phydrostatic/100/2, t) for t in Tavg]) #viscositywater(Tavg)
                self.vinj = (m_inj/reservoir.numberoflaterals)*(1.+self.waterloss)/self.rhowaterinj/(math.pi/4.*reservoir.inj_well_diam**2)
                self.Rewaterinj = 4.*(m_inj/reservoir.numberoflaterals)*(1.+self.waterloss)/(muwaterinj*math.pi*reservoir.inj_well_diam) #laminar or turbulent flow?
                self.f1 = compute_f(self.Rewaterinj, reservoir.inj_well_diam)

                #reservoir hydrostatic pressure [kPa] 
                self.Phydrostatic = reservoir.Phydrostatic

                # ORC power plant case, with pumps at both injectors and producers
                Pexcess = 344.7 #[kPa] = 50 psi. Excess pressure covers non-condensable gas pressure and net positive suction head for the pump
                self.Pprodwellhead = vaporpressurewater(T_prd_wh) + Pexcess #[kPa] is minimum production pump inlet pressure and minimum wellhead pressure
                # Following tip from CLGWG where operational settings allow for no vapor pressure to form:
                self.PIkPa = reservoir.PI/(self.rhowaterprod/1000)/100 #convert PI from l/s/bar to kg/s/kPa

                self.pumpdepth = reservoir.well_tvd + (self.Pprodwellhead - self.Phydrostatic + m_prd/self.PIkPa)/(self.f3*(self.rhowaterprod*self.vprod**2/2.)*(1/reservoir.prd_well_diam)/1E3 + self.rhowaterprod*9.81/1E3)
                self.pumpdepth = np.clip(self.pumpdepth, 0, np.inf)
                self.pumpdepthfinal = np.max(self.pumpdepth)
                if self.pumpdepthfinal <= 0:
                    self.pumpdepthfinal = 0
                elif self.pumpdepthfinal > 600:
                    print("Warning: FGEM calculates pump depth to be deeper than 600 m. Verify reservoir pressure, production well flow rate and production well dimensions")

                # Calculate production well pumping pressure [kPa]
                self.DP3 = self.Pprodwellhead - (self.Phydrostatic - m_prd/self.PIkPa - self.rhowaterprod*9.81*reservoir.well_tvd/1E3 - self.f3*(self.rhowaterprod*self.vprod**2/2.)*(reservoir.well_md/reservoir.prd_well_diam)/1E3)
                PumpingPowerProd = self.DP3*m_prd*num_prd/self.rhowaterprod/self.pumpeff/1e3 #[MWe] total pumping power for production wells
                self.PumpingPowerProd = PumpingPowerProd
                
                self.IIkPa = reservoir.II/(self.rhowaterinj/1000)/100 #convert II from l/s/bar to kg/s/kPa
                
                # Necessary injection wellhead pressure [kPa]
                self.Pinjwellhead = self.Phydrostatic + m_inj*(1+self.waterloss)/self.IIkPa - self.rhowaterinj*9.81*self.well_tvd/1E3 + self.f1*(self.rhowaterinj*self.vinj**2/2)*(reservoir.well_md/reservoir.inj_well_diam)/1e3

                # Plant outlet pressure [kPa]
                self.DPSurfaceplant = 68.95 #[kPa] assumes 10 psi pressure drop in surface equipment
                self.Pplantoutlet = self.Pprodwellhead - self.DPSurfaceplant

                # Injection pump pressure [kPa]
                self.DP1 = self.Pinjwellhead-self.Pplantoutlet
                PumpingPowerInj = self.DP1*m_inj*num_inj/self.rhowaterinj/self.pumpeff/1e3 #[MWe] total pumping power for injection wells

                self.PumpingPowerInj = PumpingPowerInj
                
                self.WHP_Prod = -self.DP3/100 #bar: net pressure out of well before pumping
                # Total pumping power
                self.PumpingPower_ideal = self.PumpingPowerInj + self.PumpingPowerProd
        
            self.res_PumpingPowerInj.append(self.PumpingPowerInj)
            self.res_PumpingPowerProd.append(self.PumpingPowerProd)
            self.res_PumpingPower_ideal.append(self.PumpingPower_ideal)
            self.res_WHP_Prod.append(self.WHP_Prod)

    def compute_economics(self, print_outputs=True):
        """Compute the project capex/opex economics.
            Args:
            print_outputs (bool, optional): whether or not to print final economic parameters.
        """

        # Compute Revenues

        self.df_records["PP Wholesale Revenue [$MM]"] = self.df_records["LMP [$/MWh]"] * np.maximum(self.df_records['Turbine Output [MWe]'] - self.df_records['Pumping Power [MWe]'], 0.0) * self.timestep_hrs / 1e6
        self.df_records["Battery Wholesale Net Revenue [$MM]"] = self.df_records["LMP [$/MWh]"] * (self.df_records["Battery Output [MWe]"] - self.df_records["Bat Charge From Grid [MWe]"]) * self.timestep_hrs / 1e6
        self.df_records["Wholesale Revenue [$MM]"] = self.df_records["PP Wholesale Revenue [$MM]"] + self.df_records["Battery Wholesale Net Revenue [$MM]"]
        self.df_records["RECs Revenue [$MM]"] = self.df_records["RECs Value [$/MWh]"] * (np.maximum(self.df_records["Turbine Output [MWe]"] - self.df_records['Pumping Power [MWe]'],0.0) + self.df_records["Bat Charge From PP [MWe]"]) * self.timestep_hrs / 1e6 # RECs revenue comes out of power from turbine to (1) market and (2) battery
        self.df_records["PP Capacity Revenue [$MM]"] = self.effective_ppc * self.timestep_hrs * self.df_records["Capacity Value [$/MW-hour]"]  / 1e6
        self.df_records["Battery Capacity Revenue [$MM]"] = self.df_records["Battery ELCC"] * self.battery_roundtrip_eff * self.df_records["Battery Power Capacity [MWe]"] * self.timestep_hrs * self.df_records["Capacity Value [$/MW-hour]"] / 1e6 if self.battery else 0.0
        self.df_records["Capacity Revenue [$MM]"] = self.df_records["PP Capacity Revenue [$MM]"] + self.df_records["Battery Capacity Revenue [$MM]"]
        self.df_records["Revenue [$MM]"] = self.df_records["Wholesale Revenue [$MM]"] + self.df_records["RECs Revenue [$MM]"] + self.df_records["Capacity Revenue [$MM]"]

        # Daily price fattening
        self.fat_factors = [1.5, 2.0]
        for fat_factor in self.fat_factors:
            self.df_records[f"Wholesale Revenue (factor {fat_factor}) [$MM]"] = self.df_records["Wholesale Revenue [$MM]"]
            fat_window = int(max(24/self.timestep_hrs, 1))
            means = np.zeros(len(self.df_records))
            for i in range(int(len(means)/fat_window)):
                means[fat_window*i:fat_window*i+fat_window] = \
                    self.df_records["Wholesale Revenue [$MM]"][fat_window*i:fat_window*i+fat_window].mean()

            self.df_records[f"Wholesale Revenue (factor {fat_factor}) [$MM]"] += fat_factor * (self.df_records["Wholesale Revenue [$MM]"] - means)

            fat_window = int(max(8760/self.timestep_hrs, 1))
            means = np.zeros(len(self.df_records))
            for i in range(int(len(means)/fat_window)):
                means[fat_window*i:fat_window*i+fat_window] = \
                    self.df_records["Wholesale Revenue [$MM]"][fat_window*i:fat_window*i+fat_window].mean()

            self.df_records[f"Wholesale Revenue (factor {fat_factor}) [$MM]"] += fat_factor * (self.df_records["Wholesale Revenue [$MM]"] - means)
            # self.df_records[f"Wholesale Revenue (factor {fat_factor}) [$MM]"] = np.maximum(self.df_records[f"Wholesale Revenue (factor {fat_factor}) [$MM]"], 0.0)
            self.df_records[f"Revenue (factor {fat_factor}) [$MM]"] = self.df_records[f"Wholesale Revenue (factor {fat_factor}) [$MM]"] + self.df_records["RECs Revenue [$MM]"] + self.df_records["Capacity Revenue [$MM]"]

        # total active wells for opex calculations
        self.num_prd_active = (self.num_prd_drilled + self.num_prd_shutoff).cumsum()
        self.num_inj_active = (self.num_inj_drilled + self.num_inj_shutoff).cumsum()
        self.num_wells_active = self.num_prd_active + self.num_inj_active

        # Reduce precision of float64 columns for plotting purposes
        float64_cols = list(self.df_records.select_dtypes(include='float64'))
        self.df_records[float64_cols] = self.df_records[float64_cols].astype('float32')

        self.capex = {}
        self.opex = {}
        
        # Power plant and upstream costs
        self.Cplant = self.powerplant.compute_cplant(np.median(self.records["WH Temp [deg C]"]), min_cost=self.powerplant_usd_per_kw_min)
        Cinterconnection = self.powerplant_interconnection_cost * 1e-6 * self.powerplant.powerplant_capacity * 1e3
        self.powerplant_capex = self.Cplant
        self.interconnection_capex = Cinterconnection #$MM
        self.capex["Power Plant"] = np.array([self.powerplant_capex] + [0 for _ in range(self.L-1)])
        self.capex["Interconnection"] = np.array([self.interconnection_capex] + [0 for _ in range(self.L-1)])
        
        # Drilling and exploration costs
        self.CInjwell = compute_drilling_cost(self.well_tvd, self.inj_well_diam, self.half_lateral_length if self.reservoir_type == "uloop" else self.lateral_length, 
                                              self.numberoflaterals, self.inj_total_drilling_length, self.drilling_cost)
        self.CPrdwell = compute_drilling_cost(self.well_tvd, self.prd_well_diam, self.half_lateral_length if self.reservoir_type == "uloop" else self.lateral_length, 
                                              self.numberoflaterals, self.prd_total_drilling_length, self.drilling_cost)
        
        self.expl_cost = (self.exploration_cost_intercept + self.exploration_cost_slope * self.CPrdwell) #MM
        self.capex["Exploration"] = np.array([self.expl_cost] + [0 for _ in range(self.L-1)])
        # self.total_drilling_cost = (self.CPrdwell * self.num_prd + self.CInjwell * self.num_inj)/self.DSR #$MM
        # self.capex["Drilling"] = self.redrill_01 * self.total_drilling_cost
        # self.capex["Drilling"] = np.array([self.total_drilling_cost] + [0 for _ in range(self.L-1)])
        self.capex["Drilling"] = (self.num_prd_drilled * self.CPrdwell + self.num_inj_drilled * self.CInjwell) / self.DSR
        self.total_drilling_cost = self.capex["Drilling"].sum()
        
        # Pumping Cost
        injpumphp = np.stack(self.df_records["Injection Pumping Power [MWe]"].values).max()*1341
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
        prodpumphp = np.stack(self.df_records["Production Pumping Power [MWe]"].values).max()*1341 #np.max(PumpingPowerProd)/nprod*1341
        Cpumpsprod = np.sum((1.5*(1750*(prodpumphp)**0.7 + 5750*(prodpumphp)**0.2  + 10000 + self.pumpdepth.max()*50*3.281)) * np.where(prodpumphp>0, 1, 0)) #see page 46 in user's manual assuming rental of rig for 1 day.
        self.prodpump_capex = Cpumpsprod/1e6 #$MM
        self.capex["Production Pumps"] = np.array([self.prodpump_capex] + [0 for _ in range(self.L-1)])
        Cpumping += Cpumpsprod
        
        # self.stimulation_capex = (self.num_inj+self.num_prd)*self.stimulation_cost if self.SSR>0.0 else 0.0 #$MM
        # self.capex["Stimulation"] = self.redrill_01 * self.stimulation_capex
        # self.capex["Stimulation"] = np.array([self.stimulation_capex] + [0 for _ in range(self.L-1)])
        self.capex["Stimulation"] = (self.num_prd_drilled+self.num_inj_drilled)*self.stimulation_cost if self.SSR>0.0 else 0.0 #$MM
        self.stimulation_capex = self.capex["Stimulation"].sum()

        
        # Gathering system and piplines
        self.Cgath = ((self.num_prd+self.num_inj) * 750 * 500)/1e6 #$MM
        self.capex["Gathering System"] = np.array([self.Cgath] + [0 for _ in range(self.L-1)])
        self.Cpipe = (750*1e3*self.pipinglength)/1e6
        self.capex["Pipelines"] = np.array([self.Cpipe] + [0 for _ in range(self.L-1)])
        
        # Upstream OPEX
        wells_are_active_01 = np.where(self.num_wells_active, 1, 0)
        self.Claborcorrelation = max(1.1*(589.*math.log(self.powerplant.powerplant_capacity)-304.)/1e3, 0) #$MM/year
        self.opex["Power Plant"] = np.array(self.L * [self.powerplant_opex_rate*self.Cplant + self.powerplant_labor_rate*self.Claborcorrelation]) * wells_are_active_01
        # self.opex["Wellsite"] = np.array(self.L * [self.wellsite_opex_rate*(self.CPrdwell + self.Cgath) + self.wellsite_labor_rate*self.Claborcorrelation])
        wellsite_and_gath = self.wellsite_opex_rate * (self.num_prd_active * self.CPrdwell + self.num_inj_active * self.CInjwell + self.Cgath)
        wellsite_labor = wells_are_active_01 * self.wellsite_labor_rate*self.Claborcorrelation
        self.opex["Wellsite"] =  wellsite_and_gath + wellsite_labor
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
        
        self.capex["Contingency"] = self.contingency * np.sum(list(self.capex.values()), axis=0) # Based on GETEM page 21
        
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
        
        self.compute_npv(print_outputs=print_outputs)

    def postprocess(self, print_outputs=True, compute_pumping=True):
        """postprocess results to compute pumping power, costs, revenues, economics, etc.

        Args:
            print_outputs (bool): whether or not to print economics parameters (e.g., NPV, LCOE)
        """

        self.df_records = pd.DataFrame.from_dict(self.records).set_index("World Time")
        self.df_records.iloc[0] = self.df_records.iloc[1] # overwrite the zero step as it is based on default params before the simulation actually starts
        
        if compute_pumping:
            self.compute_pumping()

        self.df_records['Pumping Power [MWe]'] = np.sum(self.res_PumpingPower_ideal, axis=0)[:self.step_idx]
        self.df_records['Production Pumping Power [MWe]'] = np.sum(self.res_PumpingPowerInj, axis=0)[:self.step_idx]
        self.df_records['Injection Pumping Power [MWe]'] = np.sum(self.res_PumpingPowerProd, axis=0)[:self.step_idx]
        self.df_records["Producer Wellhead Pressure [bar]"] = np.mean(self.res_WHP_Prod, axis=0)[:self.step_idx]/100 #['bar']

        self.df_records['Net Power Output [MWe]'] = self.df_records['Turbine Output [MWe]'].values + self.df_records['Battery Output [MWe]'].values - np.clip(self.df_records['Pumping Power [MWe]'].values, 0, np.inf)
        self.df_records['Net Power Generation [MWhe]'] = self.df_records['Net Power Output [MWe]'].values * pd.Series(self.df_records.index).diff().bfill().apply(lambda x: x.total_seconds()/3600).values

        # if more pumping is needed then we suppose that the operator would slow down operations and not generate
        self.df_records.loc[self.df_records["Net Power Output [MWe]"] < 0, "Net Power Output [MWe]"] = 0.0
        self.df_records.loc[self.df_records["Net Power Generation [MWhe]"] < 0, "Net Power Generation [MWhe]"] = 0.0

        self.compute_economics(print_outputs)

    def update_timestep_for_all_components(self, timestep):
        """Update the simulation timestep size across all project components.

        Args:
            timestep (datetime.timedelta): simulation timestep size
        """
        if timestep:
            self.timestep = timestep
            self.timestep_hrs = self.timestep.total_seconds() / 3600
            components = [self.powerplant, self.st, self.battery] + self.reservoirs
            for component in components:
                if component:
                    component.timestep = self.timestep

    def _reset(self, reset_market_weather=True):
        """Reseting the project to its initial state.
        """
        
        self.records = defaultdict(list)

        if reset_market_weather:
            # Create market
            self.market = TabularPowerMarket()
            self.market.create_energy_market(filepath=self.energy_filepath,
                                            resample=self.resample, 
                                            oversample_first_day=self.oversample_first_day,
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
            if self.weather_filepath or self.sup3rcc_weather_forecast:
                self.weather.create_weather_model(filepath=self.weather_filepath, 
                                                resample=False,
                                                sup3rcc_weather_forecast=self.sup3rcc_weather_forecast,
                                                project_lat=self.project_lat,
                                                project_long=self.project_long,
                                                years=range(self.start_year, self.end_year),
                                                n_jobs=self.n_jobs,
                                                )

                if self.sup3rcc_weather_forecast:
                    self.df_market = pd.merge(self.df_market, 
                            self.weather.df[['year', 'month', 'day', 'hour', "T0"]],
                            how='left', on=['year', 'month', 'day', 'hour'])
                
                else:
                    self.df_market = pd.merge(self.df_market, 
                                            self.weather.df[['month', 'day', 'hour', "T0"]],
                                            how='left', on=['month', 'day', 'hour'])
            else:
                self.df_market["T0"] = self.surface_temp

            self.df_market = self.df_market.bfill()

        # Configure simulation time
        self.step_idx = 0
        self.state = self.df_market.iloc[0]
        self.max_simulation_steps = len(self.df_market)
        self.timestep = self.df_market.TimeDiff.median()
        self.timestep_hrs = self.timestep.total_seconds() / 3600
        # self.times_arr = self.df_market.TimeDiff_seconds.values
        self.times_arr = self.df_market.TimeDiff_seconds.values
        self.times_arr[0] = 0
        self.times_arr = self.times_arr.cumsum()
        self.timesteps = self.df_market.TimeDiff.tolist()
        self.time_curr = self.time_init

        # Create TES storage tank object, if specified
        self.st = TES(time_init=self.time_init, d=self.tank_diameter, H=self.tank_height) if self.tank_diameter > 0 else None

        # Create Lithium-ion storage object, if specified
        self.battery = LiIonBattery(time_init=self.time_init, duration=self.battery_duration, power_capacity=self.battery_power_capacity, \
                                    battery_roundtrip_eff=self.battery_roundtrip_eff, lifetime=self.battery_lifetime) if max(self.battery_power_capacity) > 0 else None

        # Create power plant
        self.powerplant = self.create_powerplant()

        # Create reservoir
        self.reservoirs = [self.create_reservoir(self.num_prd, self.num_inj, self.time_init)]
        self.num_res = len(self.reservoirs)
        self.num_res_prd = np.array([r.num_prd for r in self.reservoirs])
        self.num_res_inj = np.array([r.num_inj for r in self.reservoirs])
        self.num_prd_drilled = np.array([self.num_prd] + [0 for _ in range(self.L-1)])
        self.num_inj_drilled = np.array([self.num_prd] + [0 for _ in range(self.L-1)])
        self.num_prd_shutoff = [0 for _ in range(self.L)]
        self.num_inj_shutoff = [0 for _ in range(self.L)]

        self.effective_ppc = self.powerplant_capacity #* (self.num_prd > 0) # zero if either no power plant or no wells installed
        self.update_timestep_for_all_components(self.timestep)

    def create_powerplant(self):
       # specify what production temperature the power plant should be designed for ...
        self.Tres_pp_design = self.Tres_init if self.Tres_pp_design is None else self.Tres_pp_design
        # Create power plant
        if "geophires" in self.powerplant_type.lower():
            if "flash" in self.powerplant_type.lower():
                powerplant = GEOPHIRESFlashPowerPlant(ppc=self.powerplant_capacity, Tres=self.Tres_pp_design, cf=self.cf)
            else:
                powerplant = GEOPHIRESORCPowerPlant(ppc=self.powerplant_capacity, Tres=self.Tres_pp_design, cf=self.cf)
        elif "HighEnthalpyCLGWGPowerPlant" in self.powerplant_type:
            powerplant = HighEnthalpyCLGWGPowerPlant(Tres=self.Tres_pp_design, 
                                                          Tamb=self.surface_temp, 
                                                          m_prd=self.m_prd, num_prd=self.num_prd, 
                                                          cf=self.cf)
        else:
            powerplant = ORCPowerPlant(ppc=self.powerplant_capacity, Tres=self.Tres_pp_design, Tamb=self.df_market['T0'].mean(), 
                                            m_prd=self.m_prd, m_prd_design=self.m_prd_pp_design, num_prd=self.num_prd, cf=self.cf, k=self.powerplant_k)
            powerplant_capacity = powerplant.powerplant_capacity
            if self.num_prd is None:
                self.num_prd = powerplant.num_prd
                self.num_inj = max(int(np.floor(self.inj_prd_ratio * self.num_prd)), 1) 
        
        return powerplant

    def create_reservoir(self, num_prd, num_inj, time_init):
       # if reservoir filepath is provided, then override reservoir object to use the provided tabular data
        if self.reservoir_filepath:
            reservoir = TabularReservoir(Tres_init=self.Tres_init, geothermal_gradient=self.geothermal_gradient, surface_temp=self.surface_temp, 
                            L=self.L, time_init=time_init, well_tvd=self.well_tvd, 
                            prd_well_diam=self.prd_well_diam, inj_well_diam=self.inj_well_diam, num_prd=num_prd, num_inj=num_inj, waterloss=self.waterloss,
                            powerplant_type=self.powerplant_type, pumpeff=self.pumpeff, PI=self.PI, II=self.II, SSR=self.SSR, 
                            drawdp=self.drawdp, plateau_length=self.plateau_length, reservoir_simulator_settings=self.reservoir_simulator_settings, ramey=self.ramey, PumpingModel=self.PumpingModel,
                            filepath=self.reservoir_filepath)

            self.m_prd = reservoir.df["m_prd_kg_per_sec"].median()
            self.Tres_init = reservoir.Tres_init
            self.geothermal_gradient = (self.Tres_init - self.surface_temp)/self.well_tvd*1000

        else:
            if self.reservoir_type == "energy_decline":
                reservoir = EnergyDeclineReservoir(Tres_init=self.Tres_init, Pres_init=self.Pres_init, geothermal_gradient=self.geothermal_gradient, surface_temp=self.surface_temp, 
                                            L=self.L, time_init=self.time_init, well_tvd=self.well_tvd, 
                                            prd_well_diam=self.prd_well_diam, inj_well_diam=self.inj_well_diam, num_prd=self.num_prd, 
                                            num_inj=self.num_inj, waterloss=self.waterloss,
                                            powerplant_type=self.powerplant_type, pumpeff=self.pumpeff, PI=self.PI, II=self.II, SSR=self.SSR, V_res=self.V_res, phi_res=self.phi_res,
                                            rock_energy_recovery=self.rock_energy_recovery, reservoir_simulator_settings=self.reservoir_simulator_settings, ramey=self.ramey, PumpingModel=self.PumpingModel)

            elif self.reservoir_type == "diffusion_convection":                
                reservoir = DiffusionConvection(Tres_init=self.Tres_init, geothermal_gradient=self.geothermal_gradient, surface_temp=self.surface_temp,
                                            L=self.L, time_init=self.time_init, well_tvd=self.well_tvd, 
                                            prd_well_diam=self.prd_well_diam, inj_well_diam=self.inj_well_diam, num_prd=self.num_prd, 
                                            num_inj=self.num_inj, waterloss=self.waterloss, 
                                            powerplant_type=self.powerplant_type, pumpeff=self.pumpeff, PI=self.PI, II=self.II, SSR=self.SSR, 
                                            V_res=self.V_res, phi_res=self.phi_res, lateral_length=self.lateral_length, res_thickness=self.res_thickness, 
                                            krock=self.krock, cprock=self.cprock, reservoir_simulator_settings=self.reservoir_simulator_settings, PumpingModel=self.PumpingModel, ramey=self.ramey)

            elif self.reservoir_type == "uloop":
                reservoir = ULoopSBT(Tres_init=self.Tres_init, Pres_init=self.Pres_init, surface_temp=self.surface_temp, geothermal_gradient=self.geothermal_gradient,
                            prd_well_diam=self.prd_well_diam, inj_well_diam=self.inj_well_diam, lateral_diam=self.lateral_diam, 
                            well_tvd = self.well_tvd, numberoflaterals=self.numberoflaterals, half_lateral_length = self.half_lateral_length,
                            lateral_spacing=self.lateral_spacing, L=self.L, time_init=self.time_init, num_prd=self.num_prd, num_inj=self.num_inj, 
                            waterloss=self.waterloss, powerplant_type=self.powerplant_type, pumpeff=self.pumpeff, PI=self.PI, II=self.II,
                            times_arr=np.linspace(0, self.max_simulation_steps, self.max_simulation_steps), reservoir_simulator_settings=self.reservoir_simulator_settings, PumpingModel=self.PumpingModel,
                            closedloop_design=self.closedloop_design, ramey=self.ramey, dx=self.dx, k_m=self.krock)
                self.total_drilling_length, self.prd_total_drilling_length, self.inj_total_drilling_length = reservoir.total_drilling_length, reservoir.total_drilling_length/2, reservoir.total_drilling_length/2
            
            elif self.reservoir_type == "coaxial":
                reservoir = CoaxialSBT(Tres_init=self.Tres_init, Pres_init=self.Pres_init, surface_temp=self.surface_temp, geothermal_gradient=self.geothermal_gradient,
                                            casing_inner_diam=self.casing_inner_diam, tube_inner_diam=self.tube_inner_diam, tube_thickness=self.tube_thickness, k_tube=self.k_tube,
                                            coaxialflowtype=self.coaxialflowtype,
                                            well_tvd = self.well_tvd, L=self.L, time_init=self.time_init, num_well=self.num_prd, 
                                            waterloss=self.waterloss, powerplant_type=self.powerplant_type, pumpeff=self.pumpeff,
                                            times_arr=self.times_arr, 
                                            reservoir_simulator_settings=self.reservoir_simulator_settings, 
                                            dx=self.dx, k_m=self.krock
                                            )
                self.total_drilling_length, self.prd_total_drilling_length, self.inj_total_drilling_length = reservoir.total_drilling_length, reservoir.total_drilling_length, 0.0
            else:
                # Default to percentage drawdown model
                reservoir = PercentageReservoir(Tres_init=self.Tres_init, geothermal_gradient=self.geothermal_gradient, surface_temp=self.surface_temp, 
                                L=self.L, time_init=self.time_init, well_tvd=self.well_tvd, 
                                prd_well_diam=self.prd_well_diam, inj_well_diam=self.inj_well_diam, num_prd=self.num_prd, 
                                num_inj=self.num_inj, waterloss=self.waterloss,
                                powerplant_type=self.powerplant_type, pumpeff=self.pumpeff, PI=self.PI, II=self.II, SSR=self.SSR, 
                                drawdp=self.drawdp, plateau_length=self.plateau_length, reservoir_simulator_settings=self.reservoir_simulator_settings, ramey=self.ramey, PumpingModel=self.PumpingModel)
        
        return reservoir

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
            if pct < 1:
                return ""
            return "{:.1f}%".format(pct, absolute)

        expenditures = {}

        include = ['Power Plant', 'Interconnection', 'Exploration', 
                'Drilling', 'Stimulation', 'Gathering System', 'Pumps', 'TES', 'Battery']
        costs = {k:v for k,v in self.present_capex_per_unit.items()}
        costs["Pumps"] = costs["Production Pumps"] + costs["Injection Pumps"]
        costs_final = {k:costs[k] for k in include}
        expenditures["CAPEX"] = costs_final

        include = ['Power Plant', 'Wellsite', 'Makeup Water']
        costs = {k:v for k,v in self.present_opex_per_unit.items()}
        costs_final = {k:costs[k] for k in include}
        expenditures["OPEX"] = costs_final

        for i, (title, present_per_unit) in enumerate(expenditures.items()):
            ex, labels = [], []
            for k,v in present_per_unit.items():
                if v != 0: 
                    ex.append(np.empty([]) if (v < 1 and title=='CAPEX') else v )
                    labels.append(k)
            wedges, _, _ = axes[i].pie(x=ex,
                                    pctdistance=0.8,
        #                             labels=labels, 
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

    def plot_operations(self, span=None, 
            qdict = {
            "LMP [$/MWh]": "Electricity Price \n [$/MWh]",
          "Atm Temp [deg C]": "Ambient Temp. \n [° C]",
          "Res Temp [deg C]": "Reservoir Temp. \n [° C]",
          'Inj Temp [deg C]': "Injector Temp. \n [° C]",
          "Net Power Output [MWe]": "Net Power Output \n [MWe]",
          'M_Produced [kg/s]': "Field Production \n [kg/s]",
          "Pumping Power [MWe]": "Pumping Power \n [MWe]"},
          figsize=(10,12),
          legend_loc=False,
          dpi=100,
          formattime=False,
        ):
        """Plot operational parameters.

        Args:
            span (Union[range, list], optional): range of timesteps to plot. Defaults to None.

        Returns:
            matplotlib.figure.Figure: figure
        """

        if not hasattr(self, "df_records"):
            warnings.warn("Warning: economics are computed based on the latest simulation timestep.")
            self.compute_economics()

        quantities = list(qdict.keys())
        ylabels = list(qdict.values())

        span = span if span else range(int(0.01*self.max_simulation_steps), self.step_idx)
        fig = plot_cols({" ": self.df_records}, span, quantities, 
                                figsize=figsize, ylabels=ylabels, legend_loc=legend_loc, dpi=dpi, 
                            formattime=formattime)
        
        return fig

    def compute_npv(self, ppa_price=None, ppa_escalaction_rate=None, print_outputs=True):
        """Compute NPV and other economic metrics for a completed simulation run.

        Args:
            ppa_price (float, optional): price of power purchase agreement in USD/MWh. Defaults to 75.
            ppa_escalaction_rate (float, optional): price escalation of power purchase agreement (fraction). Defaults to 0.02.
        """

        self.ppa_price = ppa_price if ppa_price else self.ppa_price
        self.ppa_escalaction_rate = ppa_escalaction_rate if ppa_escalaction_rate else self.ppa_escalaction_rate

        years = np.arange(self.L)
        self.df_annual_nominal = self.df_records.groupby('Year').sum(numeric_only=True)

        # This ensures that we captured all columns
        self.df_annual_nominal = pd.merge(self.df_annual_nominal,
                pd.DataFrame(min(self.df_annual_nominal.index) + np.arange(self.L), columns=["Year"]).set_index("Year"),
                left_index=True, right_index=True,
                how="outer").fillna(0)

        # only allow for non-negative net power generation, where we ramp down when negative
        self.df_annual_nominal["PPA Revenue [$MM]"] = self.df_annual_nominal["Net Power Generation [MWhe]"]\
            *self.ppa_price/1e6 * (1 + self.ppa_escalaction_rate)**years
        self.df_annual_nominal["CAPEX [$MM]"] = self.capex_total
        self.df_annual_nominal["OPEX [$MM]"] = self.opex_total

        self.df_annual_nominal["Cashin [$MM]"] = self.df_annual_nominal["Revenue [$MM]"]
        self.df_annual_nominal["PPA Cashin [$MM]"] = self.df_annual_nominal["PPA Revenue [$MM]"]
        self.df_annual_nominal["Cashout [$MM]"] = self.df_annual_nominal["OPEX [$MM]"] + self.df_annual_nominal["CAPEX [$MM]"]
        self.df_annual_nominal["Net_Profit [$MM]"] = self.df_annual_nominal["Cashin [$MM]"] - self.df_annual_nominal["OPEX [$MM]"]
        self.df_annual_nominal["PPA Net_Profit [$MM]"] = self.df_annual_nominal["PPA Cashin [$MM]"] - self.df_annual_nominal["OPEX [$MM]"]
        self.df_annual_nominal["Cashflow [$MM]"] = self.df_annual_nominal["Cashin [$MM]"] - self.df_annual_nominal["Cashout [$MM]"]
        self.df_annual_nominal["PPA Cashflow [$MM]"] = self.df_annual_nominal["PPA Cashin [$MM]"] - self.df_annual_nominal["Cashout [$MM]"]

        self.df_annual = self.df_annual_nominal.div((1 + self.d)**years, axis=0)
        self.df_annual["NPV [$MM]"] = self.df_annual["Cashflow [$MM]"].cumsum()
        self.df_annual["PPA NPV [$MM]"] = self.df_annual["PPA Cashflow [$MM]"].cumsum()
        self.df_annual["Revenue [$MM]"] = self.df_annual["Cashin [$MM]"].cumsum()
        self.df_annual["Cost [$MM]"] = self.df_annual["Cashout [$MM]"].cumsum()
        self.df_annual["Cum CAPEX [$MM]"] = self.df_annual["CAPEX [$MM]"].cumsum()
        self.df_annual["Cum OPEX [$MM]"] = self.df_annual["OPEX [$MM]"].cumsum()
        self.df_annual["ROI [%]"] = self.df_annual_nominal["Net_Profit [$MM]"].cumsum()/self.df_annual_nominal["CAPEX [$MM]"].cumsum() * 100
        self.df_annual["PPA ROI [%]"] = self.df_annual_nominal["PPA Net_Profit [$MM]"].cumsum()/self.df_annual_nominal["CAPEX [$MM]"].cumsum() * 100
        self.df_annual['Res Temp [deg C]'] = self.df_records.groupby('Year').mean(numeric_only=True)["Res Temp [deg C]"]
        self.df_annual['WH Temp [deg C]'] = self.df_records.groupby('Year').mean(numeric_only=True)["WH Temp [deg C]"]
        self.df_annual['Atm Temp [deg C]'] = self.df_records.groupby('Year').mean(numeric_only=True)["Atm Temp [deg C]"]
        self.NPV = self.df_annual["NPV [$MM]"].iloc[-1]
        self.ROI = self.df_annual["ROI [%]"].iloc[-1]
        self.PBP = self.df_annual_nominal.index[np.argmax((self.df_annual_nominal["Cashflow [$MM]"].cumsum()>0).values)] - self.start_year
        self.IRR = npf.irr(self.df_annual_nominal["Cashflow [$MM]"].values) * 100
        self.PPA_NPV = self.df_annual["PPA NPV [$MM]"].iloc[-1]
        self.PPA_ROI = self.df_annual["PPA ROI [%]"].iloc[-1]
        self.PPA_PBP = self.df_annual_nominal.index[np.argmax((self.df_annual_nominal["PPA Cashflow [$MM]"].cumsum()>0).values)] - self.start_year
        self.PPA_IRR = npf.irr(self.df_annual_nominal["PPA Cashflow [$MM]"].values) * 100
        # ignore years with negative power generation, which could occur upon reservoir depletion
        # self.DISCOUNTED_NET_GEN = self.df_annual.loc[self.df_annual["Net Power Generation [MWhe]"]>0, "Net Power Generation [MWhe]"].sum()
        self.DISCOUNTED_NET_GEN = self.df_annual["Net Power Generation [MWhe]"].sum()
        self.ARR = self.df_annual["Cashflow [$MM]"].mean()/self.capex_total[0]*100
        self.PPA_ARR = self.df_annual["PPA Cashflow [$MM]"].mean()/self.capex_total[0]*100
        self.equiv_ppa_price = self.df_annual["Cashin [$MM]"].sum()*1e6/nonzero(self.DISCOUNTED_NET_GEN, 1E-1)

        self.LCOE = self.df_annual["Cashout [$MM]"].sum()*1e6/nonzero(self.DISCOUNTED_NET_GEN, 1E-1)
        if (self.LCOE < 0) or (self.DISCOUNTED_NET_GEN < 0): # cases where pumping requirements are greater than gross power generation
            self.LCOE = 999
        
        # ignore years with negative power generation, which could occur upon reservoir depletion
        self.NET_GEN = self.df_annual_nominal["Net Power Generation [MWhe]"].sum()
        self.NET_CF = self.NET_GEN/(8760*self.L)/self.powerplant.powerplant_capacity
        self.AVG_T_AMB = self.df_records["Atm Temp [deg C]"].mean()
        

        # print economics
        if print_outputs:
            print(f"LCOE: {self.LCOE:.0f} $/MWh") 
            print(f"NPV: {self.NPV:.0f} $MM")
            print(f"PBP: {self.PBP:.0f} yrs")

    def set_defaults(self):
        """Set default parameters that are not specified by the user.
        """
        
        self.project_data_dir = os.path.join(os.getcwd(), "./data/")
        self.n_jobs = 1

        self.time_init = "2025-01-01"
        self.L = 30
        self.d = 0.07
        self.itc = 0.0
        self.inflation = 0.02
        self.contingency = 0.15
        self.project_lat = None
        self.project_long = None

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

        self.powerplant_capacity = None
        self.pipinglength = 5
        self.powerplant_interconnection_cost = 130
        self.bypass = False
        self.wellsite_opex_rate = 1/100
        self.wellsite_labor_rate = 0.25
        self.powerplant_opex_rate = 1.5/100
        self.powerplant_labor_rate = 0.75
        self.powerplant_usd_per_kw_min = 2000 # USD/kWe # This is meant to limit projects to a max of 100 MWe increments

        self.weather_filename = None
        self.ppa_price = 70
        self.ppa_escalaction_rate = 0.02
        self.energy_price = 40
        self.recs_price = 10
        self.capacity_price = 100
        self.energy_market_filename = None
        self.capacity_market_filename = None
        self.fat_factor = 1
        self.resample = "1Y"
        self.sup3rcc_weather_forecast = False

        self.total_drilling_length, self.prd_total_drilling_length, self.inj_total_drilling_length = None, None, None
        self.drilling_cost = None
        self.redrill_ratio = None
        self.shutoff_ratio = None
        self.exploration_cost_intercept = 1.0 #$MM
        self.exploration_cost_slope = 0.6 #USD exploration / USD producer capex
        self.stimulation_cost = 2.5 # $MM per injection well drilled and completed successfully
        self.reservoir_filename = None
        self.reservoir_type = "diffusion_convection"
        self.Pres_init = 40
        self.V_res = 5
        self.phi_res = 0.1
        self.res_thickness = 300
        self.krock = 3
        self.cprock = 1100
        self.drawdp = 0.005
        self.plateau_length = 3
        self.rock_energy_recovery = 1.0
        self.surface_temp = 20
        self.well_tvd = 3000
        self.prd_well_diam = 0.3115
        self.int_well_diam = 0.3115
        self.numberoflaterals = 1
        self.num_prd = 4
        self.inj_prd_ratio = 1.0
        self.waterloss = 0.05
        self.pumpeff = 0.75
        self.DSR = 1.0
        self.SSR = 1.0
        self.PI = 10
        self.II = 10
        self.lateral_length = 0
        self.lateral_diam = 0.3115
        self.lateral_spacing = 100
        self.PumpingModel = "OpenLoop"
        self.closedloop_design = "default"
        self.dx = None
        self.ramey = None
        self.reservoir_simulator_settings = {"fast_mode": False, "period": 31536000, 
        "accuracy": 1, "DynamicFluidProperties": True}
        self.geothermal_gradient = 35 #C/km
        self.Tres_pp_design = None
        self.m_prd_pp_design = 80
        self.m_prd = 100
        self.powerplant_k = 2
        self.oversample_first_day = None

        self.casing_inner_diam = 0.13208
        self.tube_inner_diam = 0.0620014
        self.tube_thickness = 0.0395986/2
        self.k_tube = 0.088
        self.coaxialflowtype = 1

if __name__ == '__main__':
    pass