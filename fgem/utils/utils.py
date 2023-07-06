import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import pdb 
from pyXSteam.XSteam import XSteam
from scipy.optimize import curve_fit
import pickle
from matplotlib.ticker import FormatStrFormatter
from shapely.geometry.polygon import orient

steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS) # m/kg/sec/°C/bar/W

colors = 4*sns.color_palette()

linestyles = {0: 'solid',
              1: 'dashed',
              2: 'dotted',
              3: 'dashdot'}

class FastXsteam(object):
    
    """Faster corrleations for Steam Tables."""
    
    def __init__(self, T_max=300, timesteps=30000):
        
        """Define attributes for FastXsteam class."""
        
        self.xdata = np.linspace(1, T_max, timesteps)
        self.tc = 647.096 #K
        
        ydata = [steamTable.hL_t(x) for x in self.xdata]
        self.popt_hl, pcov = curve_fit(self.func_hl, self.xdata, ydata)
        ydata = [steamTable.hV_t(x) for x in self.xdata]
        self.popt_hv, pcov = curve_fit(self.func_hv, self.xdata, ydata)
        ydata = [steamTable.vL_t(x) for x in self.xdata]
        self.popt_vl, pcov = curve_fit(self.func_vl, self.xdata, ydata)
        ydata = [steamTable.vV_t(x) for x in self.xdata]
        self.popt_vv, pcov = curve_fit(self.func_vv, self.xdata, ydata)
        
        
    def func_hl(self, t, a, b, c, d, e):
        """Liquid enthalpy correlation."""
        tr = (t+273.15)/self.tc
        return (a + b * np.log(1/tr)**0.35 + c/tr**2 + d/tr**3 + e/tr**4)
    def func_hv(self, t, a, b, c, d, e):
        """Vapor enthalpy correlation."""
        tr = (t+273.15)/self.tc
        return (a + b * np.log(1/tr)**0.35 + c/tr**2 + d/tr**3 + e/tr**4)**(1/2)
    def func_vl(self, t, a, b, c, d, e):
        """Liquid specific volume calculator."""
        tr = (t+273.15)/self.tc
        return (a + b * np.log(1/tr)**0.35 + c/tr**2 + d/tr**3 + e/tr**4)
    def func_vv(self, t, a, b, c, d, e):
        """Vapor specific volume calculator."""
        tr = (t+273.15)/self.tc
        return np.exp((a + b * np.log(1/tr)**0.35 + c/tr**2 + d/tr**3 + e/tr**4))

    def hL_t(self, t):
        """Liquid enthalpy correlation."""
        t = t[0] if isinstance(t, np.ndarray) else t
        return self.func_hl(t, *self.popt_hl)
        
    def hV_t(self, t):
        """Vapor enthalpy correlation."""
        t = t[0] if isinstance(t, np.ndarray) else t
        return self.func_hv(t, *self.popt_hv)

    def vL_t(self, t):
        """Liquid specific volume calculator."""
        t = t[0] if isinstance(t, np.ndarray) else t
        return self.func_vl(t, *self.popt_vl)
    
    def vV_t(self, t):
        """Vapor specific volume calculator."""
        t = t[0] if isinstance(t, np.ndarray) else t
        return self.func_vv(t, *self.popt_vv)


def plot_cols(dfs, 
              span,
              quantities, 
              figsize=(10,10),
              xlabel="World Time",
              legend_loc="lower right",
              format_yticks=True,
              dpi=100,
              colors=colors):
    
    """Plotting of dataframe columns (specifically suitable for time-series headers)."""
    
    fig, axes = plt.subplots(len(quantities), 1, figsize=figsize, sharex=True, dpi=dpi)

    df_plots = [df.iloc[span].copy() for df in dfs.values()]

    for k, df_plot in enumerate(df_plots):
        counter = 0
        for i, q in enumerate(quantities):
            if format_yticks:
                axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            q = q if isinstance(q, list) else [q]
            for col in q:
                if all([qi in df_plot.columns for qi in q]):
                    axes[i].plot(df_plot.index, df_plot[col], color=colors[counter], linestyle=linestyles[k])
                counter += 1
                if legend_loc:
                    axes[i].legend(q, loc=legend_loc)
    
    if len(dfs.keys()) > 1:
        axes[0].set_title("\n".join([f"{k}: {linestyles[i]}" for i, k in enumerate(dfs.keys())]))
    else:
        axes[0].set_title(list(dfs.keys())[0])
    axes[i].set_xlabel(xlabel)
    plt.show()

def plot_ex(worlds,
            figsize=(10,10),
            dpi=150,
            colors=colors,
            fontsize=10
               ):
    
    """Create pie chart plots for capital and operational expenditure."""
    
    fig, axes = plt.subplots(1, len(worlds), figsize=figsize, dpi=dpi)
    axes = [axes] if len(worlds) == 1 else axes

    def func(pct, allvals):
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        # return "{:.0f}%\n({:d} $MM)".format(pct, absolute)
        return "{:.0f}%".format(pct, absolute)
    
    for i, (title, present_per_unit) in enumerate(worlds.items()):
        ex = [v for _,v in present_per_unit.items() if v > 0]
        labels = [k for k,v in present_per_unit.items() if v > 0]
        wedges, _, _ = axes[i].pie(x=ex, 
                                   labels=labels, 
                                   colors=colors, 
                                   autopct=lambda pct: func(pct, ex),
                                   textprops=dict(color="w", weight="bold", fontsize=fontsize))

        axes[i].legend(wedges, labels,
                title=title,
                loc="center left",
                bbox_to_anchor=(0.3, 0.6, 0.0, 1))

    plt.show()

def prepare_tabular_world(world):
    
    """Create a tabular version of the world environment based on historical/forecast market/weather data."""
    
    df_market = world.market.df.copy()
    df_market = df_market[(df_market.year >= world.start_year) & (df_market.year < world.end_year)].copy()
    df_market["capacity_price"] = df_market["year"].apply(lambda t: world.market.get_capacity_price(t))
    df_market["battery_elcc"] = df_market["year"].apply(
        lambda t: world.market.get_elcc(t, world.battery_duration[int(t-world.start_year > world.battery_lifetime)]) if world.battery else 0.0)
    df_weather = pd.DataFrame(np.tile(world.weather.df.values, (world.L, 1)), columns = world.weather.df.columns)
    df_market["T0"] = df_weather["T0"].values

    return df_market

def geothermal_trader(world,
                      m_prd,
                      m_inj,
                      m_g,
                      m_tes_ins,
                      m_tes_outs,
                      p_bat_ins,
                      p_bat_outs,
                      disable_tqdm=True):
    
    """Demonstration of how to use the world environment to flexibly control thermal/battery storage facilities."""
    
    world._reset()

    df_L = prepare_tabular_world(world)
    
    for i, (market_date, row) in tqdm(enumerate(df_L.iterrows()), total=len(df_L), disable=disable_tqdm):

        # Skip first step as a warmup
        if i == 0:
            continue
        
        timestep = row["TimeDiff"]
        timestep_sec = timestep.seconds
        timestep_hr = timestep_sec/3600

        # Constrained TES
        m_tes_in, m_tes_out = m_tes_ins[i], m_tes_outs[i]
        
        if world.st:
            power_output_MWh_kg = world.pp.compute_power_output(world.reservoir.T_prd.mean(), row["T0"])
            leftover_ppcap_kg = max(0, world.ppc / power_output_MWh_kg /3600 - m_g)
        
            m_tes_in = max(min(m_tes_in, m_g, world.st.mass_max_charge/timestep_sec), 0)
            m_tes_out = max(min(m_tes_out, leftover_ppcap_kg, world.st.mass_max_discharge/timestep_sec), 0)

        else:
            m_tes_in, m_tes_out = 0.0, 0.0
        
        # Constrained Battery
        m_wh_to_turbine = m_g - m_tes_in
        m_turbine = m_wh_to_turbine + m_tes_out
        p_bat_in, p_bat_out = p_bat_ins[i], p_bat_outs[i]
        if world.battery:
            _, wh_power_output_MWe, _, _, _= world.pp.power_plant_outputs(timestep, 
                                                                        m_turbine, 
                                                                        m_wh_to_turbine, 
                                                                        m_tes_out,
                                                                        world.reservoir.T_prd.mean(), 
                                                                        world.st.Tw if world.st else 100, 
                                                                        row["T0"])
            empty_battery_capacity = (world.battery.energy_capacity - world.battery.energy_content) / timestep_hr
            content_battery_capacity = world.battery.energy_content / timestep_hr
            battery_power_capacity = world.battery.power_capacity
            p_bat_in = max(min(p_bat_in, wh_power_output_MWe, empty_battery_capacity, battery_power_capacity), 0)
            p_bat_out = max(min(p_bat_out, content_battery_capacity, battery_power_capacity), 0)
            
        else:
            p_bat_in, p_bat_out = 0.0, 0.0
        
        # Step World
        power_output_MWe, power_generation_MWh, T_inj = world.step(timestep, row, 
                                                                    m_prd, m_inj, 
                                                                    m_tes_in, m_tes_out,
                                                                    p_bat_in, p_bat_out)
        # Record World
        world.record_step()
        
        # if world.st.Tw < 150:
        #     pdb.set_trace()

    df_records = pd.DataFrame.from_dict(world.records).set_index("World Time")

    if world.battery:
        df_records["WoG Charge Cost [$MM]"] = df_records["Bat Charge [MWe]"] * df_records["LMP [$/MWh]"]/1e6
        df_records["WoG Wholesale Revenue [$MM]"] = df_records["Bat Discharge [MWe]"] * df_records["LMP [$/MWh]"]/1e6
        df_records["WoG Capacity Revenue [$MM]"] = df_records["Battery Capacity Revenue [$MM]"]

    df_annual = df_records.groupby('Year').sum()
    df_annual["WG CAPEX [$MM]"] = world.capex_total
    df_annual["WG OPEX [$MM]"] = world.opex_total
    df_annual["WoG CAPEX [$MM]"] = world.capex["Battery"]
    df_annual["WoG OPEX [$MM]"] = world.opex["Battery"]

    df_annual["WG Cashin [$MM]"] = df_annual["Revenue [$MM]"]
    df_annual["WG Cashout [$MM]"] = df_annual["WG OPEX [$MM]"] + df_annual["WG CAPEX [$MM]"]
    df_annual["WG Cashflow [$MM]"] = df_annual["WG Cashin [$MM]"] - df_annual["WG Cashout [$MM]"]
    # Additional battery calculations
    if world.battery:
        df_annual["WoG Cashin [$MM]"] = df_annual["WoG Wholesale Revenue [$MM]"] + df_annual["WoG Capacity Revenue [$MM]"]
        df_annual["WoG Cashout [$MM]"] = df_annual["WoG Charge Cost [$MM]"] + df_annual["WoG OPEX [$MM]"] + df_annual["WoG CAPEX [$MM]"]
        df_annual["WoG Cashflow [$MM]"] = df_annual["WoG Cashin [$MM]"] - df_annual["WoG Cashout [$MM]"]
    
    years = np.arange(world.L)
    df_annual = df_annual.div((1 + world.d)**years, axis=0)
    df_annual["WG NPV [$MM]"] = df_annual["WG Cashflow [$MM]"].cumsum()
    df_annual["WG Revenue [$MM]"] = df_annual["WG Cashin [$MM]"].cumsum()
    df_annual["WG Cost [$MM]"] = df_annual["WG Cashout [$MM]"].cumsum()
    df_annual["WG Cum CAPEX [$MM]"] = df_annual["WG CAPEX [$MM]"].cumsum()
    df_annual["WG Cum OPEX [$MM]"] = df_annual["WG OPEX [$MM]"].cumsum()
    df_annual["WG ROI [%]"] = ((df_annual["WG Cashin [$MM]"] - df_annual["WG OPEX [$MM]"])/df_annual["WG CAPEX [$MM]"].sum()).cumsum() * 100
    
    if world.battery:
        df_annual["WoG NPV [$MM]"] = df_annual["WoG Cashflow [$MM]"].cumsum()
        df_annual["WoG Revenue [$MM]"] = df_annual["WoG Cashin [$MM]"].cumsum()
        df_annual["WoG Cost [$MM]"] = df_annual["WoG Cashout [$MM]"].cumsum()
        df_annual["WoG Cum CAPEX [$MM]"] = df_annual["WoG CAPEX [$MM]"].cumsum()
        df_annual["WoG ROI [%]"] = ((df_annual["WoG Cashin [$MM]"] - df_annual["WoG OPEX [$MM]"])/df_annual["WoG CAPEX [$MM]"].sum()).cumsum() * 100
   
    NPV = df_annual["WG NPV [$MM]"].iloc[-1]
    Revenue = df_annual["WG Revenue [$MM]"].iloc[-1]
    CAPEX = df_annual["WG Cum CAPEX [$MM]"].iloc[-1]
    Net_Income = NPV + CAPEX
    ROI = df_annual["WG ROI [%]"].iloc[-1]

    return NPV, Revenue, CAPEX, Net_Income, ROI, [], df_records, df_annual

def check_conv(in_size, kernels, strides, pads):
    
    """Check a convolutional layer has the right size."""
    
    h = in_size
    for k, s, p in zip(kernels, strides, pads):
        h = (h - k + 2 * p)//s + 1
        print(h)
    return h

def plot_cols_v2(dfs, 
              span,
              quantities, 
              figsize=(10,10),
              xlabel="World Time",
              ylabels=None,
              ylogscale=None,
              use_title=True,
              legend_loc="lower right",
              manual_legends=None,
              color_per_col=True,
              use_linestyles=True,
              blackout_first=False,
              formattime = False,
              dpi=100):
    
    """A more involved version of plotting columns of a dataframe (specifically made for time-series headers)."""
    
    fig, axes = plt.subplots(len(quantities), 1, figsize=figsize, sharex=True, dpi=dpi)

    df_plots = [df.iloc[span].copy() for df in dfs.values()]
    counter = 0
    
    for k, df_plot in enumerate(df_plots):
        for i, q in enumerate(quantities):
            q = q if isinstance(q, list) else [q]
            for col in q:
                if all([qi in df_plot.columns for qi in q]):
#                     axes[i].plot(df_plot.index, df_plot[col].rolling(10).mean(), color=colors[counter], linestyle=linestyles[k if use_linestyles else 0])
                    axes[i].plot(df_plot.index, df_plot[col], color=colors[counter], linestyle=linestyles[k if use_linestyles else 0])
                if color_per_col:
                    counter += 1
                if legend_loc:
                    axes[i].legend(q, loc=legend_loc)
        if not color_per_col:
            counter += 1
        else:
            counter = 0
    
    if use_title:
        if len(dfs.keys()) > 1:
            axes[0].set_title("\n".join([f"{k}: {linestyles[i]}" for i, k in enumerate(dfs.keys())]))
        else:
            axes[0].set_title(list(dfs.keys())[0])
    
    axes[i].set_xlabel(xlabel)
    if formattime:
        if "time" in xlabel.lower() or "date" in xlabel.lower():
            plt.gcf().autofmt_xdate()
    
    if ylabels:
        for i, ylabel in enumerate(ylabels):
            axes[i].set_ylabel(ylabel)
    if ylogscale:
        for i, log in enumerate(ylogscale):
            if log:
                axes[i].set_yscale("log")
    if manual_legends:
        axes[0].legend(list(dfs.keys()), loc='upper right', fontsize=10)
    if blackout_first:
        axes[0].plot(df_plot.index, df_plot[quantities[0]], color='black')
    
    plt.show()

def viz_wholesale(filepath, span=range(2025,2056,2)):
    
    """Visualize wholesale market data."""
    
    if isinstance(filepath, str):    
        df = pd.read_csv(filepath)
        df.Date = pd.to_datetime(df.Date, format="%m/%d/%y %H:%M")
    else:
        df = filepath.copy()

    df["Year"], df["Hour"] = df.Date.dt.year, df.Date.dt.hour
    df = pd.pivot_table(df, values='price', index=['Hour'],
                           columns=['Year'], aggfunc=np.mean)
    mappaple = df.groupby("Hour").mean().plot(y=span, cmap='jet')
    plt.legend(ncol=4, loc='upper right')
    plt.xlabel("Hour of Day")
    plt.ylabel("Price [Nominal $/MWh]")
    plt.show()
    
def preprocess_cambium_capacity(scenario, state, infl,
                                base_year=2023, span=range(2024,2061), 
                                plot=False, dst_dir='.', save=None, adjustment=0):

    """Preprocess and plot Cambium capacity market data for a chosen state and forecast scenario."""
    
    filepath = f"../../Cambium_2022/{scenario}/Cambium22_{scenario}_annual_state.csv"
    df = pd.read_csv(filepath, header=5)
    df = df.loc[df.state == state, ["t", "capacity_shadow_price"]]
    df = df.rename(columns={"t": "Year", "capacity_shadow_price": "capacity cost"})
    df["capacity cost"] /= 1e3
    df_full = pd.merge(pd.DataFrame({"Year": span}), df, how='outer', on="Year").interpolate(method="linear", order=1)
    if adjustment > 0:
        df_full["capacity cost"] += adjustment #based on Fig 3-C of https://eta-publications.lbl.gov/sites/default/files/berkeley_lab_2021.11-_integrating_cambium_prices_into_electric-sector_decisions.pdf
    df_full["capacity cost"] = df_full["capacity cost"] * (1+infl)**(df_full.Year - base_year)
    if plot:
        # plt.scatter(df["Year"], df["capacity cost"], color="black")
        plt.plot(df_full["Year"], df_full["capacity cost"])
        plt.xlabel("Year")
        plt.ylabel("Capacity Value [Nominal $/kW-yr]")
        plt.show()
    
    if save:
        df_full.to_csv(os.path.join(dst_dir, "Capacity.csv"), index=False)
        
    return df_full

def preprocess_cambium_wholesale(scenario, state, infl, daily_fat_factor=1.0, 
                                 seasonal_fat_factor=1.0, base_year=2022, span=range(2024, 2061), 
                                 plot=False, dst_dir='.', save=False):
    
    """Preprocess and plot Cambium wholesale market data for a chosen state and forecast scenario."""
    
    base_dir = os.path.join("../../Cambium_2022/", scenario)
    data_dir = os.path.join(base_dir, "hourly_state")
    filepaths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if f"_{state}_" in filename]
    avail_years = np.sort([int(f.split(".csv")[0].split("_")[-1]) for f in filepaths])

    df_recs = pd.read_csv(os.path.join(base_dir, "RECs.csv"))
    df_recs["Date"] = pd.to_datetime(df_recs["Date"], format="%m/%d/%y %H:%M")
    df_recs = df_recs.loc[df_recs.Date.dt.year >= 2024]

    df = []
    for i, year in enumerate(avail_years):
        filepath = os.path.join(data_dir, f"Cambium22_{scenario}_hourly_{state}_{year}.csv")
        df_temp = pd.read_csv(filepath, header=5)
        
        if i == 0:
            df_temp["Date"] = pd.to_datetime(df_temp["timestamp_local"])
            df_temp["Month"] = df_temp.Date.dt.month
            df_temp["Day"] = df_temp.Date.dt.day
            df_temp["Hour"] = df_temp.Date.dt.hour
            df_temp["DayofYear"] = df_temp.Date.dt.dayofyear
            df_temp["HourofYear"] = np.arange(1, len(df_temp)+1)
            df_temp[year] = df_temp["energy_cost_enduse"].values
            df = df_temp[["Month", "Day", "Hour", "DayofYear", "HourofYear", year]].copy()
        else:
            df[year] = df_temp["energy_cost_enduse"].values

    for i, year in enumerate(span):

        if year in avail_years:
            continue

        if any(year < avail_years):
            upper_years = avail_years[avail_years > year]
            lower_years = avail_years[avail_years < year]
            upper_year = upper_years[np.argmin(np.abs(year - upper_years))]
            lower_year = lower_years[np.argmin(np.abs(lower_years - year))]
            assert (year < upper_year) and (year > lower_year), "Issue with lower/upper year allocation ..."

            dpdy = (df[upper_year] - df[lower_year])/(upper_year - lower_year)
            
        df[year] = df[lower_year] + dpdy * (year - lower_year)
    
    for i, year in enumerate(span):
        df[year] =  df[year] * (1+infl)**(year - base_year)
        
        daily_price_means = np.zeros(len(df))
        for i in range(int(len(daily_price_means)/24)):
            daily_price_means[24*i:24*i+24] = df[year][24*i:24*i+24].mean()
        
        seasonal_price_means = np.zeros(len(df))
        for i in range(int(len(seasonal_price_means)/8760)):
            seasonal_price_means[8760*i:8760*i+8760] = df[year][8760*i:8760*i+8760].mean()
            
        df[year] += daily_fat_factor * (df[year] - daily_price_means) + \
                    seasonal_fat_factor * (df[year] - seasonal_price_means)
    
    df_melt = pd.melt(df, id_vars=["Hour", "Day", "Month"], value_vars=span,var_name="Year", value_name="price")
    df_melt["Date"] = pd.to_datetime(df_melt[["Year","Month","Day","Hour"]])
    df_final = pd.merge(df_melt, df_recs, how="inner", on="Date")[["Date", "Hour", "price", "recs_price"]]
    
    baseload_score = df_final.price.mean() #$/MWh
    arbitrage_score = df_final.groupby("Hour").mean(numeric_only=True)["price"].max() - \
    df_final.groupby("Hour").mean(numeric_only=True)["price"].min() #$/MWh
    scores = {"baseload": round(baseload_score,2),
              "arbitrage": round(arbitrage_score,2)}
    if plot:
        viz_wholesale(df_final, span=span[::2])
    
    df_final["Date"] = df_final.Date.dt.strftime("%m/%d/%y %H:%M")
    if save:
        df_final.to_csv(os.path.join(dst_dir, "DA.csv"), index=False)
    
    return df_final, df, scores

def preprocess_cambium_elcc(scenario, state, span=range(2024,2061), plot=False, dst_dir="../../Data/dummy", save=False):
    
    """Preprocess and plot Cambium ELCC data for a chosen state and forecast scenario."""
    
    filepath = f"../../Cambium_2022/{scenario}/Cambium22_{scenario}_annual_state.csv"
    df = pd.read_csv(filepath, header=5)
    df = df.loc[df.state == state, ["t", "battery_MW"]]
    df = df.rename(columns={"t": "Year", "battery_MW": "battery_GW"})
    df["battery_GW"] = df["battery_GW"]/1e3
    df_full = pd.merge(pd.DataFrame({"Year": span}), df, how='outer', on="Year").interpolate(method="linear", order=1)

    # based on Figure 20 in https://www.ethree.com/wp-content/uploads/2019/06/E3_Long_Run_Resource_Adequacy_CA_Deep-Decarbonization_Final.pdf
    d = {0: 1, 5:0.80, 10: 0.60, 15: 0.40, 20:0.35, 25: 0.30, 30: 0.20, 40:0.11, 45:0.10, 55: 0.06, 75:0.05}
    df_elcc = pd.DataFrame([d.keys(), d.values()]).T.rename(columns={0:"battery_GW", 1: "elcc"})
    df_elcc = pd.merge(pd.DataFrame({"battery_GW": range(0, 101)}), df_elcc, on="battery_GW", how="outer").interpolate(method="linear")

    df_full["elcc"] = df_full["battery_GW"].apply(lambda x: df_elcc.loc[df_elcc["battery_GW"]==int(x), "elcc"].iloc[0])

    bat_filepath = os.path.join(dst_dir, "battery_costs.csv")
    df_bat = pd.read_csv(bat_filepath)
    df_bat.drop(columns="elcc", inplace=True)
    df_bat = pd.merge(df_full, df_bat, on="Year", how="left")
    df_bat.drop(columns=[i for i in df_bat.columns if "battery_GW" in i], inplace=True)
    
    if save:
        df_bat.to_csv(bat_filepath, index=False)
    
    if plot:
        df_bat.plot(x="Year", y="elcc", legend=False)
        plt.ylabel("ELCC [fraction]")
        
    return df_bat

def viz_trial_logs(trial_dir, quantities, span=2000, figsize=(10,10)):
    
    """Visualize logs produced from an Rllib training session."""
    
    df = pd.read_json(os.path.join(trial_dir, "result.json"), lines=True)[:span] 
    fig, ax = plt.subplots(len(quantities), 1, figsize=figsize, sharex=True)

    for i, (q, label) in enumerate(quantities.items()):
        data = df["custom_metrics"].apply(lambda x: dict(x).get(q+"_mean")).values
        ax[i].plot(range(len(df)), data, label=label, color=colors[i])
        ax[i].legend([q], loc="upper right")
        if q in ["NPV", "ROI"]:
            ax[i].set_ylim([0, None])
    plt.show()
    
def compute_f(Rewaterprod, well_diam):
    
    """Compute f3 constant to be used for determining Reynold's number."""

    if Rewaterprod < 2300.:
        f = 64./nonzero(Rewaterprod)
    else:
        relroughness = 1e-4/well_diam
        f = 1./np.power(-2*np.log10(relroughness/3.7+5.74/np.power(Rewaterprod,0.9)),2.)
        f = 1./np.power((-2*np.log10(relroughness/3.7+2.51/Rewaterprod/np.sqrt(f))),2.)
        f = 1./np.power((-2*np.log10(relroughness/3.7+2.51/Rewaterprod/np.sqrt(f))),2.)
        f = 1./np.power((-2*np.log10(relroughness/3.7+2.51/Rewaterprod/np.sqrt(f))),2.)
        f = 1./np.power((-2*np.log10(relroughness/3.7+2.51/Rewaterprod/np.sqrt(f))),2.)
        f = 1./np.power((-2*np.log10(relroughness/3.7+2.51/Rewaterprod/np.sqrt(f))),2.) #6 iterations to converge
    return f

# def compute_f1(Rewaterinj, well_diam):
    
#     """Compute f1 constant to be used for determining Reynold's number."""
    
#     if Rewaterinj < 2300. : #laminar flow
#         f1 = 64./nonzero(Rewaterinj)
#     else: #turbulent flow
#         relroughness = 1e-4/well_diam
#         f1 = 1./np.power(-2*np.log10(relroughness/3.7+5.74/np.power(Rewaterinj,0.9)),2.)
#         f1 = 1./np.power((-2*np.log10(relroughness/3.7+2.51/Rewaterinj/np.sqrt(f1))),2.)
#         f1 = 1./np.power((-2*np.log10(relroughness/3.7+2.51/Rewaterinj/np.sqrt(f1))),2.)
#         f1 = 1./np.power((-2*np.log10(relroughness/3.7+2.51/Rewaterinj/np.sqrt(f1))),2.)
#         f1 = 1./np.power((-2*np.log10(relroughness/3.7+2.51/Rewaterinj/np.sqrt(f1))),2.)
#         f1 = 1./np.power((-2*np.log10(relroughness/3.7+2.51/Rewaterinj/np.sqrt(f1))),2.)  #6 iterations to converge 
#     return f1

def densitywater(Twater): 
    
    """Correlation for water density based on the GEOPHIRES tool."""
    
    # Based on GEOPHIRES correlations (more stable than XSTEAM)  
    T = Twater+273.15
    rhowater = ( .7983223 + (1.50896E-3 - 2.9104E-6*T) * T) * 1E3 #water density correlation as used in Geophires v1.2 [kg/m3]
    return  rhowater

def viscositywater(Twater):
    
    """Correlation for water viscosity based on the GEOPHIRES tool."""
    
    # Based on GEOPHIRES correlations (more stable than XSTEAM)
    muwater = 2.414E-5*np.power(10,247.8/(Twater+273.15-140))     #accurate to within 2.5% from 0 to 370 degrees C [Ns/m2]
    #xp = np.linspace(5,150,30)
    #fp = np.array([1519.3, 1307.0, 1138.3, 1002.0, 890.2, 797.3, 719.1, 652.7, 596.1, 547.1, 504.4, 467.0, 433.9, 404.6, 378.5, 355.1, 334.1, 315.0, 297.8, 282.1, 267.8, 254.4, 242.3, 231.3, 221.3, 212.0, 203.4, 195.5, 188.2, 181.4])
    #muwater = np.interp(Twater,xp,fp)
    return muwater

def heatcapacitywater(Twater):
    
    """Correlation for water heat capacity based on the GEOPHIRES tool."""
    
    # Based on GEOPHIRES correlations (more stable XSTEAM)
    Twater = (Twater + 273.15)/1000
    A = -203.6060
    B = 1523.290
    C = -3196.413
    D = 2474.455
    E = 3.855326
    cpwater = (A + B*Twater + C*Twater**2 + D*Twater**3 + E/(Twater**2))/18.02*1000 #water specific heat capacity in J/kg-K
    return cpwater

def vaporpressurewater(Twater): 
    
    """Correlation for water vapor pressure based on the GEOPHIRES tool."""
    
    if Twater < 100:
        A = 8.07131
        B = 1730.63
        C = 233.426
    else:
        A = 8.14019
        B = 1810.94
        C = 244.485 
    vaporpressurewater = 133.322*(10**(A-B/(C+Twater)))/1000 #water vapor pressure in kPa using Antione Equation
    return vaporpressurewater
    

def compute_npv(df_records, capex_total, opex_total, baseline_year, L, d, ppa_price=75):
    
    """Compute NPV and other economic metrics for a completed simulation run."""
    
    years = np.arange(L)
    df_records["PPA Revenue [$MM]"] = df_records["Net Power Generation [MWhe]"]*ppa_price/1e6
    df_annual = df_records.groupby('Year').sum(numeric_only=True)
    df_annual["CAPEX [$MM]"] = capex_total
    df_annual["OPEX [$MM]"] = opex_total

    df_annual["Cashin [$MM]"] = df_annual["Revenue [$MM]"]
    df_annual["PPA Cashin [$MM]"] = df_annual["PPA Revenue [$MM]"]
    df_annual["Cashout [$MM]"] = df_annual["OPEX [$MM]"] + df_annual["CAPEX [$MM]"]
    df_annual["Cashflow [$MM]"] = df_annual["Cashin [$MM]"] - df_annual["Cashout [$MM]"]
    df_annual["PPA Cashflow [$MM]"] = df_annual["PPA Cashin [$MM]"] - df_annual["Cashout [$MM]"]
    
    df_annual = df_annual.div((1 + d)**years, axis=0)
    df_annual["NPV [$MM]"] = df_annual["Cashflow [$MM]"].cumsum()
    df_annual["PPA NPV [$MM]"] = df_annual["PPA Cashflow [$MM]"].cumsum()
    df_annual["Revenue [$MM]"] = df_annual["Cashin [$MM]"].cumsum()
    df_annual["Cost [$MM]"] = df_annual["Cashout [$MM]"].cumsum()
    df_annual["Cum CAPEX [$MM]"] = df_annual["CAPEX [$MM]"].cumsum()
    df_annual["Cum OPEX [$MM]"] = df_annual["OPEX [$MM]"].cumsum()
    df_annual["ROI [%]"] = df_annual["Cashflow [$MM]"].cumsum()/df_annual["CAPEX [$MM]"].cumsum() * 100
    df_annual["PPA ROI [%]"] = df_annual["PPA Cashflow [$MM]"].cumsum()/df_annual["CAPEX [$MM]"].cumsum() * 100
    df_annual['Res Temp [deg C]'] = df_records.groupby('Year').mean(numeric_only=True)["Res Temp [deg C]"]
    df_annual['WH Temp [deg C]'] = df_records.groupby('Year').mean(numeric_only=True)["WH Temp [deg C]"]
    NPV = df_annual["NPV [$MM]"].iloc[-1]
    ROI = df_annual["ROI [%]"].iloc[-1]
    PBP = df_annual.index[np.argmax((df_annual["ROI [%]"]>0).values)] - baseline_year
    PPA_NPV = df_annual["PPA NPV [$MM]"].iloc[-1]
    PPA_ROI = df_annual["PPA ROI [%]"].iloc[-1]
    PPA_PBP = df_annual.index[np.argmax((df_annual["PPA ROI [%]"]>0).values)] - baseline_year
    LCOE = df_annual["Cashout [$MM]"].sum()*1e6/nonzero(df_annual["Net Power Generation [MWhe]"].sum(), 1E-1)
    if LCOE < 0: # cases where pumping requirements are greater than gross power generation
        LCOE = 999
    return NPV, ROI, PBP, PPA_NPV, PPA_ROI, PPA_PBP, LCOE, df_annual

def compute_drilling_cost(well_depth, well_diam):
    
    """Correlations for computing drilling costs."""
    
    # Divide by 1.15 to remove th 15% contiengency already included by SNL (ref: GETEM page 85)
    if well_diam > 0.3: # units in meters
        return (0.2818*well_depth**2 + 1275.5213*well_depth + 632315.)*1e-6/1.15
    else:
        return (0.3021*well_depth**2 + 584.9112*well_depth + 751368.)*1e-6/1.15

def compute_latlon_distance(lat1, lon1, lat2, lon2):
    
    """Compute Euclidean distance in kilometers based on latlon cooredinates of two locations."""
    
    return np.arccos(np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1))*6371 #kilometers

def egs_mw_from_temp(T):
    
    """Find GETEM default for power plant size in MW given resource temperature."""
    
    if T < 140:
        return 10
    elif T < 175:
        return 15
    elif T < 250:
        return 25
    else:
        return 30
    
def gpd_geographic_area(geodf):
    
    """Compute the geographic area of latlon polygons in a geopandas df."""
    
    if not geodf.crs and geodf.crs.is_geographic:
        raise TypeError('geodataframe should have geographic coordinate system')
        
    geod = geodf.crs.get_geod()
    def area_calc(geom):
        if geom.geom_type not in ['MultiPolygon','Polygon']:
            return np.nan
        
        # For MultiPolygon do each separately
        if geom.geom_type=='MultiPolygon':
            return np.sum([area_calc(p) for p in geom.geoms])

        # orient to ensure a counter-clockwise traversal. 
        # See https://pyproj4.github.io/pyproj/stable/api/geod.html
        # geometry_area_perimeter returns (area, perimeter)
        return geod.geometry_area_perimeter(orient(geom, 1))[0]
    
    return geodf.geometry.apply(area_calc)

def latlon_tres_to_depth(df_maps, query_lat, query_lon, tres, MJth_per_km3=5.1e9, eta=0.1, L=25):
    
    """Get depth required to reach a target temperature at a latlon location."""
    
    all_depths = np.array(list(df_maps.keys()))
    depths = all_depths
    temps = []
    for depth in all_depths:
        df_temp = df_maps[depth]
        row = df_temp.iloc[[(np.sqrt((query_lat - df_temp["centroid_lat"])**2 + (query_lon - df_temp["centroid_lon"])**2)).argmin()]]
        temp = row["T"].values[0]
        temps.append(temp)
        if temp >= tres:
            depths = all_depths[:len(temps)]
            break
    temps = np.array(sorted(temps))
    surface_temp = temps[0]
    geothermal_gradient = np.mean(np.clip(np.diff(temps[[0,-1]]) / (np.diff(depths[[0,-1]])/1000), 0.0, np.inf))
    if geothermal_gradient <= 0.0:
        print(temps, depths, tres, temp)
        print("Non-Positive Geothermal Gradient ... !!!")

    if tres > max(temps):
        z = np.polyfit(temps, all_depths, 1)
        p = np.poly1d(z)
        well_depth = p(tres)
    else:
        well_depth = np.interp(tres, temps, depths)
    
    area = gpd_geographic_area(row).values[0]/1e6
    electric_energy_MJe = MJth_per_km3 * area * eta
    electric_power_MWe = electric_energy_MJe/(L*365*24*3600)
    
    return well_depth, surface_temp, geothermal_gradient, electric_power_MWe

def nonzero(x, thresh=1E-6):
    return np.maximum(thresh, x)

if __name__ == "__main__":
    pass
