import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pdb 
from pyXSteam.XSteam import XSteam
from scipy.optimize import curve_fit
import pickle
from matplotlib.ticker import FormatStrFormatter
from shapely.geometry.polygon import orient
from shapely.geometry import Polygon
from shapely.prepared import prep
import geopandas as gpd
from timezonefinder import TimezoneFinder
from meteostat import Hourly, Stations
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import StrMethodFormatter

steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS) # m/kg/sec/°C/bar/W

colors = 24*sns.color_palette()

linestyles = {0: 'solid',
              1: 'dashed',
              2: 'dashdot',
              3: 'dotted'}


class FastXsteam(object):
    
    """Faster corrleations for Steam Tables."""
    
    def __init__(self, T_max=300, steps=30000):
        """Initiate class for fast Xsteam tables.

        Args:
            T_max (int, optional): maximum temperature. Defaults to 300.
            steps (int, optional): number of steps. Defaults to 30000.
        """
        
        self.xdata = np.linspace(1, T_max, steps)
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
        """Liquid enthalpy correlation.

        Args:
            t (float): temperature in C
            a (float): correlation coefficient.
            b (float): correlation coefficient.
            c (float): correlation coefficient.
            d (float): correlation coefficient.
            e (float): correlation coefficient.

        Returns:
            float: liquid enthalpy in kJ/kg at the given temperature
        """
        tr = (t+273.15)/self.tc
        return (a + b * np.log(1/tr)**0.35 + c/tr**2 + d/tr**3 + e/tr**4)

    def func_hv(self, t, a, b, c, d, e):
        """Vapor enthalpy correlation.

        Args:
            t (float): temperature in C
            a (float): correlation coefficient.
            b (float): correlation coefficient.
            c (float): correlation coefficient.
            d (float): correlation coefficient.
            e (float): correlation coefficient.

        Returns:
            float: vapor enthalpy in kJ/kg at the given temperature
        """

        tr = (t+273.15)/self.tc
        return (a + b * np.log(1/tr)**0.35 + c/tr**2 + d/tr**3 + e/tr**4)**(1/2)

    def func_vl(self, t, a, b, c, d, e):
        """Liquid specific volume correlation.

        Args:
            t (float): temperature in C
            a (float): correlation coefficient.
            b (float): correlation coefficient.
            c (float): correlation coefficient.
            d (float): correlation coefficient.
            e (float): correlation coefficient.

        Returns:
            float: liquid specific volume in m3/kg at the given temperature
        """

        tr = (t+273.15)/self.tc
        return (a + b * np.log(1/tr)**0.35 + c/tr**2 + d/tr**3 + e/tr**4)

    def func_vv(self, t, a, b, c, d, e):
        """Vapor specific volume correlation.

        Args:
            t (float): temperature in C
            a (float): correlation coefficient.
            b (float): correlation coefficient.
            c (float): correlation coefficient.
            d (float): correlation coefficient.
            e (float): correlation coefficient.

        Returns:
            float: vapor specific volume in m3/kg at the given temperature
        """

        tr = (t+273.15)/self.tc
        return np.exp((a + b * np.log(1/tr)**0.35 + c/tr**2 + d/tr**3 + e/tr**4))

    def hL_t(self, t):
        """Liquid enthalpy correlation.

        Args:
            t (float): temperature in C

        Returns:
            float: liquid enthalpy in kJ/kg at the given temperature
        """
        t = t[0] if isinstance(t, np.ndarray) else t
        return self.func_hl(t, *self.popt_hl)
        
    def hV_t(self, t):
        """Vapor enthalpy correlation.

        Args:
            t (float): temperature in C

        Returns:
            float: vapor enthalpy in kJ/kg at the given temperature
        """

        t = t[0] if isinstance(t, np.ndarray) else t
        return self.func_hv(t, *self.popt_hv)

    def vL_t(self, t):
        """Liquid specific volume correlation.

        Args:
            t (float): temperature in C

        Returns:
            float: liquid specific volume in m3/kg at the given temperature
        """

        t = t[0] if isinstance(t, np.ndarray) else t
        return self.func_vl(t, *self.popt_vl)
    
    def vV_t(self, t):
        """Vapor specific volume correlation.

        Args:
            t (float): temperature in C

        Returns:
            float: vapor specific volume in m3/kg at the given temperature
        """

        t = t[0] if isinstance(t, np.ndarray) else t
        return self.func_vv(t, *self.popt_vv)

def plot_ex(worlds,
            figsize=(10,10),
            dpi=150,
            colors=colors,
            fontsize=10
               ):
    """Visualize expenditure of multiple projects. 

    Args:
        worlds (dict): projects or worlds to visualize
        figsize (tuple, optional): figure size. Defaults to (10,10).
        dpi (int, optional): figure dpi resolution. Defaults to 150.
        colors (list, optional): list of colors to be used. Defaults to seaborn.color_palette().
        fontsize (int, optional): font size. Defaults to 10.

    Returns:
        matplotlib.figure.Figure: figure object
    """
    
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
                                   pctdistance=0.8,
                                   labels=labels, 
                                   colors=colors, 
                                   autopct=lambda pct: func(pct, ex),
                                   textprops=dict(color="w", weight="bold", fontsize=fontsize))

        axes[i].legend(wedges, labels,
                title=title,
                loc="center left",
                bbox_to_anchor=(0.3, 0.6, 0.0, 1))

    return fig

def plot_cols(dfs,
              span,
              quantities, 
              figsize=(10,10),
              xlabel="World Time",
              ylabels=None,
              ylogscale=False,
              use_title=True,
              legend_loc="lower right",
              manual_legends=None,
              color_per_col=True,
              use_linestyles=True,
              blackout_first=False,
              formattime = False,
              grid=True,
              dpi=100):
    """Visualize columns of a dataframe.

    Args:
        dfs (dict): dictionary of dataframes to plot with key as names.
        span (Union[range, list]): range of indices to plot.
        quantities (list): dataframe columns to plot.
        figsize (tuple, optional): figure size. Defaults to (10,10).
        xlabel (str, optional): label of the x axis. Defaults to "World Time".
        ylabels (list, optional): labels of the y axes. Defaults to None.
        ylogscale (bool, optional): whether or not to use log-scale on the y axis. Defaults to False.
        use_title (bool, optional): whether or not to use a figure title. Defaults to True.
        legend_loc (str, optional): legend location. Defaults to "lower right".
        manual_legends (list, optional): list to override default legends. Defaults to None.
        color_per_col (bool, optional): whether or not to assign one color to each column. Defaults to True.
        use_linestyles (bool, optional): whether or not to use linestyles. If False, all quantities are plotted with solid linestyle. Defaults to True.
        blackout_first (bool, optional): whether or not to plot the very first quantity in black color. Defaults to False.
        formattime (bool, optional): whether or not to apply time formatting. Defaults to False.
        dpi (int, optional): figure dpi resolution. Defaults to 100.

    Returns:
        matplotlib.figure.Figure: figure object

    """

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
            
            axes[i].yaxis.tick_right()

            if grid:
                axes[i].grid(color='lightgray')

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

    return fig

def viz_wholesale(filepath, span=range(2025,2056,2)):
    """Visualize wholesale market data.

    Args:
        filepath (str): file path to csv file.
        span (range, optional): range or list of years to visualize. Defaults to range(2025,2056,2).
    """

    
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
                                plot=False, dst_dir='.', save=None, adjustment=0,
                                states_with_no_capacity_market = ["TX"]):
    """Preprocess and plot Cambium capacity market data for a chosen state and forecast scenario"""
    
    filepath = f"../../Cambium_2022/{scenario}/Cambium22_{scenario}_annual_state.csv"
    df = pd.read_csv(filepath, header=5)
    df = df.loc[df.state == state, ["t", "capacity_cost_enduse"]]
    df = df.rename(columns={"t": "Year", "capacity_cost_enduse": "capacity cost"})
    df["capacity cost"] *= (8760/1e3) # to convert $/MWh to $/kW-yr
    
    if state in states_with_no_capacity_market:
        df["capacity cost"] = 0.0
        
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
    """Compute Darcy friction coefficient.

    Args:
        Rewaterprod (Union[ndarray, float]): Reynold's number
        well_diam (Union[ndarray, float]): well diameter

    Returns:
        Union[ndarray, float]: Darcy friction factor
    """

    relroughness = 1e-4/well_diam
    f = 1./np.power(-2*np.log10(relroughness/3.7+5.74/np.power(Rewaterprod,0.9)),2.)
    f = 1./np.power((-2*np.log10(relroughness/3.7+2.51/Rewaterprod/np.sqrt(f))),2.)
    f = 1./np.power((-2*np.log10(relroughness/3.7+2.51/Rewaterprod/np.sqrt(f))),2.)
    f = 1./np.power((-2*np.log10(relroughness/3.7+2.51/Rewaterprod/np.sqrt(f))),2.)
    f = 1./np.power((-2*np.log10(relroughness/3.7+2.51/Rewaterprod/np.sqrt(f))),2.)
    f = 1./np.power((-2*np.log10(relroughness/3.7+2.51/Rewaterprod/np.sqrt(f))),2.) #6 iterations to converge
    
    # if laminar
    f = np.where(Rewaterprod < 2300., 64./nonzero(Rewaterprod), f)
    
    return f

def densitywater(Twater):
    """Computationally efficient correlation for water density based on GEOPHIRES.

    Args:
        Twater (Union[ndarray, float]): water temperature in deg C

    Returns:
        Union[ndarray, float]: water density in kg/m3
    """
    
    T = Twater+273.15
    rhowater = ( .7983223 + (1.50896E-3 - 2.9104E-6*T) * T) * 1E3 #water density correlation as used in Geophires v1.2 [kg/m3]
    return  rhowater

def viscositywater(Twater):
    """Computationally efficient correlation for water viscosity based on the GEOPHIRES.

    Args:
        Twater (Union[ndarray, float]): water temperature in [deg C]

    Returns:
        Union[ndarray, float]: water dynamic viscosity in [Ns/m2] or [Pa . s]
    """

    muwater = 2.414E-5*np.power(10,247.8/(Twater+273.15-140))     #accurate to within 2.5% from 0 to 370 degrees C [Ns/m2] or [Pa . s]
    return muwater

def heatcapacitywater(Twater):
    
    """Computationally efficient correlation for water specific heat capacity based on the GEOPHIRES.

    Args:
        Twater (Union[ndarray, float]): water temperature in deg C

    Returns:
        Union[ndarray, float]: water specific heat capacity in J/kg-K
    """

    # Based on GEOPHIRES correlations (more stable XSTEAM)
    Twater = (Twater + 273.15)/1000
    A = -203.6060
    B = 1523.290
    C = -3196.413
    D = 2474.455
    E = 3.855326
    cpwater = (A + B*Twater + C*Twater**2 + D*Twater**3 + E/(Twater**2))/18.02*1000
    return cpwater

def vaporpressurewater(Twater): 
    """Computationally efficient correlation for vapor pressure based on the GEOPHIRES.

    Args:
        Twater (Union[ndarray, float]): water temperature in deg C

    Returns:
        Union[ndarray, float]: vapor pressure in kPa
    """
    
    return np.where(Twater < 100, 133.322*(10**(8.07131-1730.63/(233.426+Twater)))/1000, 
         133.322*(10**(8.14019-1810.94/(244.485 +Twater)))/1000)

def compute_drilling_cost(well_tvd, well_diam, lateral_length=0, numberoflaterals=1,
                          total_drilling_length=None, drilling_cost=None):
    
    """Correlations for computing drilling costs."""
    well_md = well_tvd + lateral_length * numberoflaterals

    if total_drilling_length is not None:
        if not drilling_cost:
            raise ValueError('You must specify drilling drilling_cost for the provided system design.')
        else:
            return total_drilling_length * drilling_cost / 1e6

    if drilling_cost:
        return well_md * drilling_cost / 1e6
    # Divide by 1.15 to remove the 15% contiengency already included by SNL (ref: GETEM page 85)
    if well_diam > 0.3: # units in meters
        if lateral_length>0:
            return (0.2553*well_md**2 + 1716.7157*well_md + 500867.)*1e-6/1.15
        else:
            return (0.2818*well_tvd**2 + 1275.5213*well_tvd + 632315.)*1e-6/1.15
    else:
        if lateral_length>0:
            return (0.2898*well_md**2 + 822.1507*well_md + 680563.)*1e-6/1.15
        else:
            return (0.3021*well_tvd**2 + 584.9112*well_tvd + 751368.)*1e-6/1.15

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
        return geod.geometry_area_perimeter(orient(geom, 1))[0] #m2
    
    return geodf.geometry.apply(area_calc)

def latlon_tres_to_depth(df_maps, query_northing, query_easting, tres, MJth_per_km3=5.1e9, thickness=500, eta=0.1, L=25):
    
    """Get depth required to reach a target temperature at a latlon location."""
    
    all_depths = np.array(list(df_maps.keys()))
    depths = all_depths
    temps = []
    for depth in all_depths:
        df_temp = df_maps[depth]
        row = df_temp.iloc[[(np.sqrt((query_northing - df_temp["Northing"])**2+ (query_easting - df_temp["Easting"])**2)).argmin()]]
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
        z = np.polyfit(temps, depths, 1)
        p = np.poly1d(z)
        well_depth = p(tres)
    else:
        well_depth = np.interp(tres, temps, depths)

    A_r = gpd_geographic_area(row).values[0]/1e6 #km2
    V_r = A_r * thickness/1000 # km3
    electric_energy_MJe = MJth_per_km3 * V_r * eta
    electric_power_MWe = electric_energy_MJe/(L*365*24*3600)
    
    return well_depth, surface_temp, geothermal_gradient, electric_power_MWe, A_r, V_r

def latlon_depth_to_tres(df_maps, query_northing, query_easting, well_depth, MJth_per_km3=5.1e9, thickness=1000, eta=0.1, L=25):
    
    """Get depth required to reach a target temperature at a latlon location."""
    
    all_depths = np.sort(list(df_maps.keys()))
    ref_depth = all_depths[np.argmin(np.abs(np.array(list(df_maps.keys())) - well_depth))]
    df_temp = df_maps[ref_depth]
    ref_row = df_temp.iloc[[(np.sqrt((query_northing - df_temp["Northing"]).abs() + (query_easting - df_temp["Easting"]).abs())).argmin()]]
    
    depths = all_depths[all_depths <= well_depth]
    temps = []
    for depth in depths:
        df_temp = df_maps[depth]
        row = df_temp.iloc[[(np.sqrt((query_northing - df_temp["Northing"]).abs() + (query_easting - df_temp["Easting"]).abs())).argmin()]]
        temp = row["T"].values[0]
        temps.append(temp)
    
    temps = np.array(sorted(temps))
    surface_temp = temps[0]
    geothermal_gradient = np.mean(np.clip(np.diff(temps[[0,-1]]) / (np.diff(depths[[0,-1]])/1000), 0.0, np.inf))
    if geothermal_gradient <= 0.0:
        print(temps, depths, tres, temp)
        print("Non-Positive Geothermal Gradient ... !!!")
        
    z = np.polyfit(depths[-2:], temps[-2:], 1)
    p = np.poly1d(z)
    tres = p(well_depth)

    A_r = gpd_geographic_area(ref_row).values[0]/1e6 #km2
    V_r = A_r * thickness/1000 # km3
    electric_energy_MJe = MJth_per_km3 * V_r * eta
    electric_power_MWe = electric_energy_MJe/(L*365*24*3600)

    
    return tres, surface_temp, geothermal_gradient, electric_power_MWe, A_r, V_r

def augustine_MWeperkm3(tres):
    """Based on page22 of Augustine (2011): https://www.nrel.gov/docs/fy12osti/47459.pdf"""
    if tres <= 200:
        return 0.59
    elif tres <= 250:
        return 0.76
    elif tres <= 300:
        return 0.86
    elif tres <= 350:
        return 0.97
    else:
        return 1.19
    
def augustine_eff(tres):
    """Based on page22 of Augustine (2011): https://www.nrel.gov/docs/fy12osti/47459.pdf"""
    if tres <= 200:
        return 0.11
    elif tres <= 250:
        return 0.14
    elif tres <= 300:
        return 0.16
    elif tres <= 350:
        return 0.18
    else:
        return 0.22

def nonzero(x, thresh=1E-6):
    """Ensure input is nonzero

    Args:
        x (ndarray): input numbers
        thresh (float, optional): threshold below which numbers are set to zero. Defaults to 1E-6.

    Returns:
        ndarray: nonzero version of input array
    """
    return np.maximum(thresh, x)

def grid_bounds(geom, delta):
    """Create a grid/mesh based on geometry."""
    minx, miny, maxx, maxy = geom.bounds
    nx = int((maxx - minx)/delta)
    ny = int((maxy - miny)/delta)
    gx, gy = np.linspace(minx,maxx,nx), np.linspace(miny,maxy,ny)
    grid = []
    for i in range(len(gx)-1):
        for j in range(len(gy)-1):
            poly_ij = Polygon([[gx[i],gy[j]],[gx[i],gy[j+1]],[gx[i+1],gy[j+1]],[gx[i+1],gy[j]]])
            grid.append( poly_ij )
    return grid

def partition(geom, delta):
    """Partition a geometry based on delta value."""
    prepared_geom = prep(geom)
    grid = list(filter(prepared_geom.intersects, grid_bounds(geom, delta)))
    return grid
        
def harrison(z):
    """Harrison temperature correction.

    Args:
        z (ndarray): true vertical depth in meters

    Returns:
        ndarray: harrison correction in deg C
    """
    return -2.3449e-6 * z**2 + 0.018268 * z - 16.512

def forster(z):
    """Forster temperature correction.

    Args:
        z (ndarray): true vertical depth in meters

    Returns:
        ndarray: forster correction in deg C
    """
    return 0.017*z - 6.58

def retrieve_weather(lat, lon, year, station_idx_limit=10):
    """Retrieve weather."""

    # Set time period
    findtz = TimezoneFinder()
    findstations = Stations()
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31, 23, 59)

    data = []
    station_idx = 0
    stations = findstations.nearby(lat, lon).fetch(station_idx_limit+1)

    while len(data) != 8760:
        station = stations.iloc[[station_idx]]
        data = Hourly(station, start-timedelta(days=2), end+timedelta(days=2)) #get more days to account for timezone shifting
        data = data.fetch()
        data = data.bfill().ffill()
        data.reset_index(inplace=True)
        tz = findtz.timezone_at(lng=lon, lat=lat)
        if len(data):
            data["time"] = data.time.dt.tz_localize('utc').dt.tz_convert(tz).dt.tz_localize(None)
            year_col = data.time.dt.year
            data = data[year_col == year].reset_index(drop=True)

        if station_idx >= station_idx_limit:
            print(state, f"##### {len(data)} #####")
            break
        station_idx += 1
    
    return data

class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        inv_distances = 1/(distances + 1e-3)**4
        inv_distances = inv_distances/inv_distances.sum(axis=1, keepdims=True)
        predictions = (inv_distances[:,:,None] * self.y[indices]).sum(axis=1)

        return predictions

if __name__ == "__main__":
    pass
