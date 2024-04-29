import pandas as pd
import numpy as np
import pdb
from datetime import timedelta
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class TabularPowerMarket:

    """Tabular power markets class."""

    def __init__(self):
        """Initating attributes."""
        pass

    def create_energy_market(self,
                            filepath=None,
                            energy_price=40,
                            recs_price=30,
                            L=30,
                            time_init=pd.to_datetime('today'),
                            resample=None,
                            fat_factor=1,
                            fat_window=24,
                                ):
        """Loading and processnig wholesale market data.

        Args:
            filepath (str, optional): If available, csv filepath to market data. Defaults to None.
            energy_price (float, optional): energy price in USD/MWh. Defaults to 40.
            recs_price (float, optional): renewable energy credits (RECs) price in USD/MWh. Defaults to 30.
            L (int, optional): project lifetime in years. Defaults to 30.
            time_init (datetime, optional): project start date. Defaults to pd.to_datetime('today').
            resample (str, optional): timeframe to which project market data is resampled (Options: "1Y", "1m", "1w", "1d", "1h" for yearly, monthly, weekly, daily, or hourly timestepping). Defaults to False.
            fat_factor (float, optional): multiplier of mean diversion to make the arbitrage opportunity more pronounced. Defaults to 1.
            fat_window (int, optional): window considered when applying mean diversion to make the arbitrage opportunity more pronounced. Defaults to 24.
        """

        if filepath:
            self.df = pd.read_csv(filepath)
            try:
                self.df.Date = pd.to_datetime(self.df.Date, format="%m/%d/%y %H:%M")
            except:
                print("Slow formating of datetime while loading of wholesale market data ... ")
                self.df.Date = pd.to_datetime(self.df.Date)
        else:
            self.df = time_init + pd.DataFrame(np.cumsum(L * 8760 * [timedelta(hours=1)]), columns=["Date"])
            # self.df.set_index("Date", inplace=True)
            self.df["price"] = energy_price
            self.df["recs_price"] = 30
        
        if resample:
            self.resample = resample
            self.df = self.df.resample(rule=resample, on='Date').mean()
        
        self.df.reset_index(inplace=True)
        self.df['TimeDiff'] = self.df.Date.diff().bfill()
        self.df['TimeDiff_seconds'] = self.df.TimeDiff.apply(lambda x: x.seconds)

        self.df['year'] = self.df.Date.apply(lambda x: x.year)
        self.df['month'] = self.df.Date.apply(lambda x: x.month)
        self.df['day'] = self.df.Date.apply(lambda x: x.day)
        self.df['hour'] = self.df.Date.apply(lambda x: x.hour)
        self.df['minute'] = self.df.Date.apply(lambda x: x.minute)
        self.df['dayofyear'] = self.df.Date.apply(lambda x: x.dayofyear)

        # Drop Feb 29 rows that could appear from resampling over leap years
        self.df = self.df[~((self.df.month == 2) & (self.df.day == 29))]

        self.df.set_index('Date', inplace=True)

        if self.df.isna().sum().max() > 10:
            warnings.warn("WARNING: Too many missing market data ... ")
        self.df.ffill(inplace=True)

        self.df["price_raw"] = self.df["price"]

        if fat_factor > 1:
            price_means = np.zeros(len(self.df))
            for i in range(int(len(price_means)/fat_window)):
                price_means[fat_window*i:fat_window*i+fat_window] = \
                    self.df["price_raw"][fat_window*i:fat_window*i+fat_window].mean()
        
            self.df["price"] = self.df["price_raw"] + fat_factor * (self.df["price_raw"] - price_means)

    def create_capacity_market(self,
                                filepath=None,
                                capacity_price=100,
                                col="capacity cost",
                                convert_to_usd_per_mwh=True):
        """Load and process capacity market data.

        Args:
            filepath (str, optional): csv filepath to capacity market data, if available. Defaults to None.
            capacity_price (float, optional): capacity price in USD/kW-year. Defaults to 100.
            col (str, optional): column header to be used for capacity price in the created dataframe. Defaults to "capacity cost".
            convert_to_usd_per_mwh (bool, optional): whether or not to convert the capacity price from USD/kW-year to USD/MWh. Defaults to True.
        """

        if filepath:
            self.df_capacity = pd.read_csv(filepath)
            if convert_to_usd_per_mwh:
                self.df_capacity[col] = self.df_capacity[col] * 1e3 / 8760
            self.capacity_dict = self.df_capacity.set_index("Year")[col].to_dict()
            self.df["capacity_price"] = self.df["year"].apply(lambda t: self.get_capacity_price(t))
        else:
            self.df["capacity_price"] = capacity_price
            if convert_to_usd_per_mwh:
                self.df["capacity_price"] = self.df["capacity_price"] * 1e3 / 8760

    def get_market_price(self,
                         t,
                         col):
        """Query wholesale market for a given point of time.

        Args:
            t (datetime): timestamp
            col (str): energy price column name

        Returns:
            float: energy price in USD/MWh
        """
        
        if 'min' in self.resample:
            return self.df.loc[(self.df.year == t.year) & (self.df.month == t.month) & (self.df.day == t.day) & (self.df.hour == t.hour) & (self.df.minute == t.minute), col].mean()
        else:
            return self.df.loc[(self.df.year == t.year) & (self.df.month == t.month) & (self.df.day == t.day) & (self.df.hour == t.hour), col].mean()

    def get_recs_price(self,
                     t):
        """Query RECs market for a given point of time.

        Args:
            t (datetime): timestamp

        Returns:
            float: energy price in USD/MWh
        """
        return self.get_market_price(col="recs_price")
    
    def get_capacity_price(self,
                        year):

        """Query wholesale market for a given point of time.

        Args:
            year (int): year to get data

        Returns:
            float: energy price in either USD/MWh or USD/kW-year
        """

        return self.capacity_dict[year]

    def create_elcc_forecast(self,
                            filepath=None,
                            battery_elcc = 1.0,
                            start_year=None,
                            col="elcc"):
        """Create market for the effective load carrying capacity (ELCC) of batteries.

        Args:
            filepath (str, optional): csv file path to ELCC data, if available. Defaults to None.
            battery_elcc (float, optional): battery ELCC value. Defaults to 1.0.
            start_year (int, optional): start year. Defaults to None.
            col (str, optional): column name where ELCC can be accessed. Defaults to "elcc".
        """

        if filepath and battery_duration:
            self.df_elcc = pd.read_csv(filepath)
            self.elcc_dict = self.df_elcc.set_index("Year")[col].to_dict()
            self.df["battery_elcc"] = self.df["year"].apply(lambda t: self.get_elcc(t))
        
        else:
            self.df["battery_elcc"] = battery_elcc

    def get_elcc(self,
                year):
        """Query effective load carrying capacity (ELCC) data for batteries.

        Args:
            year (int): year to get data

        Returns:
            float: ELCC
        """

        """"""
        return self.elcc_dict[year]

    def plot_prices(self,
                    show=True,
                    ymin_percentile=10,
                    ymax_percentile=90):
        """Visualize the loaded and preprocessed wholesale market data.

        Args:
            show (bool, optional): whether or not to show the figure upon plotting. Defaults to True.
            ymin_percentile (float, optional): minimum price percentile to threshold plot range. Defaults to 10.
            ymax_percentile (float, optional): maximum price percentile to threshold plot range. Defaults to 90.

        Returns:
            _type_: _description_
        """
        
        df_gruoped_mean = self.df.groupby(by="hour").mean()
        df_gruoped_std = self.df.groupby(by="hour").std()

        self.fig, self.axes = plt.subplots(1, 1, figsize=(7,6), sharex=True)

        self.axes.plot(df_gruoped_mean.index, df_gruoped_mean.price_raw, 'darkblue', label="Raw Duck")
        self.axes.fill_between(df_gruoped_mean.index, df_gruoped_mean.price_raw-df_gruoped_std.price_raw, df_gruoped_mean.price_raw+df_gruoped_std.price_raw, color='darkblue', alpha=0.03)

        self.axes.plot(df_gruoped_mean.index, df_gruoped_mean.price, 'darkred', label="Fat Duck")
        self.axes.fill_between(df_gruoped_mean.index, df_gruoped_mean.price-df_gruoped_std.price, df_gruoped_mean.price+df_gruoped_std.price, color='darkred', alpha=0.03)

        self.axes.set_xlabel("Hour of Day", fontsize=14)
        self.axes.set_ylabel("LMP Price [$/MWh]", fontsize=14)
        mn = np.abs(np.percentile((df_gruoped_mean - df_gruoped_std)["price"].values, ymin_percentile))
        mx = np.abs(np.percentile((df_gruoped_mean + df_gruoped_std)["price"].values, ymax_percentile))
        plt.ylim([-max(mn, mx), max(mn, mx)])
        plt.legend(fontsize=14)

        if show:
            plt.show()
        return self.fig, self.axes
