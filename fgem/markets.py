import pandas as pd
import numpy as np
import pdb
from datetime import timedelta
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class TabularPowerMarket:

    """Power markets class."""

    def __init__(self):
        """Initating attributes for the TabularPowerMarket class."""
        pass

    def create_energy_market(self,
                            filepath=None,
                            energy_price=40,
                            recs_price=30,
                            L=30,
                            time_init=pd.to_datetime('today'),
                            resample=False,
                            fat_factor=1,
                            fat_window=24,
                                ):

        """Loading and preprocessnig wholesale market data."""
        
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
            self.df = self.df.resample(rule=resample, on='Date').mean() # I use the mean statistic here as it is reasonable and yields resmapling at hour zero
        
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
        
        """Load and preprocess capacity market data."""

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
        """Query wholesale market data for a given point of time."""
        
        if 'min' in self.resample:
            return self.df.loc[(self.df.year == t.year) & (self.df.month == t.month) & (self.df.day == t.day) & (self.df.hour == t.hour) & (self.df.minute == t.minute), col].mean()
        else:
            return self.df.loc[(self.df.year == t.year) & (self.df.month == t.month) & (self.df.day == t.day) & (self.df.hour == t.hour), col].mean()

    def get_realtime_price(self,
                     t):
        """Query real-time wholesale market data."""
        return self.get_market_price(col="price")

    def get_recs_price(self,
                     t):
        """Query renewable energy credits market data."""
        return self.get_market_price(col="recs_price")
    
    def get_capacity_price(self,
                        year):

        """Query capacity market data for a point of time."""

        return self.capacity_dict[year]

    # def elcc_model(self, x, a, b, c):
    #     year, duration = x
    #     year = np.maximum((year - self.x_min[0]) / (self.x_max[0] - self.x_min[0]), 0.0) # in case the provided year is smaller than the smallest in the dataset
    #     duration = (duration - self.x_min[1]) / (self.x_max[1] - self.x_min[1])
    #     return 1/(1+np.exp(-(a * duration + b * year**(-0.1) + c)))

    # def create_elcc_forecast(self):
    #     X = np.array([[2023, 4],
    #                 [2023, 6],
    #                 [2023, 8],
    #                 [2024, 4],
    #                 [2024, 6],
    #                 [2024, 8],
    #                 [2025, 4],
    #                 [2025, 6],
    #                 [2025, 8],
    #                 [2026, 4],
    #                 [2026, 6],
    #                 [2026, 8]])
    #     self.x_min = np.min(X, axis=0)
    #     self.x_max = np.max(X, axis=0)
    #     y = np.array([0.963, 0.98, 0.982, 0.907, 0.934, 0.943, 0.742, 0.796, 0.822, 0.69, 0.751, 0.782])
    #     self.popt, self.pcov = curve_fit(self.elcc_model, (X[:,0], X[:,1]), y)

    def create_elcc_forecast(self,
                            filepath=None,
                            battery_elcc = 1.0,
                            battery_duration=None,
                            battery_lifetime=None,
                            start_year=None,
                            col="elcc"):
        
        """Load and preprocess effective load carrying capacity (ELCC) data for Li-ion batteries."""
        
        if filepath and battery_duration:
            self.df_elcc = pd.read_csv(filepath)
            self.elcc_dict = self.df_elcc.set_index("Year")[col].to_dict()
            self.df["battery_elcc"] = self.df["year"].apply(lambda t: self.get_elcc(t, battery_duration[int(t-start_year > battery_lifetime)]))
        
        else:
            self.df["battery_elcc"] = battery_elcc

    def get_elcc(self,
                year,
                duration):
        """Query effective load carrying capacity (ELCC) data for Li-ion batteries."""
        return self.elcc_dict[year]
        # return self.elcc_model((year, duration), *self.popt)

    def plot_prices(self,
                    show=True,
                    ymin_percentile=10,
                    ymax_percentile=90):
        
        """Visualize the loaded and preprocessed wholesale market data."""
        
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
