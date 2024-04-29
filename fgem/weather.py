import pandas as pd
import warnings
import pdb

class Weather:
    """Weather model holder."""
    def __init__(self):
        """Initiate attributes for Weather class."""
        pass
    
    def create_weather_model(self,
                            filepath,
                            resample=False
                            ):
        
        """Load and preprocess weather data.

        Args:
            filepath (str, optional): If available, csv filepath to weather data. Defaults to None.
            resample (bool, optional): whether or not to resample the project to a specific timestep (Options: "1Y", "1m", "1w", "1d", "1h" for yearly, monthly, weekly, daily, or hourly timestepping). Defaults to False.
        """

        self.df = pd.read_csv(filepath)
        self.df["Date"] = pd.to_datetime(self.df.time if "time" in self.df.columns else self.df.Date)
        self.df.rename(columns={'temp': 'T0', 'wspd':'wind_speed'}, inplace=True)
        self.df = self.df[["Date", "T0", "wind_speed"]]
        self.df['year'] = self.df.Date.apply(lambda x: x.year)
        self.df['month'] = self.df.Date.apply(lambda x: x.month)
        self.df['day'] = self.df.Date.apply(lambda x: x.day)
        self.df['hour'] = self.df.Date.apply(lambda x: x.hour)
        self.df['minute'] = self.df.Date.apply(lambda x: x.minute)
        self.df['dayofyear'] = self.df.Date.apply(lambda x: x.dayofyear)

        if resample:
            self.resample = resample
            self.df = self.df.resample(rule=resample, on='Date').last()

        else:
            self.df.set_index('Date', inplace=True)

        if self.df.isna().sum().max() > 10:
            warnings.warn("WARNING: Too many missing weather data ... ")
        self.df.ffill(inplace=True)

    def amb_temp(self,
                 t):
        
        """Query ambient temperature for a point in time.

        Args:
            t (datetime): timestamp

        Returns:
            float: ambient temperature in deg C
        """
        
        if 'min' in self.resample:
            return self.df.loc[(self.df.month == t.month) & (self.df.day == t.day) & (self.df.hour == t.hour) & (self.df.minute == t.minute), "T0"].mean()
        else:
            return self.df.loc[(self.df.month == t.month) & (self.df.day == t.day) & (self.df.hour == t.hour), "T0"].mean()
