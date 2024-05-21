import pandas as pd
import warnings
import pdb
import os
import fsspec
import ujson
import zarr
from kerchunk.hdf import SingleHdf5ToZarr
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import pickle
import numpy as np

from pathlib import Path
parent_path = Path(__file__).parent

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

class Weather:
    """Weather model holder."""
    def __init__(self):
        """Initiate attributes for Weather class."""
        pass
    
    def create_weather_model(self,
                            filepath,
                            resample=False,
                            sup3rcc_weather_forecast=False,
                            project_lat=None,
                            project_long=None,
                            years=None,
                            n_jobs=5
                            ):
        
        """Load and preprocess weather data.

        Args:
            filepath (str, optional): If available, csv filepath to weather data. Defaults to None.
            resample (bool, optional): whether or not to resample the project to a specific timestep (Options: "1Y", "1m", "1w", "1d", "1h" for yearly, monthly, weekly, daily, or hourly timestepping). Defaults to False.
            sup3rcc_weather_forecast (bool, optional): whether or not to use forecasts by sup3rcc (Buster et al. 2024, Nature Energy). Defaults to False.
            project_lat (float, optional): project latitude. Defaults to None.
            project_long (float, optional): project longitude. Defaults to None.
            years (iterable, optional): years spanning the project lifetime. Defaults to None.
            n_jobs (int, optional): number of processes used to collect the sup3rcc forecasts. Defaults to 5.

        """

        self.sup3rcc_weather_forecast = sup3rcc_weather_forecast
        self.project_lat = project_lat
        self.project_long = project_long
        self.years = years
        self.n_jobs = n_jobs

        if self.sup3rcc_weather_forecast:
            print("Query weather forecasts from Sup3rCC ...")
            if self.n_jobs<= 1:
                self.df = pd.concat([query_sup3rcc_trh(y, self.project_lat, self.project_long) for y in tqdm(self.years)])
            else:
                self.df = pd.concat(Parallel(n_jobs=self.n_jobs)(delayed(query_sup3rcc_trh)(y, self.project_lat, self.project_long) for y in tqdm(self.years)))
        else:
            self.df = pd.read_csv(filepath)

        self.df["Date"] = pd.to_datetime(self.df.time if "time" in self.df.columns else self.df.Date)
        self.df = self.df[["Date", "T0"]]
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

def query_sup3rcc_trh(year, lat, long, dst_dir="data/sup3rcc_cache"):
    """Access Sup3rCC ambient temperature forecasts within FGEM, where meta data is aleady downloaded.

    Args:
        year (int): year for which to query ambient temperature.
        lat (float): latitude.
        long (float): longitude.
        dst_dir (str): distination directory where meta data is already saved.

        Returns:
        pd.DataFrame: hourly ambient temperature forecasts.
    """

    dst_dir = os.path.join(parent_path, dst_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    
    s3_path = f"s3://nrel-pds-sup3rcc/conus_mriesm20_ssp585_r1i1p1f1/v0.1.0/sup3rcc_conus_mriesm20_ssp585_r1i1p1f1_trh_{year}.h5"
    metadata_json_filepath = f"{dst_dir}/{year}.pkl"
    meta_pickle_filepath = f"{dst_dir}/meta_{year}.pkl"
    meta_cols = ["lat", "long", "timezone", "elevation", "country", "state", "county", "offshore", "eez"]

    fo = pickle.load(open(metadata_json_filepath, 'rb'))
    mapper = fsspec.get_mapper("reference://",
                               fo=fo,
                               remote_protocol="s3",
                               remote_options={"anon": True})
    data = zarr.open(mapper)
    df_meta, time_index = pickle.load(open(meta_pickle_filepath, 'rb'))

    idx = ((df_meta["lat"] - lat).abs() + (df_meta["long"] - long).abs()).argmin()
    gid = df_meta.loc[idx, "gid"]
    
    arr = data['temperature_2m'][:, gid] / data['temperature_2m'].attrs["scale_factor"]
    
    df = pd.DataFrame()
    df["Date"] = time_index
    df["Date"] = df["Date"].dt.tz_localize(None)
    df["T0"] = arr
    
    return df

def download_query_sup3rcc(year,
                          lat, 
                          long, 
                          data_type="trh",
                          attribute="temperature_2m",
                          country="United States",
                          dst_dir="sup3rcc_cache"):
    
    """Download and save metadata Sup3rCC forecasts within FGEM.

    Args:
        year (int): year for which to query ambient temperature.
        lat (list): list of latitudes.
        long (list): list of longitudes.
        data_type (str): data to retrieve: either 'trh', 'wind', 'solar', or 'pressure'.
        attribute (str): the attribute/variable to be retrieved. This must correspond to the specified data-type.
        country (str): if the retreiveal is for a single country, specifying it can speed up the process.
        dst_dir (str): distination directory where meta data should be saved.


        Returns:
        pd.DataFrame: hourly forecasts for the given attribute, year, and locations.

        Examples:
            >>> data_type = "wind"
            >>> attribute = "windspeed_100m"
            >>> year = 2050
            >>> lat = np.random.randint(24.521208, 49.382808, size=10)
            >>> long = np.random.randint(-124.736342, -66.945392, size=10)
            >>> df = download_query_su3rcc_trh(year, lat, long, data_type, attribute)

    """

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
        
    s3_path = f"s3://nrel-pds-sup3rcc/conus_mriesm20_ssp585_r1i1p1f1/v0.1.0/sup3rcc_conus_mriesm20_ssp585_r1i1p1f1_{data_type}_{year}.h5"
    metadata_filepath = f"{dst_dir}/{data_type}_{year}.pkl"
    
    meta_cols = ["lat", "long", "timezone", "elevation", "country", "state", "county", "offshore", "eez"]
    
    if not os.path.exists(metadata_filepath):
        print(f"Retrieve metadata from NREL AWS for {data_type}_{year} ... ")
        storage_opts = dict(mode="rb", anon=True, default_fill_cache=False,
                default_cache_type="none")

        h5chunks = SingleHdf5ToZarr(s3_path, storage_options=storage_opts,
                                    inline_threshold=0)
        
        fo = h5chunks.translate()
        
        mapper = fsspec.get_mapper("reference://",
                                   fo=fo,
                                   remote_protocol="s3",
                                   remote_options={"anon": True})
    
    
        data = zarr.open(mapper)
        time_index = pd.to_datetime(data.time_index[:].astype(str))
        df_meta = pd.DataFrame(data.meta, columns=meta_cols)
        if country:
            df_meta = df_meta[df_meta["country"] == country]
        df_meta = df_meta.reset_index(names=["gid"])
        df_meta[["lat", "long"]] = df_meta[["lat", "long"]].astype(float)

        pickle.dump((mapper, df_meta, time_index), open(metadata_filepath, "wb"))
    
    else:
        print(f"Retrieve metadata from cache for {data_type}_{year} ... ")
        mapper, df_meta, time_index = pickle.load(open(metadata_filepath, "rb"))
        data = zarr.open(mapper)
    
    print("Compute weather at coordinates ...")
    idx = np.abs(np.vstack((lat, long)).T[:,:,None] - df_meta[["lat", "long"]].values.T).sum(axis=1).argmin(axis=1)
    gid = df_meta.loc[idx, "gid"]
    arr = data[attribute][:, gid] / data[attribute].attrs["scale_factor"]
    columns = ["gid", "lat", "long"] + list(range(1, 8761))
    df = pd.DataFrame(np.vstack((gid.values, lat, long, arr[:8760])).T, columns=columns)
    
    return df