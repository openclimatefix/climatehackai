import glob
import os

import fsspec
import numpy as np
import xarray as xr
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

SAT_MEAN = {
    "HRV": 236.13257536395903,
    "IR_016": 291.61620182554185,
    "IR_039": 858.8040610176552,
    "IR_087": 738.3103442750336,
    "IR_097": 773.0910794778366,
    "IR_108": 607.5318145165666,
    "IR_120": 860.6716261423857,
    "IR_134": 925.0477987594331,
    "VIS006": 228.02134593063957,
    "VIS008": 257.56333202381205,
    "WV_062": 633.5975770915588,
    "WV_073": 543.4963868823854,
}

SAT_STD = {
    "HRV": 935.9717382401759,
    "IR_016": 172.01044433112992,
    "IR_039": 96.53756504807913,
    "IR_087": 96.21369354283686,
    "IR_097": 86.72892737648276,
    "IR_108": 156.20651744208888,
    "IR_120": 104.35287930753246,
    "IR_134": 104.36462050405994,
    "VIS006": 150.2399269307514,
    "VIS008": 152.16086321818398,
    "WV_062": 111.8514878214775,
    "WV_073": 106.8855172848904,
}


SATELLITE_CHANNEL_ORDER = ("example", "time", "channel", "y", "x")


def _set_sat_coords(dataset: xr.Dataset) -> xr.Dataset:
    """Set variables as coordinates."""
    return dataset.set_coords(
        ["time_utc", "channel_name", "y_osgb", "x_osgb", "y_geostationary", "x_geostationary"]
    )


def load_netcdf(filename, engine="h5netcdf", *args, **kwargs) -> xr.Dataset:
    """Load a NetCDF dataset from local file system or cloud bucket."""
    with fsspec.open(filename, mode="rb") as file:
        dataset = xr.load_dataset(file, engine=engine, *args, **kwargs)
    return dataset


class Satellite(DataLoader):
    def __init__(
        self,
        channels=[
            "IR_016",
            "IR_039",
            "IR_087",
            "IR_097",
            "IR_108",
            "IR_120",
            "IR_134",
            "VIS006",
            "VIS008",
            "WV_062",
            "WV_073",
        ],
        data_dir="./",
    ):
        self.channels = channels
        self.data_dir = data_dir
        if "HRV" in self.channels:
            self.data_dir = self.data_dir + "hrvsatellite/"
        else:
            self.data_dir = self.data_dir + "satellite/"
        self.files = list(glob.glob(os.path.join(self.data_dir, "*.nc")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        dataset = load_netcdf(self.files[item])
        dataset = dataset.drop_vars(
            [
                "example",
                "y_geostationary_index",
                "x_geostationary_index",
                "time_index",
                "channels_index",
            ]
        )

        # Rename coords to be more explicit about exactly what some coordinates hold:
        dataset = dataset.rename_vars(
            {
                "channels": "channel_name",
                "time": "time_utc",
            }
        )

        # Rename dimensions. Standardize on the singular (time, channel, etc.).
        # Remove redundant "index" from the dim name. These are *dimensions* so,
        # by definition, they are indicies!
        dataset = dataset.rename_dims(
            {
                "y_geostationary_index": "y",
                "x_geostationary_index": "x",
                "time_index": "time",
                "channels_index": "channel",
            }
        )

        # Setting coords won't be necessary once this is fixed:
        # https://github.com/openclimatefix/nowcasting_dataset/issues/627
        dataset = _set_sat_coords(dataset)

        dataset = dataset.transpose(*SATELLITE_CHANNEL_ORDER)
        # Prepare the satellite imagery itself
        hrvsatellite = dataset["data"]
        # hrvsatellite is int16 on disk
        hrvsatellite = hrvsatellite.astype(np.float32)
        mean = np.array([SAT_MEAN[b] for b in self.channels])
        std = np.array([SAT_STD[b] for b in self.channels])
        # Need to get to the same shape, so add 3 1-dimensions
        mean = np.expand_dims(mean, axis=[1, 2, 3])
        std = np.expand_dims(std, axis=[1, 2, 3])
        hrvsatellite = hrvsatellite - mean
        hrvsatellite = hrvsatellite / std
        input_data = hrvsatellite.values[:, :7]
        target_data = hrvsatellite.values[:, 7:]
        merged_data = np.concatenate(
            (
                input_data,
                np.expand_dims(dataset["y_osgb"].values, axis=[1, 2]),
                np.expand_dims(dataset["x_osgb"].values, axis=[1, 2]),
            ),
            1,
        )  # Now in Batch, Time+Coord, Channel, W, H orderk
        return np.squeeze(merged_data), np.squeeze(target_data)
