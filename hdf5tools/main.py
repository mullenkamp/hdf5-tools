"""
Created on 2022-09-30.

@author: Mike K
"""
import h5py
import io
import os
import numpy as np
import xarray as xr
# from time import time
# from datetime import datetime
import cftime
# import dateutil.parser as dparser
# import numcodecs
# import utils
from hdf5tools import utils
import hdf5plugin
from typing import Union, List
import pathlib


##############################################
### Parameters



##############################################
### Functions


###################################################
### Class


class H5(object):
    """

    """
    def __init__(self, data: Union[List[Union[str, pathlib.Path, io.BytesIO, xr.Dataset]], Union[str, pathlib.Path, io.BytesIO, xr.Dataset]], group=None):
        """

        """
        ## Read paths input into the appropriate file objects
        files = utils.open_files(data, group)

        ## Get encodings
        encodings = utils.get_encodings(files)

        ## Get the extended coords
        coords_dict = utils.extend_coords(files)

        ## Add the variables as datasets
        vars_dict = utils.extend_variables(files, coords_dict)

        ## Assign attributes
        self._files = files
        self._coords_dict = coords_dict
        self._data_vars_dict = vars_dict


    def __repr__(self):
        """

        """

        # if hasattr(self, '_empty_ds'):




    def close(self):
        """

        """
        utils.close_files(self._files)






















######################################
### Testing
