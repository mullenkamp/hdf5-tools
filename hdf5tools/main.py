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
# import numcodecs
# import utils
from hdf5tools import utils
import hdf5plugin
from typing import Union, List
import pathlib
import copy

##############################################
### Parameters



##############################################
### Functions


###################################################
### Class


class H5(object):
    """
    Class to load and combine one or more HDF5 data files (or xarray datasets) with optional filters. The class will then export the combined data to an HDF5 file, file object, or xr.Dataset.

    Parameters
    ----------
    data : str, pathlib.Path, io.BytesIO, xr.Dataset, or list of str, pathlib.Path, io.BytesIO, xr.Dataset
        The input data need to either be a path to HDF5 file(s), BytesIO objects, or xr.Datasets (or some combo of those).
    group : str or None
        The group or group path within the hdf5 file(s) to the datasets.

    Returns
    -------
    H5 instance
    """
    def __init__(self, data: Union[List[Union[str, pathlib.Path, io.BytesIO, xr.Dataset]], Union[str, pathlib.Path, io.BytesIO, xr.Dataset]], group=None):
        """
        Class to load and combine one or more HDF5 data files (or xarray datasets) with optional filters. The class will then export the combined data to an HDF5 file, file object, or xr.Dataset.

        Parameters
        ----------
        data : str, pathlib.Path, io.BytesIO, xr.Dataset, or list of str, pathlib.Path, io.BytesIO, xr.Dataset
            The input data need to either be a path to HDF5 file(s), BytesIO objects, or xr.Datasets (or some combo of those).
        group : str or None
            The group or group path within the hdf5 file(s) to the datasets.

        Returns
        -------
        H5 instance
        """
        ## Read paths input into the appropriate file objects
        if isinstance(data, list):
            data1 = data
        else:
            data1 = [data]

        files = utils.open_files(data1, group)

        ## Get encodings
        encodings = utils.get_encodings(files)

        ## Get attrs
        attrs, global_attrs = utils.get_attrs(files)

        ## Get the extended coords
        coords_dict = utils.extend_coords(files, encodings)

        ## Add the variables as datasets
        vars_dict = utils.index_variables(files, coords_dict, encodings)

        ## Close files
        utils.close_files(files)

        ## Assign attributes
        self._files = data1
        self._group = group
        self._coords_dict = coords_dict
        self._data_vars_dict = vars_dict
        self._attrs = attrs
        self._global_attrs = global_attrs
        self._encodings = encodings


    def _build_empty_ds(self):
        """

        """
        if self._data_vars_dict:

            ## get all of the coords associated with the existing data vars
            all_data_coords = set()
            for ds in self._data_vars_dict:
                for dim in self._data_vars_dict[ds]['dims']:
                    all_data_coords.add(dim)

            ## Create empty xr.Dataset
            data_vars = {}
            for k, v in self._data_vars_dict.items():
                if 'datetime' in v['dtype_decoded'].name:
                    data_vars[k] = (v['dims'], np.empty(v['shape'], dtype=np.dtype('datetime64[ns]')))
                else:
                    data_vars[k] = (v['dims'], np.empty(v['shape'], dtype=v['dtype_decoded']))

            coords = {}
            for k, v in self._coords_dict.items():
                if k in all_data_coords:
                    if 'datetime' in v.dtype.name:
                        coords[k] = v.astype('datetime64[ns]')
                    else:
                        coords[k] = v

            xr_ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=self._global_attrs)

            for ds_name, attr in self._attrs.items():
                if ds_name in xr_ds:
                    xr_ds[ds_name].attrs = attr
            for ds_name, enc in self._encodings.items():
                if ds_name in xr_ds:
                    xr_ds[ds_name].encoding = enc

        else:
            xr_ds = xr.Dataset()

        return xr_ds


    def __repr__(self):
        """

        """
        xr_ds = self._build_empty_ds()

        return xr_ds.__repr__()


    def sel(self, selection: dict=None, include_coords: list=None, exclude_coords: list=None, include_data_vars: list=None, exclude_data_vars: list=None):
        """
        Filter the data by a selection, include, and exclude. Returns a new H5 instance. The selection parameter is very similar to xarry's .sel method.

        Parameters
        ----------
        selection : dict
            This filter requires a dict of coordinates using three optional types of filter values. These include slice instances (the best and preferred option), a list/np.ndarray of coordinate values, or a bool np.ndarray of the coordinate data length.
        include_coords : list
            A list of coordinates to include in the output. Only data variables with included coordinates will be included in the output.
        exclude_coords : list
            A list of coordinates to exclude from the output. Only data variables with coordinates that have not been excluded will be included in the output.
        include_data_vars : list
            A list of data variables to include in the output. Only coordinates that have data variables will be included in the output.
        exclude_data_vars : list
            A list of data variables to exclude from the output. Only coordinates that have data variables will be included in the output.

        Returns
        -------
        H5 instance
        """
        c = self.copy()
        if selection is not None:
            files = utils.open_files(self._files, self._group)
            utils.filter_coords(files, c._coords_dict, selection)
            vars_dict = utils.index_variables(files, c._coords_dict, c._encodings)

            c._data_vars_dict = vars_dict

            ## Close files
            utils.close_files(files)

        if include_coords is not None:
            coords_rem_list = []
            for k in list(c._coords_dict.keys()):
                if k not in include_coords:
                    _ = c._coords_dict.pop(k)
                    coords_rem_list.append(k)

            if coords_rem_list:
                for k in list(c._data_vars_dict.keys()):
                    for coord in coords_rem_list:
                        if coord in c._data_vars_dict[k]['dims']:
                            c._data_vars_dict.pop(k)
                            break

        if exclude_coords is not None:
            coords_rem_list = []
            for k in list(c._coords_dict.keys()):
                if k in exclude_coords:
                    _ = c._coords_dict.pop(k)
                    coords_rem_list.append(k)

            if coords_rem_list:
                for k in list(c._data_vars_dict.keys()):
                    for coord in coords_rem_list:
                        if coord in c._data_vars_dict[k]['dims']:
                            c._data_vars_dict.pop(k)
                            break

        if include_data_vars is not None:
            c._data_vars_dict = {k: v for k, v in c._data_vars_dict.items() if k in include_data_vars}

            include_coords = set()
            for k, v in c._data_vars_dict.items():
                include_coords.update(set(v['dims']))

            for k in list(c._coords_dict.keys()):
                if k not in include_coords:
                    _ = c._coords_dict.pop(k)

        if exclude_data_vars is not None:
            c._data_vars_dict = {k: v for k, v in c._data_vars_dict.items() if k not in exclude_data_vars}

            include_coords = set()
            for k, v in c._data_vars_dict.items():
                include_coords.update(set(v['dims']))

            for k in list(c._coords_dict.keys()):
                if k not in include_coords:
                    _ = c._coords_dict.pop(k)

        return c


    def copy(self):
        """
        Deep copy an H5 instance.
        """
        c = copy.deepcopy(self)

        return c


    def coords(self):
        """
        A Summary of the coordinates.
        """
        coords_summ = {}
        for k, v in self._coords_dict.items():
            encs = copy.deepcopy(self._encodings[k])
            coords_summ[k] = {'shape': v.shape}
            coords_summ[k].update(encs)

        return coords_summ


    def data_vars(self):
        """
        A summary of the data variables.
        """
        vars_summ = {}
        for k, v in self._data_vars_dict.items():
            encs = copy.deepcopy(self._encodings[k])
            vars_summ[k] = {k1: v1 for k1, v1 in v.items() if k1 in ['dims', 'shape']}
            vars_summ[k].update(encs)

        return vars_summ


    def variables(self):
        """
        A summary of all variables/datasets. Both coordinates and data variables.
        """
        coords_summ = self.coords()
        vars_summ = self.data_vars()

        coords_summ.update(vars_summ)

        return coords_summ


    def to_hdf5(self, output: Union[str, pathlib.Path, io.BytesIO], group=None, chunks=None, unlimited_dims=None, compression='zstd'):
        """
        Method to output the filtered data to an HDF5 file or file object.

        Parameters
        ----------
        output : str, pathlib.Path, or io.BytesIO
            The output path of the new combined hdf5 file.
        group : str or None
            The group or group path within the hdf5 file to save the datasets.
        chunks : dict of tuples
            The chunks per dataset. Must be a dictionary of dataset names with tuple values of appropriate dimensions. A value of None will perform auto-chunking.
        unlimited_dims : str, list of str, or None
            The dimensions/coordinates that should be assigned as "unlimited" in the hdf5 file.
        compression : str
            The compression used for the chunks in the hdf5 files. Must be one of gzip, lzf, zstd, or None. gzip is compatible with any hdf5 installation (not only h5py), so this should be used if interoperability across platforms is important. lzf is compatible with any h5py installation, so if only python users will need to access these files then this is a better option than gzip. zstd requires the hdf5plugin python package, but is the best compression option if users have access to the hdf5plugin package. None has no compression and is generally not recommended except in niche situations.

        Returns
        -------
        None
        """
        ## Check if there's anything to save
        if self._coords_dict:

            ## Set up initial parameters
            if isinstance(unlimited_dims, str):
                unlimited_dims = [unlimited_dims]
            else:
                unlimited_dims = []

            compressor = utils.get_compressor(compression)

            files = utils.open_files(self._files, self._group)

            ## Create new file
            with h5py.File(output, 'w', libver='latest', rdcc_nbytes=3*1024*1024) as nf:

                if isinstance(group, str):
                    nf1 = nf.create_group(group)
                else:
                    nf1 = nf

                ## Add the coords as datasets
                for coord, arr in self._coords_dict.items():
                    enc_arr = utils.encode_data(arr, **self._encodings[coord])
                    shape = enc_arr.shape

                    maxshape = tuple([s if s not in unlimited_dims else None for s in shape])

                    chunks1 = utils.guess_chunk(shape, maxshape, enc_arr.dtype)

                    if isinstance(chunks, dict):
                        if coord in chunks:
                            chunks1 = chunks[coord]

                    ds = nf1.create_dataset(coord, shape, chunks=chunks1, maxshape=maxshape, dtype=enc_arr.dtype, **compressor)

                    ds[:] = enc_arr

                    ds.make_scale(coord)

                ## Add the variables as datasets
                vars_dict = copy.deepcopy(self._data_vars_dict)

                for var_name in vars_dict:
                    shape = vars_dict[var_name]['shape']
                    dims = vars_dict[var_name]['dims']
                    maxshape = tuple([s if dims[i] not in unlimited_dims else None for i, s in enumerate(shape)])

                    chunks1 = utils.guess_chunk(shape, maxshape, vars_dict[var_name]['dtype'])

                    if isinstance(chunks, dict):
                        if var_name in chunks:
                            chunks1 = chunks[var_name]

                    if len(shape) == 0:
                        chunks1 = None
                        compressor1 = {}
                        vars_dict[var_name]['fillvalue'] = None
                        maxshape = None
                    else:
                        compressor1 = compressor

                    ds = nf1.create_dataset(var_name, shape, chunks=chunks1, maxshape=maxshape, dtype=vars_dict[var_name]['dtype'], fillvalue=vars_dict[var_name]['fillvalue'], **compressor1)

                    ds_dims = ds.dims
                    for i, dim in enumerate(dims):
                        ds_dims[i].attach_scale(nf1[dim])
                        ds_dims[i].label = dim

                    # Load the data by file
                    for i in vars_dict[var_name]['data']:
                        file = files[i]

                        ds_old = file[var_name]

                        if ds.chunks is None:
                            if isinstance(ds_old, xr.DataArray):
                                ds[()] = utils.encode_data(ds_old.values, **self._encodings[var_name])
                            else:
                                ds[()] = ds_old[()]
                        else:
                            global_index = vars_dict[var_name]['data'][i]['global_index']
                            local_index = vars_dict[var_name]['data'][i]['local_index']
                            dims_order = vars_dict[var_name]['data'][i]['dims_order']
                            local_dims = tuple(dims[i] for i in dims_order)
                            transpose_order = tuple(dims_order.index(i) for i in range(len(dims_order)))

                            global_chunks, local_chunks = utils.index_chunks(shape, chunks1, global_index, local_index, dims_order)

                            if isinstance(ds_old, xr.DataArray):
                                for global_chunk, local_chunk in zip(global_chunks, local_chunks):
                                    data = ds_old[local_chunk].copy().load()

                                    if dims == local_dims:
                                        ds[global_chunk] = utils.encode_data(data.values, **self._encodings[var_name])
                                    else:
                                        ds[global_chunk] = utils.encode_data(data.values.transpose(dims_order), **self._encodings[var_name])
                                    data.close()
                                    del data
                            else:
                                for global_chunk, local_chunk in zip(global_chunks, local_chunks):
                                    if dims == local_dims:
                                        ds[global_chunk] = ds_old[local_chunk]
                                    else:
                                        ds[global_chunk] = ds_old[local_chunk].transpose(transpose_order)

                ## Assign attrs
                for ds_name, attr in self._attrs.items():
                    if ds_name in nf1:
                        nf1[ds_name].attrs.update(attr)

                for ds_name, encs in self._encodings.items():
                    if ds_name in nf1:
                        for f, enc in encs.items():
                            if 'dtype' in f:
                                enc = enc.name
                            nf1[ds_name].attrs.update({f: enc})

                nf1.attrs.update(self._global_attrs)

            if isinstance(output, io.BytesIO):
                output.seek(0)

            utils.close_files(files)
        else:
            print('No data to save')


    def to_xarray(self):
        """
        Save an HDF5 file to an io.BytesIO object which is then opened by xr.open_dataset using the h5netcf engine.

        Returns
        -------
        xr.Dataset
        """
        if self._coords_dict:
            b1 = io.BytesIO()

            self.to_hdf5(b1)

            xr_ds = xr.open_dataset(b1, engine='h5netcdf')
        else:
            xr_ds = xr.Dataset()

        return xr_ds

















######################################
### Testing
