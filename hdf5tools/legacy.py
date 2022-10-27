#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:47:36 2022

@author: mike
"""
import h5py
import io
import xarray as xr
import numpy as np
import cftime
# from time import time
# from datetime import datetime
# import dateutil.parser as dparser
# import numcodecs
# import utils
from hdf5tools import utils
import hdf5plugin
from typing import Union, List
import pathlib


##############################################
### Parameters

time_str_conversion = {'days': 'datetime64[D]',
                       'hours': 'datetime64[h]',
                       'minutes': 'datetime64[m]',
                       'seconds': 'datetime64[s]',
                       'milliseconds': 'datetime64[ms]'}

##############################################
### Functions


def encode_datetime(data, units=None, calendar='gregorian'):
    """

    """
    if units is None:
        output = data.astype('datetime64[s]').astype(int)
    else:
        if '1970-01-01' in units:
            time_unit = units.split()[0]
            output = data.astype(time_str_conversion[time_unit]).astype(int)
        else:
            output = cftime.date2num(data.astype('datetime64[s]').tolist(), units, calendar)

    return output


def decode_datetime(data, units=None, calendar='gregorian'):
    """

    """
    if units is None:
        output = data.astype('datetime64[s]')
    else:
        if '1970-01-01' in units:
            time_unit = units.split()[0]
            output = data.astype(time_str_conversion[time_unit])
        else:
            output = cftime.num2pydate(data, units, calendar).astype('datetime64[s]')

    return output


def encode_data(data, dtype, missing_value=None, add_offset=0, scale_factor=None, units=None, calendar=None, **kwargs):
    """

    """
    if 'datetime64' in data.dtype.name:
        data = encode_datetime(data, units, calendar)
    elif isinstance(scale_factor, (int, float, np.number)):
        data = (data - add_offset)/scale_factor

        if isinstance(missing_value, (int, np.number)):
            data[np.isnan(data)] = missing_value

    data = data.astype(dtype)

    return data


def get_xr_encoding(xr_array):
    """

    """
    encoding = xr_array.encoding.copy()

    if (xr_array.dtype.name == 'object') or ('str' in xr_array.dtype.name):
        encoding['dtype'] = h5py.string_dtype()
    elif 'datetime64' in xr_array.dtype.name:
        encoding['dtype'] = np.dtype('int64')
        encoding['calendar'] = 'gregorian'

    if 'dtype' not in encoding:
        encoding['dtype'] = xr_array.dtype
    elif isinstance(encoding['dtype'], str):
        encoding['dtype'] = np.dtype(encoding['dtype'])

    if 'calendar' in encoding:
        encoding['units'] = 'seconds since 1970-01-01 00:00:00'

    if 'missing_value' in encoding:
        encoding['_FillValue'] = encoding['missing_value']
        fillvalue = encoding['missing_value']
    else:
        fillvalue = None

    return encoding, fillvalue


def is_scale(dataset):
    """

    """
    check = h5py.h5ds.is_scale(dataset._id)

    return check


def is_regular_index(arr_index):
    """

    """
    reg_bool = np.all(np.diff(arr_index) == 1) or len(arr_index) == 1

    return reg_bool


def extend_coords(files):
    """

    """
    coords_dict = {}

    for file in files:
        ds_list = [ds_name for ds_name in file.keys() if is_scale(file[ds_name])]

        for ds_name in ds_list:
            ds = file[ds_name]

            if ds.dtype.name == 'object':
                if ds_name in coords_dict:
                    coords_dict[ds_name] = np.union1d(coords_dict[ds_name], ds[:]).astype(h5py.string_dtype())
                else:
                    coords_dict[ds_name] = ds[:].astype(h5py.string_dtype())
            else:
                if ds_name in coords_dict:
                    coords_dict[ds_name] = np.union1d(coords_dict[ds_name], ds[:])
                else:
                    coords_dict[ds_name] = ds[:]

    return coords_dict


def extend_variables(files, coords_dict):
    """
    
    """
    vars_dict = {}

    for i, file in enumerate(files):
        ds_list = [ds_name for ds_name in file.keys() if not is_scale(file[ds_name])]

        for ds_name in ds_list:
            ds = file[ds_name]

            dims = []
            slice_index = []

            for dim in ds.dims:
                dim_name = dim[0].name.split('/')[-1]
                dims.append(dim_name)
                arr_index = np.where(np.isin(coords_dict[dim_name], dim[0][:]))[0]

                if is_regular_index(arr_index):
                    slice1 = slice(arr_index.min(), arr_index.max() + 1)
                    slice_index.append(slice1)
                else:
                    slice_index.append(arr_index)

            if ds_name in vars_dict:
                if not np.in1d(vars_dict[ds_name]['dims'], dims).all():
                    raise ValueError('dims are not consistant between the same named datasets.')
                if vars_dict[ds_name]['dtype'] != ds.dtype:
                    raise ValueError('dtypes are not consistant between the same named datasets.')

                vars_dict[ds_name]['data'][i] = {'dims_order': tuple(dims), 'slice_index': tuple(slice_index)}
            else:
                shape = tuple([coords_dict[dim_name].shape[0] for dim_name in dims])

                if isinstance(ds.dtype, np.number):
                    fillvalue = ds.fillvalue
                else:
                    fillvalue = None

                vars_dict[ds_name] = {'data': {i: {'dims_order': tuple(dims), 'slice_index': tuple(slice_index)}}, 'dims': tuple(dims), 'shape': shape, 'dtype': ds.dtype, 'fillvalue': fillvalue}

    return vars_dict


def create_nc_dataset(hdf, xr_dataset, var_name, chunks, compressor, unlimited_dims):
    """

    """
    ds_xr = xr_dataset[var_name].copy()
    shape = ds_xr.shape
    dims = ds_xr.dims
    maxshape = tuple([s if dims[i] not in unlimited_dims else None for i, s in enumerate(shape)])

    encoding, fillvalue = get_xr_encoding(ds_xr)

    chunks1 = utils.guess_chunk(shape, maxshape, encoding['dtype'])

    if isinstance(chunks, dict):
        if var_name in chunks:
            chunks1 = chunks[var_name]

    if len(shape) == 0:
        chunks1 = None
        compressor1 = {}
        fillvalue = None
        maxshape = None
    else:
        compressor1 = compressor

    ds = hdf.create_dataset(var_name, shape, chunks=chunks1, maxshape=maxshape, dtype=encoding['dtype'], fillvalue=fillvalue, **compressor1)

    if ds.chunks is None:
        ds[()] = ds_xr.values

    elif ('scale_factor' in encoding) or ('add_offset' in encoding) or ('calendar' in encoding):
        if ds.chunks == shape:
            ds[:] = encode_data(ds_xr.values, **encoding)
        else:
            new_slices, source_slices = utils.copy_chunks_simple(shape, chunks1)

            for new_slice, source_slice in zip(new_slices, source_slices):
                # print(new_slice, source_slice)
                ds[new_slice] = encode_data(xr_dataset[var_name][source_slice].copy().load().values, **encoding)

    else:
        if ds.chunks == shape:
            ds[:] = xr_dataset[var_name].copy().load().values
        else:
            new_slices, source_slices = utils.copy_chunks_simple(shape, chunks1)

            for new_slice, source_slice in zip(new_slices, source_slices):
                # print(new_slice, source_slice)
                ds[new_slice] = xr_dataset[var_name][source_slice].copy().load().values

    _ = encoding.pop('dtype')
    # print(enc)

    ## Attributes
    attrs = xr_dataset[var_name].attrs.copy()
    attrs.update(encoding)

    ds.attrs.update(attrs)

    if var_name in xr_dataset.dims:
        ds.make_scale(var_name)

    ds_dims = ds.dims
    for i, dim in enumerate(dims):
        if dim != var_name:
            ds_dims[i].attach_scale(hdf[dim])
            ds_dims[i].label = dim

    return ds


def xr_to_hdf5(xr_dataset, new_path, group=None, chunks=None, unlimited_dims=None, compression='zstd'):
    """
    Function to convert an xarray dataset to an hdf5 file.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Xarray Dataset.
    new_path : str or pathlib
        Output path.
    group : str or None
        The group or group path within the hdf5 file to the datasets.
    chunks : dict of tuples
        The chunks per dataset. Must be a dictionary of dataset name keys with tuple values of appropriate dimensions. A value of None will perform auto-chunking.
    unlimited_dims : str, list of str, or None
        The dimensions that should be assigned as "unlimited".
    compression : str
        The compression used for the chunks in the hdf5 files. Must be one of gzip, lzf, zstd, or None.

    Returns
    -------
    None
    """
    if isinstance(unlimited_dims, str):
        unlimited_dims = [unlimited_dims]
    else:
        unlimited_dims = []

    compressor = utils.get_compressor(compression)

    xr_dims_list = list(xr_dataset.dims)

    with h5py.File(new_path, 'w', libver='latest', rdcc_nbytes=3*1024*1024) as f:

        if isinstance(group, str):
            g = f.create_group(group)
        else:
            g = f

        ## Create coords
        for coord in xr_dims_list:
            _ = create_nc_dataset(g, xr_dataset, coord, chunks, compressor, unlimited_dims)

        ## Create data vars
        for var in list(xr_dataset.data_vars):
            _ = create_nc_dataset(g, xr_dataset, var, chunks, compressor, unlimited_dims)

        ## Dataset attrs
        attrs = {}
        attrs.update(xr_dataset.attrs)
        g.attrs.update(attrs)

    if isinstance(new_path, io.BytesIO):
        new_path.seek(0)


def combine_hdf5(paths: List[Union[str, pathlib.Path, io.BytesIO]], new_path: Union[str, pathlib.Path], group=None, chunks=None, unlimited_dims=None, compression='zstd'):
    """
    Function to combine hdf5 files with flattened datasets within a single group.

    Parameters
    ----------
    paths : list of str, pathlib.Path, or io.BytesIO
        The list of input hdf5s to combine.
    new_path : str or io.BytesIO
        The output path of the new combined hdf5 fie.
    group : str or None
        The group or group path within the hdf5 file to the datasets.
    chunks : dict of tuples
        The chunks per dataset. Must be a dictionary of dataset name keys with tuple values of appropriate dimensions. A value of None will perform auto-chunking.
    unlimited_dims : str, list of str, or None
        The dimensions that should be assigned as "unlimited".
    compression : str
        The compression used for the chunks in the hdf5 files. Must be one of gzip, lzf, zstd, or None.

    Returns
    -------
    None
    """
    if isinstance(unlimited_dims, str):
        unlimited_dims = [unlimited_dims]
    else:
        unlimited_dims = []

    compressor = utils.get_compressor(compression)

    ## Read paths input into the appropriate file objects
    files = utils.open_files(paths, group)

    ## Create new file
    with h5py.File(new_path, 'w', libver='latest', rdcc_nbytes=3*1024*1024) as nf:

        if isinstance(group, str):
            nf1 = nf.create_group(group)
        else:
            nf1 = nf

        ## Get the extended coords
        coords_dict = extend_coords(files)

        ## Add the coords as datasets
        for coord, arr in coords_dict.items():
            shape = arr.shape

            maxshape = tuple([s if s not in unlimited_dims else None for s in shape])

            chunks1 = utils.guess_chunk(shape, maxshape, arr.dtype)

            if isinstance(chunks, dict):
                if coord in chunks:
                    chunks1 = chunks[coord]

            ds = nf1.create_dataset(coord, shape, chunks=chunks1, maxshape=maxshape, dtype=arr.dtype, **compressor)

            ds[:] = arr

            ds.make_scale(coord)

        ## Add the variables as datasets
        vars_dict = extend_variables(files, coords_dict)

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

            # Load the data by chunk
            for i in vars_dict[var_name]['data']:
                file = files[i]
    
                ds_old = file[var_name]

                if isinstance(ds_old, xr.Dataset):
                    if ds.chunks is None:
                        ds[()] = ds_old.data
                    else:
                        source_slice_index = vars_dict[var_name]['data'][i]['slice_index']
                        dims_order = vars_dict[var_name]['data'][i]['dims_order']
        
                        source_dim_index = [dims_order.index(dim) for dim in dims]
                        source_slice_index = tuple(source_slice_index[pos] for pos in source_dim_index)
        
                        new_slices, source_slices = utils.copy_chunks_complex(shape, chunks1, source_slice_index, source_dim_index)
        
                        for new_slice, source_slice in zip(new_slices, source_slices):
                            # print(new_slice, source_slice)
                            data = ds_old[source_slice].copy().load()
                            if dims == dims_order:
                                ds[new_slice] = data.data
                            else:
                                ds[new_slice] = data.data.transpose(source_dim_index)
                            data.close()
                            del data
                else:

                    if ds.chunks is None:
                        ds[()] = ds_old[()]
                    else:
                        source_slice_index = vars_dict[var_name]['data'][i]['slice_index']
                        dims_order = vars_dict[var_name]['data'][i]['dims_order']
        
                        source_dim_index = [dims_order.index(dim) for dim in dims]
                        source_slice_index = tuple(source_slice_index[pos] for pos in source_dim_index)
        
                        new_slices, source_slices = utils.copy_chunks_complex(shape, chunks1, source_slice_index, source_dim_index)
        
                        for new_slice, source_slice in zip(new_slices, source_slices):
                            # print(new_slice, source_slice)
                            if dims == dims_order:
                                ds[new_slice] = ds_old[source_slice]
                            else:
                                ds[new_slice] = ds_old[source_slice].transpose(source_dim_index)

                # for chunk in ds.iter_chunks(slice_index):
                #     # print(chunk)
                #     source_chunk = tuple([slice(chunk[i].start - slice_index[i].start, chunk[i].stop - slice_index[i].start) for i in dims_index])
                #     ds[chunk] = ds_old[source_chunk]

        ## Assign attrs
        global_attrs = {}
        for file in files:
            if isinstance(file, xr.Dataset):
                ds_list = list(file.variables)
            else:
                ds_list = list(file.keys())

            for ds_name in ds_list:
                attrs = {k: v for k, v in file[ds_name].attrs.items() if k not in ['DIMENSION_LABELS', 'DIMENSION_LIST', 'CLASS', 'NAME', '_Netcdf4Coordinates', '_Netcdf4Dimid', 'REFERENCE_LIST']}
                # print(attrs)
                nf1[ds_name].attrs.update(attrs)

            global_attrs.update(dict(file.attrs))

        nf1.attrs.update(global_attrs)

    if isinstance(new_path, io.BytesIO):
        new_path.seek(0)

    utils.close_files(files)


def open_dataset(path, **kwargs):
    """
    The Xarray open_dataset function, but specifically with the h5netcdf engine to open hdf5 files.
    """
    ds = xr.open_dataset(path, engine='h5netcdf', cache=False, **kwargs)

    return ds


def load_dataset(path, **kwargs):
    """
    The Xarray load_dataset function, but specifically with the h5netcdf engine to open hdf5 files.
    """
    ds = xr.load_dataset(path, engine='h5netcdf', **kwargs)

    return ds