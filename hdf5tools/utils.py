#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 19:52:08 2022

@author: mike
"""
import io
import pathlib
import h5py
import os
import numpy as np
import xarray as xr
# from time import time
# from datetime import datetime
import cftime
# import dateutil.parser as dparser
# import numcodecs
import hdf5plugin


########################################################
### Parmeters


CHUNK_BASE = 32*1024    # Multiplier by which chunks are adjusted
CHUNK_MIN = 32*1024      # Soft lower limit (32k)
CHUNK_MAX = 3*1024*1024   # Hard upper limit (4M)

time_str_conversion = {'days': 'datetime64[D]',
                       'hours': 'datetime64[h]',
                       'minutes': 'datetime64[m]',
                       'seconds': 'datetime64[s]',
                       'milliseconds': 'datetime64[ms]'}

enc_fields = ('units', 'calendar', 'dtype', 'missing_value', '_FillValue', 'add_offset', 'scale_factor')

#########################################################
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
        if (calendar is None):
            raise TypeError('data is datetime, so calendar must be assigned.')
        output = encode_datetime(data, units, calendar)
    elif dtype == h5py.string_dtype():
        output = data.astype(h5py.string_dtype())
    elif isinstance(scale_factor, (int, float, np.number)):
        output = (data - add_offset)/scale_factor

        if isinstance(missing_value, (int, np.number)):
            output[np.isnan(output)] = missing_value

        output = output.astype(dtype)

    return output


def get_encoding(data):
    """

    """
    if isinstance(data, xr.DataArray):
        encoding = {f: v for f, v in data.encoding.items() if f in enc_fields}
    else:
        encoding = {f: v for f, v in data.attrs.items() if f in enc_fields}

    if (data.dtype.name == 'object') or ('str' in data.dtype.name):
        encoding['dtype'] = h5py.string_dtype()
    elif ('datetime64' in data.dtype.name) or ('calendar' in encoding):
        encoding['dtype'] = np.dtype('int64')
        encoding['calendar'] = 'gregorian'
        encoding['units'] = 'seconds since 1970-01-01 00:00:00'

    if 'dtype' not in encoding:
        encoding['dtype'] = data.dtype
    elif isinstance(encoding['dtype'], str):
        encoding['dtype'] = np.dtype(encoding['dtype'])

    if 'missing_value' in encoding:
        encoding['_FillValue'] = encoding['missing_value']

    return encoding


def get_encodings(files):
    """
    I should add checking across the files for conflicts at some point.
    """
    # file_encs = {}
    encs = {}
    for i, file in enumerate(files):
        # file_encs[i] = {}
        for name in file:
            enc = get_encoding(file[name])
            # file_encs[i].update({name: enc})

            if name in encs:
                encs[name].update(enc)
            else:
                encs[name] = enc

    return encs


def decode_data(data, dtype=None, missing_value=None, add_offset=0, scale_factor=1, units=None, calendar=None, **kwargs):
    """

    """
    if isinstance(calendar, str):
        output = decode_datetime(data, units, calendar)
    else:
        output = data.astype(dtype)

        if isinstance(missing_value, (int, np.number)):
            output[output == missing_value] = np.nan

        output = (output * scale_factor) + add_offset

    return output


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


def open_file(path, group=None):
    """

    """
    if isinstance(path, (str, pathlib.Path, io.BytesIO)):
        if isinstance(group, str):
            f = h5py.File(path, 'r')[group]
        else:
            f = h5py.File(path, 'r')
    elif isinstance(path, h5py.File):
        if isinstance(group, str):
            try:
                f = path[group]
            except:
                f = path
        else:
            f = path
    elif isinstance(path, xr.Dataset):
        f = path
    else:
        raise TypeError('path must be a str/pathlib path to an HDF5 file, an h5py.File, or an xarray Dataset.')

    return f


def open_files(paths, group=None):
    """

    """
    files = []
    append = files.append
    for path in paths:
        f = open_file(path, group)
        append(f)

    return files


def close_files(files):
    """
    
    """
    for f in files:
        f.close()
        if isinstance(f, xr.Dataset):
            del f
            xr.backends.file_manager.FILE_CACHE.clear()


def extend_coords_xr(coords_dict, file):
    """

    """
    ds_list = list(file.coords)

    for ds_name in ds_list:
        ds = file[ds_name]

        if ds.dtype.name == 'object':
            if ds_name in coords_dict:
                coords_dict[ds_name] = np.union1d(coords_dict[ds_name], ds.values).astype(h5py.string_dtype())
            else:
                coords_dict[ds_name] = ds.values.astype(h5py.string_dtype())
        else:
            if ds_name in coords_dict:
                coords_dict[ds_name] = np.union1d(coords_dict[ds_name], ds.values)
            else:
                coords_dict[ds_name] = ds.values


def extend_coords_hdf5(coords_dict, file):
    """

    """
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


def extend_coords(files):
    """

    """
    coords_dict = {}

    for file in files:
        if isinstance(file, xr.Dataset):
            extend_coords_xr(coords_dict, file)
        else:
            extend_coords_hdf5(coords_dict, file)

    return coords_dict


def extend_variables_hdf5(file, i, coords_dict, vars_dict):
    """
    
    """
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


def extend_variables_xr(file, i, coords_dict, vars_dict):
    """
    
    """
    ds_list = list(file.data_vars)

    for ds_name in ds_list:
        ds = file[ds_name]

        dims = []
        slice_index = []

        for dim in ds.dims:
            dims.append(dim)
            arr_index = np.where(np.isin(coords_dict[dim], ds[dim].data))[0]

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


def extend_variables(files, coords_dict):
    """
    
    """
    vars_dict = {}

    for i, file in enumerate(files):
        if isinstance(file, xr.Dataset):
            extend_variables_xr(file, i, coords_dict, vars_dict)
        else:
            extend_variables_hdf5(file, i, coords_dict, vars_dict)

    return vars_dict


def index_coords_hdf5(file, selection: dict):
    """

    """
    coords_list = [ds_name for ds_name in file if is_scale(file[ds_name])]
    
    index_coords_dict = {}
    
    for coord, sel in selection.items():
        if coord not in coords_list:
            raise ValueError(coord + ' is not in the coordinates of the hdf5 file.')
    
        attrs = dict(file[coord].attrs)
        enc = {k: v for k, v in attrs.items() if k in decode_data.__code__.co_varnames}
    
        arr = decode_data(file[coord][:], **enc)
    
        if isinstance(sel, slice):
            if 'datetime64' in arr.dtype.name:
                start = np.datetime64(sel.start, 's')
                end = np.datetime64(sel.stop, 's')
                bool_index = (start <= arr) & (arr < end)
            else:
                bool_index = (sel.start <= arr) & (arr < sel.stop)
    
        else:
            if isinstance(sel, (int, float)):
                sel = [sel]
    
            try:
                sel1 = np.array(sel)
            except:
                raise TypeError('selection input could not be coerced to an ndarray.')
    
            if sel1.dtype.name == 'bool':
                if sel1.shape[0] != arr.shape[0]:
                    raise ValueError('The boolean array does not have the same length as the coord array.')
                bool_index = sel1
            else:
                bool_index = np.in1d(arr, sel1)
    
        int_index = np.where(bool_index)[0]
        slice_index = slice(int_index.min(), int_index.max())
        len1 = slice_index.stop - slice_index.start
    
        index_coords_dict[coord] = {'int_index': int_index, 'slice_index': slice_index, 'slice_len': len1}

    return index_coords_dict


def index_coords_xr(file, selection: dict):
    """

    """
    coords_list = list(file.coords)
    
    index_coords_dict = {}
    
    for coord, sel in selection.items():
        if coord not in coords_list:
            raise ValueError(coord + ' is not in the coordinates of the hdf5 file.')
    
        attrs = dict(file[coord].attrs)
        enc = {k: v for k, v in attrs.items() if k in decode_data.__code__.co_varnames}
    
        arr = decode_data(file[coord].data, **enc)
    
        if isinstance(sel, slice):
            if 'datetime64' in arr.dtype.name:
                start = np.datetime64(sel.start, 's')
                end = np.datetime64(sel.stop, 's')
                bool_index = (start <= arr) & (arr < end)
            else:
                bool_index = (sel.start <= arr) & (arr < sel.stop)
    
        else:
            if isinstance(sel, (int, float)):
                sel = [sel]
    
            try:
                sel1 = np.array(sel)
            except:
                raise TypeError('selection input could not be coerced to an ndarray.')
    
            if sel1.dtype.name == 'bool':
                if sel1.shape[0] != arr.shape[0]:
                    raise ValueError('The boolean array does not have the same length as the coord array.')
                bool_index = sel1
            else:
                bool_index = np.in1d(arr, sel1)
    
        int_index = np.where(bool_index)[0]
        slice_index = slice(int_index.min(), int_index.max())
        len1 = slice_index.stop - slice_index.start
    
        index_coords_dict[coord] = {'int_index': int_index, 'slice_index': slice_index, 'slice_len': len1}

    return index_coords_dict



def guess_chunk(shape, maxshape, dtype):
    """ Guess an appropriate chunk layout for a dataset, given its shape and
    the size of each element in bytes.  Will allocate chunks only as large
    as MAX_SIZE.  Chunks are generally close to some power-of-2 fraction of
    each axis, slightly favoring bigger values for the last index.
    Undocumented and subject to change without warning.
    """

    if len(shape) > 0:

        # For unlimited dimensions we have to guess 1024
        shape1 = []
        for i, x in enumerate(maxshape):
            if x is None:
                if shape[i] > 1024:
                    shape1.append(shape[i])
                else:
                    shape1.append(1024)
            else:
                shape1.append(x)
    
        shape = tuple(shape1)
    
        ndims = len(shape)
        if ndims == 0:
            raise ValueError("Chunks not allowed for scalar datasets.")
    
        chunks = np.array(shape, dtype='=f8')
        if not np.all(np.isfinite(chunks)):
            raise ValueError("Illegal value in chunk tuple")
    
        # Determine the optimal chunk size in bytes using a PyTables expression.
        # This is kept as a float.
        typesize = dtype.itemsize
        # dset_size = np.product(chunks)*typesize
        # target_size = CHUNK_BASE * (2**np.log10(dset_size/(1024.*1024)))
    
        # if target_size > CHUNK_MAX:
        #     target_size = CHUNK_MAX
        # elif target_size < CHUNK_MIN:
        #     target_size = CHUNK_MIN
    
        target_size = CHUNK_MAX
    
        idx = 0
        while True:
            # Repeatedly loop over the axes, dividing them by 2.  Stop when:
            # 1a. We're smaller than the target chunk size, OR
            # 1b. We're within 50% of the target chunk size, AND
            #  2. The chunk is smaller than the maximum chunk size
    
            chunk_bytes = np.product(chunks)*typesize
    
            if (chunk_bytes < target_size or \
             abs(chunk_bytes-target_size)/target_size < 0.5) and \
             chunk_bytes < CHUNK_MAX:
                break
    
            if np.product(chunks) == 1:
                break  # Element size larger than CHUNK_MAX
    
            chunks[idx%ndims] = np.ceil(chunks[idx%ndims] / 2.0)
            idx += 1
    
        return tuple(int(x) for x in chunks)
    else:
        return None


def copy_chunks_simple(shape, chunks, factor=3):
    """

    """
    n_shapes = []

    copy_shape = tuple([s*factor if s*factor <= shape[i] else shape[i] for i, s in enumerate(chunks)])
    for i, s in enumerate(copy_shape):
        shapes = np.arange(0, shape[i], s)
        n_shapes.append(shapes)

    # cart = np.array(np.meshgrid(n_shapes)).T.reshape(-1, len(shape))
    cart = cartesian(n_shapes)

    slices = []
    append = slices.append
    for arr in cart:
        slices1 = tuple([slice(s, s + copy_shape[i]) if s + copy_shape[i] <= shape[i] else slice(s, shape[i]) for i, s in enumerate(arr)])
        append(slices1)

    source_slices = slices

    return slices, source_slices


def copy_chunks_complex(shape, chunks, source_slice_index, source_dim_index, factor=3):
    """

    """
    source_shapes = []
    new_shapes = []
    big_chunks = []
    for i, s in enumerate(chunks):
        s1 = s*factor

        ssi = source_slice_index[i]
        if isinstance(ssi, slice):
            len1 = ssi.stop - ssi.start
        else:
            len1 = len(ssi)

        if s1 <= len1:
            s2 = s1
        else:
            s2 = len1

        if isinstance(ssi, slice):
            shapes = np.arange(source_slice_index[i].start, source_slice_index[i].stop, s2)
            s_shapes = np.arange(0, source_slice_index[i].stop - source_slice_index[i].start, s2)
        else:
            shapes1 = [ssi[i * s2:(i + 1) * s2] for i in range((len(ssi) + s2 - 1) // s2 )]
            a = np.arange(0, len(ssi))
            shapes2 = [a[i * s2:(i + 1) * s2] for i in range((len(a) + s2 - 1) // s2 )]

            if len(shapes1) == 1:
                shapes = np.empty(1, dtype='object')
                shapes[:] = shapes1
                s_shapes = np.empty(1, dtype='object')
                s_shapes[:] = shapes2
            else:
                shapes = shapes1
                s_shapes = shapes2

        source_shapes.append(s_shapes)
        new_shapes.append(shapes)
        big_chunks.append(s2)

    try:
        cart = cartesian(new_shapes)
        source_cart = cartesian(source_shapes)
    except:
        cart = np.array(np.meshgrid(new_shapes)).T.reshape(-1, len(shape))
        source_cart = np.array(np.meshgrid(source_shapes)).T.reshape(-1, len(shape))

    slices = []
    append = slices.append
    for arr in cart:
        slices1 = []
        for i, val in enumerate(arr):
            if isinstance(val, np.ndarray):
                slice2 = val
            else:
                if val + big_chunks[i] <= source_slice_index[i].stop:
                    slice2 = slice(val, val + big_chunks[i])
                else:
                    slice2 = slice(val, source_slice_index[i].stop)

            slices1.append(slice2)

        append(tuple(slices1))

    source_slices = []
    append = source_slices.append
    for arr in source_cart:
        slices1 = []
        for i, val in enumerate(arr):
            if isinstance(val, np.ndarray):
                slice2 = val
            else:
                if val + big_chunks[i] <= source_slice_index[i].stop:
                    slice2 = slice(val, val + big_chunks[i])
                else:
                    slice2 = slice(val, source_slice_index[i].stop)

            slices1.append(slice2)

        append(tuple(slices1[source_dim_index.index(i)] for i in range(len(source_dim_index))))

    return slices, source_slices


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
            [1, 4, 7],
            [1, 5, 6],
            [1, 5, 7],
            [2, 4, 6],
            [2, 4, 7],
            [2, 5, 6],
            [2, 5, 7],
            [3, 4, 6],
            [3, 4, 7],
            [3, 5, 6],
            [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]

    return out


def get_compressor(name: str = None):
    """

    """
    if name is None:
        compressor = {}
    elif name == 'gzip':
        compressor = {'compression': name}
    elif name == 'lzf':
        compressor = {'compression': name}
    elif name == 'zstd':
        compressor = hdf5plugin.Zstd(1)
    else:
        raise ValueError('name must be one of gzip, lzf, zstd, or None.')

    return compressor





























































































