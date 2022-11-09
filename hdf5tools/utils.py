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

missing_value_dict = {'int8': -128, 'int16': -32768, 'int32': -2147483648, 'int64': -9223372036854775808}

#########################################################
### Functions


def encode_datetime(data, units=None, calendar='gregorian'):
    """

    """
    if units is None:
        output = data.astype('datetime64[s]').astype('int64')
    else:
        if '1970-01-01' in units:
            time_unit = units.split()[0]
            output = data.astype(time_str_conversion[time_unit]).astype('int64')
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

    if (data.dtype != dtype) or (data.dtype.name == 'object'):
        data = data.astype(dtype)

    return data


def decode_data(data, dtype_decoded, missing_value=None, add_offset=0, scale_factor=None, units=None, calendar=None, **kwargs):
    """

    """
    if isinstance(calendar, str):
        data = decode_datetime(data, units, calendar)

    elif isinstance(scale_factor, (int, float, np.number)):
        data = data.astype(dtype_decoded)

        if isinstance(missing_value, (int, np.number)):
            data[data == missing_value] = np.nan

        data = (data * scale_factor) + add_offset

    elif (data.dtype.name == 'object'):
        data = data.astype(str).astype(dtype_decoded)

    elif (data.dtype != dtype_decoded):
        data = data.astype(dtype_decoded)

    return data


def get_encoding(data):
    """

    """
    if isinstance(data, xr.DataArray):
        encoding = {f: v for f, v in data.encoding.items() if f in enc_fields}
    else:
        encoding = {}
        for f, v in data.attrs.items():
            if f in enc_fields:
                if isinstance(v, bytes):
                    encoding[f] = v.decode()
                else:
                    encoding[f] = v

    if (data.dtype.name == 'object') or ('str' in data.dtype.name):
        encoding['dtype'] = h5py.string_dtype()
    elif ('datetime64' in data.dtype.name): # which means it's an xr.DataArray
        encoding['dtype'] = np.dtype('int64')
        encoding['calendar'] = 'gregorian'
        encoding['units'] = 'seconds since 1970-01-01 00:00:00'
        encoding['missing_value'] = missing_value_dict['int64']
        encoding['_FillValue'] = encoding['missing_value']

    elif ('calendar' in encoding): # Which means it's not an xr.DataArray
        encoding['dtype'] = np.dtype('int64')
        if 'units' not in encoding:
            encoding['units'] = 'seconds since 1970-01-01 00:00:00'
        encoding['missing_value'] = missing_value_dict['int64']
        encoding['_FillValue'] = encoding['missing_value']

    if 'dtype' not in encoding:
        encoding['dtype'] = data.dtype
    elif isinstance(encoding['dtype'], str):
        encoding['dtype'] = np.dtype(encoding['dtype'])

    if 'scale_factor' in encoding:
        if not isinstance(encoding['scale_factor'], (int, float, np.number)):
            raise TypeError('scale_factor must be an int or float.')

        if not np.issubdtype(encoding['dtype'], np.integer):
            raise TypeError('If scale_factor is assigned, then the dtype must be a np.integer.')

    if 'int' in encoding['dtype'].name:
        if ('_FillValue' in encoding) and ('missing_value' not in encoding):
            encoding['missing_value'] = encoding['_FillValue']

        if 'missing_value' not in encoding:
            encoding['missing_value'] = missing_value_dict[encoding['dtype'].name]
            encoding['_FillValue'] = encoding['missing_value']

    return encoding


def assign_dtype_decoded(encoding):
    """

    """
    if encoding['dtype'] == h5py.string_dtype():
        encoding['dtype_decoded'] = encoding['dtype']
    elif ('calendar' in encoding) and ('units' in encoding):
        encoding['dtype_decoded'] = np.dtype('datetime64[s]')

    if 'scale_factor' in encoding:

        # if isinstance(encoding['scale_factor'], (int, np.integer)):
        #     encoding['dtype_decoded'] = np.dtype('float32')
        if encoding['dtype'].itemsize > 2:
            encoding['dtype_decoded'] = np.dtype('float64')
        else:
            encoding['dtype_decoded'] = np.dtype('float32')

    if 'dtype_decoded' not in encoding:
        encoding['dtype_decoded'] = encoding['dtype']

    return encoding


def get_encodings(files):
    """
    I should add checking across the files for conflicts at some point.
    """
    # file_encs = {}
    encs = {}
    for i, file in enumerate(files):
        # file_encs[i] = {}
        if isinstance(file, xr.Dataset):
            ds_list = list(file.variables)
        else:
            ds_list = list(file.keys())

        for name in ds_list:
            enc = get_encoding(file[name])
            enc = assign_dtype_decoded(enc)
            # file_encs[i].update({name: enc})

            if name in encs:
                encs[name].update(enc)
            else:
                encs[name] = enc

        for name, enc in encs.items():
            enc = assign_dtype_decoded(enc)
            encs[name] = enc

    return encs


def get_attrs(files):
    """

    """
    # file_attrs = {}
    global_attrs = {}
    attrs = {}
    for i, file in enumerate(files):
        global_attrs.update(dict(file.attrs))

        # file_attrs[i] = {}
        for name in file:
            attr = {f: v for f, v in file[name].attrs.items() if (f not in enc_fields) and (f not in ['DIMENSION_LABELS', 'DIMENSION_LIST', 'CLASS', 'NAME', '_Netcdf4Coordinates', '_Netcdf4Dimid', 'REFERENCE_LIST'])}
            # file_attrs[i].update({name: attr})

            if name in attrs:
                attrs[name].update(attr)
            else:
                attrs[name] = attr

    return attrs, global_attrs


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
    elif isinstance(path, bytes):
        if isinstance(group, str):
            f = h5py.File(io.BytesIO(path), 'r')[group]
        else:
            f = h5py.File(io.BytesIO(path), 'r')
    else:
        raise TypeError('path must be a str/pathlib path to an HDF5 file, an h5py.File, a bytes object of an HDF5 file, or an xarray Dataset.')

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


def extend_coords(files, encodings):
    """

    """
    coords_dict = {}

    for file in files:
        if isinstance(file, xr.Dataset):
            ds_list = list(file.coords)
        else:
            ds_list = [ds_name for ds_name in file.keys() if is_scale(file[ds_name])]

        for ds_name in ds_list:
            ds = file[ds_name]

            if isinstance(file, xr.Dataset):
                # data = decode_data(ds.values, **encodings[ds_name])
                data = ds.values
            else:
                data = decode_data(ds[:], **encodings[ds_name])

            if ds_name in coords_dict:
                coords_dict[ds_name] = np.union1d(coords_dict[ds_name], data)
            else:
                coords_dict[ds_name] = data

    return coords_dict


def index_variables(files, coords_dict, encodings):
    """

    """
    vars_dict = {}

    for i, file in enumerate(files):

        if isinstance(file, xr.Dataset):
            ds_list = list(file.data_vars)
        else:
            ds_list = [ds_name for ds_name in file.keys() if not is_scale(file[ds_name])]

        for ds_name in ds_list:
            ds = file[ds_name]

            var_enc = encodings[ds_name]

            dims = []
            global_index = []
            local_index = []
            remove_ds = False

            for dim in ds.dims:
                if isinstance(file, xr.Dataset):
                    dim_name = dim
                    # dim_data = decode_data(ds[dim_name].values, **encodings[dim_name])
                    dim_data = ds[dim_name].values
                else:
                    dim_name = dim[0].name.split('/')[-1]
                    dim_data = decode_data(dim[0][:], **encodings[dim_name])

                dims.append(dim_name)

                global_arr_index = np.where(np.isin(coords_dict[dim_name], dim_data))[0]
                local_arr_index = np.where(np.isin(dim_data, coords_dict[dim_name]))[0]

                if len(global_arr_index) > 0:

                    if is_regular_index(global_arr_index):
                        slice1 = slice(global_arr_index.min(), global_arr_index.max() + 1)
                        global_index.append(slice1)
                    else:
                        global_index.append(global_arr_index)

                    if is_regular_index(local_arr_index):
                        slice1 = slice(local_arr_index.min(), local_arr_index.max() + 1)
                        local_index.append(slice1)
                    else:
                        local_index.append(local_arr_index)
                else:
                    remove_ds = True
                    break

            if remove_ds:
                if ds_name in vars_dict:
                    if i in vars_dict[ds_name]['data']:
                        del vars_dict[ds_name]['data'][i]

            else:
                dict1 = {'dims_order': tuple(i for i in range(len(dims))), 'global_index': global_index, 'local_index': local_index}

                if ds_name in vars_dict:
                    if not np.in1d(vars_dict[ds_name]['dims'], dims).all():
                        raise ValueError('dims are not consistant between the same named dataset: ' + ds_name)
                    # if vars_dict[ds_name]['dtype'] != ds.dtype:
                    #     raise ValueError('dtypes are not consistant between the same named dataset: ' + ds_name)

                    dims_order = [vars_dict[ds_name]['dims'].index(dim) for dim in dims]
                    dict1['dims_order'] = tuple(dims_order)
                    dict1['global_index'] = [dict1['global_index'][dims_order.index(i)] for i in range(len(dims_order))]
                    dict1['local_index'] = [dict1['local_index'][dims_order.index(i)] for i in range(len(dims_order))]

                    vars_dict[ds_name]['data'][i] = dict1
                else:
                    shape = tuple([coords_dict[dim_name].shape[0] for dim_name in dims])

                    if 'missing_value' in var_enc:
                        fillvalue = var_enc['missing_value']
                    else:
                        fillvalue = None

                    vars_dict[ds_name] = {'data': {i: dict1}, 'dims': tuple(dims), 'shape': shape, 'dtype': var_enc['dtype'], 'fillvalue': fillvalue, 'dtype_decoded': var_enc['dtype_decoded']}

    return vars_dict


# def index_coords_file(file, coords_dict, encodings, selection: dict):
#     """

#     """
#     index_coords_dict = {}

#     for coord in coords_dict:
#         if coord in selection:
#             sel = selection[coord]

#             coord_enc = encodings[coord]

#             if isinstance(file, xr.Dataset):
#                 arr = decode_data(file[coord].values, **coord_enc)
#             else:
#                 arr = decode_data(file[coord][:], **coord_enc)

#             if isinstance(sel, slice):
#                 if 'datetime64' in arr.dtype.name:
#                     if not isinstance(sel.start, (str, np.datetime64)):
#                         raise TypeError('Input for datetime selection should be either a datetime string or np.datetime64.')
#                     start = np.datetime64(sel.start, 's')
#                     end = np.datetime64(sel.stop, 's')
#                     bool_index = (start <= arr) & (arr < end)
#                 else:
#                     bool_index = (sel.start <= arr) & (arr < sel.stop)

#             else:
#                 if isinstance(sel, (int, float)):
#                     sel = [sel]

#                 try:
#                     sel1 = np.array(sel)
#                 except:
#                     raise TypeError('selection input could not be coerced to an ndarray.')

#                 if sel1.dtype.name == 'bool':
#                     if sel1.shape[0] != arr.shape[0]:
#                         raise ValueError('The boolean array does not have the same length as the coord array.')
#                     bool_index = sel1
#                 else:
#                     bool_index = np.in1d(arr, sel1)

#             arr_index = np.where(bool_index)[0]

#             if len(arr_index) > 0:
#                 if is_regular_index(arr_index):
#                     slice_index = slice(arr_index.min(), arr_index.max() + 1)
#                 else:
#                     slice_index = arr_index
#             else:
#                 return None

#         else:
#             slice_index = slice(None, None)

#         index_coords_dict[coord] = slice_index

#     return index_coords_dict


def filter_coords(files, coords_dict, selection):
    """

    """
    for coord, sel in selection.items():
        if coord not in coords_dict:
            raise ValueError(coord + ' one of the coordinates.')

        coord_data = coords_dict[coord]

        if isinstance(sel, slice):
            if 'datetime64' in coord_data.dtype.name:
                if not isinstance(sel.start, (str, np.datetime64)):
                    raise TypeError('Input for datetime selection should be either a datetime string or np.datetime64.')
                start = np.datetime64(sel.start, 's')
                end = np.datetime64(sel.stop, 's')
                bool_index = (start <= coord_data) & (coord_data < end)
            else:
                bool_index = (sel.start <= coord_data) & (coord_data < sel.stop)

        else:
            if isinstance(sel, (int, float)):
                sel = [sel]

            try:
                sel1 = np.array(sel)
            except:
                raise TypeError('selection input could not be coerced to an ndarray.')

            if sel1.dtype.name == 'bool':
                if sel1.shape[0] != coord_data.shape[0]:
                    raise ValueError('The boolean array does not have the same length as the coord array.')
                bool_index = sel1
            else:
                bool_index = np.in1d(coord_data, sel1)

        new_coord_data = coord_data[bool_index]

        coords_dict[coord] = new_coord_data








# def index_coords(files, coords_dict, vars_dict, encodings, selection):
#     """

#     """
#     sel_dict = {}

#     for i, file in enumerate(files):
#         index_coords_dict = {}

#         for coord in coords_dict:
#             if coord in selection:
#                 sel = selection[coord]

#                 coord_enc = encodings[coord]

#                 if isinstance(file, xr.Dataset):
#                     arr = decode_data(file[coord].values, **coord_enc)
#                 else:
#                     arr = decode_data(file[coord][:], **coord_enc)

#                 if isinstance(sel, slice):
#                     if 'datetime64' in arr.dtype.name:
#                         if not isinstance(sel.start, (str, np.datetime64)):
#                             raise TypeError('Input for datetime selection should be either a datetime string or np.datetime64.')
#                         start = np.datetime64(sel.start, 's')
#                         end = np.datetime64(sel.stop, 's')
#                         bool_index = (start <= arr) & (arr < end)
#                     else:
#                         bool_index = (sel.start <= arr) & (arr < sel.stop)

#                 else:
#                     if isinstance(sel, (int, float)):
#                         sel = [sel]

#                     try:
#                         sel1 = np.array(sel)
#                     except:
#                         raise TypeError('selection input could not be coerced to an ndarray.')

#                     if sel1.dtype.name == 'bool':
#                         if sel1.shape[0] != arr.shape[0]:
#                             raise ValueError('The boolean array does not have the same length as the coord array.')
#                         bool_index = sel1
#                     else:
#                         bool_index = np.in1d(arr, sel1)

#                 arr_index = np.where(bool_index)[0]

#                 if len(arr_index) > 0:
#                     if is_regular_index(arr_index):
#                         slice_index = slice(arr_index.min(), arr_index.max() + 1)
#                     else:
#                         slice_index = arr_index
#                 else:
#                     slice_index = None

#             else:
#                 slice_index = slice(None, None)

#             index_coords_dict[coord] = slice_index

#         sel_dict[i] = index_coords_dict

#     sel_dict1 = {}
#     for k, v in sel_dict.items():
#         if None in v.values():
#             sel_dict1[k] = None
#         else:
#             sel_dict1[k] = v

#     return sel_dict1


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


def index_chunks(shape, chunks, global_index, local_index, dims_order, factor=3):
    """

    """
    local_shapes = []
    global_shapes = []

    for i, s in enumerate(chunks):
        chunk_size = s*factor

        g_slices, l_slices = array_index_to_slices(global_index[i], local_index[i], chunk_size)

        global_shapes.append(g_slices)
        local_shapes.append(l_slices)

    try:
        global_cart = cartesian(global_shapes)
        local_cart = cartesian(local_shapes)
    except:
        global_cart = np.array(np.meshgrid(global_shapes)).T.reshape(-1, len(shape))
        local_cart = np.array(np.meshgrid(local_shapes)).T.reshape(-1, len(shape))

    global_slices = [tuple(g) for g in global_cart]

    # local_order = tuple(dims_order.index(i) for i in range(len(dims_order)))
    local_slices = []
    append = local_slices.append
    for l in local_cart:
        append(tuple(l[i] for i in dims_order))

    return global_slices, local_slices


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


def array_index_to_slices(g_arr, l_arr, chunk_size):
    """

    """
    # if isinstance(g_arr, slice) and isinstance(l_arr, slice):
    #     return [g_arr], [l_arr]

    if isinstance(l_arr, slice):
        l_start = l_arr.start
        l_stop = l_arr.stop
        l_arr = np.arange(l_start, l_stop)
    # else:
    #     l_start = l_arr.min()
    #     l_stop = l_arr.max()
        # reg1 = np.append(np.diff(l_arr) != 1, True)

    if isinstance(g_arr, slice):
        g_start = g_arr.start
        g_stop = g_arr.stop
        g_arr = np.arange(g_start, g_stop)
    else:
        g_start = g_arr.min()
        g_stop = g_arr.max()

    reg1 = np.append(np.diff(g_arr) != 1, True)

    chunk_start = (g_start//chunk_size) * chunk_size
    chunk_stop = ((g_stop // chunk_size) + 1) * chunk_size

    chunk_stops = np.arange(chunk_start, chunk_stop, chunk_size) - 1
    chunk_stop_pos = np.in1d(g_arr, chunk_stops)

    reg2 = reg1 + chunk_stop_pos

    stop_pos = np.where(reg2)[0]

    g_reg_list = []
    l_reg_list = []
    for i, pos in enumerate(stop_pos):
        if i == 0:
            g_start = g_arr[0]
            l_start = l_arr[0]
        else:
            g_start = g_arr[stop_pos[i-1]+1]
            l_start = l_arr[stop_pos[i-1]+1]
        g_stop = g_arr[pos]+1
        l_stop = l_arr[pos]+1
        g_reg_list.append(slice(g_start, g_stop))
        l_reg_list.append(slice(l_start, l_stop))

    return g_reg_list, l_reg_list






























































































