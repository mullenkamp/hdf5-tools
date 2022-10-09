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


##############################################
### Parameters



##############################################
### Functions


def create_nc_dataset(hdf, xr_dataset, var_name, chunks, compressor, unlimited_dims):
    """

    """
    shape = xr_dataset[var_name].shape
    dims = xr_dataset[var_name].dims
    maxshape = tuple([s if dims[i] not in unlimited_dims else None for i, s in enumerate(shape)])

    encoding = xr_dataset[var_name].encoding.copy()

    if (xr_dataset[var_name].dtype.name == 'object') or ('str' in xr_dataset[var_name].dtype.name):
        xr_dataset[var_name] = xr_dataset[var_name].astype(h5py.string_dtype())
        encoding['dtype'] = h5py.string_dtype()
    elif 'datetime64' in xr_dataset[var_name].dtype.name:
        encoding['dtype'] = np.dtype('int64')
        encoding['calendar'] = 'gregorian'

    if 'dtype' not in encoding:
        encoding['dtype'] = xr_dataset[var_name].dtype
    elif isinstance(encoding['dtype'], str):
        encoding['dtype'] = np.dtype(encoding['dtype'])

    attrs = xr_dataset[var_name].attrs.copy()

    enc = {k: v for k, v in encoding.items() if k in utils.encode_data.__code__.co_varnames}

    if 'calendar' in enc:
        enc['units'] = 'seconds since 1970-01-01 00:00:00'

    if 'missing_value' in enc:
        enc['_FillValue'] = enc['missing_value']
        fillvalue = enc['missing_value']
    else:
        fillvalue = None

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
        ds[()] = xr_dataset[var_name].copy().load().values

    elif ('scale_factor' in enc) or ('add_offset' in enc) or ('calendar' in enc):
        if ds.chunks == shape:
            ds[:] = utils.encode_data(xr_dataset[var_name].copy().load().values, **enc)
        else:
            new_slices, source_slices = utils.copy_chunks_simple(shape, chunks1)

            for new_slice, source_slice in zip(new_slices, source_slices):
                # print(new_slice, source_slice)
                ds[new_slice] = utils.encode_data(xr_dataset[var_name][source_slice].copy().load().values, **enc)

    else:
        if ds.chunks == shape:
            ds[:] = xr_dataset[var_name].copy().load().values
        else:
            new_slices, source_slices = utils.copy_chunks_simple(shape, chunks1)

            for new_slice, source_slice in zip(new_slices, source_slices):
                # print(new_slice, source_slice)
                ds[new_slice] = xr_dataset[var_name][source_slice].copy().load().values

    _ = enc.pop('dtype')
    # print(enc)
    attrs.update(enc)

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


def combine_hdf5(paths, new_path, group=None, chunks=None, unlimited_dims=None, compression='zstd'):
    """
    Function to combine hdf5 files with flattened datasets within a single group.

    Parameters
    ----------
    paths : list of str
        The list of input hdf5 paths to combine.
    new_path : str
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

    ## Create new file
    with h5py.File(new_path, 'w', libver='latest', rdcc_nbytes=3*1024*1024) as nf:

        if isinstance(group, str):
            nf1 = nf.create_group(group)
        else:
            nf1 = nf

        ## Get the extended coords
        coords_dict = utils.extend_coords(paths, group)

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
        vars_dict = utils.extend_variables(paths, coords_dict, group)

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
            for path in vars_dict[var_name]['data']:
                with h5py.File(path, 'r') as f:

                    if isinstance(group, str):
                        f1 = f[group]
                    else:
                        f1 = f
    
                    ds_old = f1[var_name]

                    if ds.chunks is None:
                        ds[()] = ds_old[()]
                    else:
                        source_slice_index = vars_dict[var_name]['data'][path]['slice_index']
                        dims_order = vars_dict[var_name]['data'][path]['dims_order']
        
                        source_dim_index = [dims_order.index(dim) for dim in dims]
                        source_slice_index = tuple(source_slice_index[i] for i in source_dim_index)
        
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
        for path in paths:
            with h5py.File(path, 'r') as f:

                if isinstance(group, str):
                    f1 = f[group]
                else:
                    f1 = f
    
                ds_list = list(f1.keys())
    
                for ds_name in ds_list:
                    attrs = {k: v for k, v in f1[ds_name].attrs.items() if k not in ['DIMENSION_LABELS', 'DIMENSION_LIST', 'CLASS', 'NAME', '_Netcdf4Coordinates', '_Netcdf4Dimid', 'REFERENCE_LIST']}
                    # print(attrs)
                    nf1[ds_name].attrs.update(attrs)
    
                global_attrs.update(dict(f1.attrs))

        nf1.attrs.update(global_attrs)

    if isinstance(new_path, io.BytesIO):
        new_path.seek(0)


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

























######################################
### Testing
