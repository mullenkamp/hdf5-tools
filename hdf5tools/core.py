#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:11:20 2023

@author: mike
"""
import h5py
import io
import os
import numpy as np
import hdf5plugin
from typing import Union, List
import pathlib
# import copy
import tempfile

# from hdf5tools import utils

import utils

h5py.get_config().track_order = True

###################################################
### Parameters

name = '/media/data01/cache/tethys/test/test1.h5'


###################################################
### Classes


class File:
    """

    """
    def __init__(self, name: Union[str, pathlib.Path, io.BytesIO]=None, mode: str='r', compression: str='zstd', **kwargs):
        """
        The top level object for managing hdf5 data. Is equivalent to the h5py.File object.

        Parameters
        ----------
        name : str, pathlib.Path, io.BytesIO, or None
            A str or pathlib.Path object to a file on disk, a BytesIO object, or None. If None, it will create an in-memory hdf5 File.
        mode : str
            The typical python open mode.
        **kwargs
            Any other kwargs that will be passed to the h5py.File object.
        """
        if name is None:
            name = tempfile.NamedTemporaryFile()
            kwargs.setdefault('driver', 'core')
            if 'backing_store' not in kwargs:
                kwargs.setdefault('backing_store', False)
            file = h5py.File(name=name.name, mode='w', **kwargs)
        else:
            file = h5py.File(name=name, mode=mode, **kwargs)

        # if group is not None:
        #     if group in file:
        #         grp = file[group]
        #     else:
        #         grp = file.create_group(group)
        # else:
        #     grp = file['/']

        self._file = file
        # self._grp = grp
        self.filename = file.filename

        for ds_name in file:
            ds = file[ds_name]
            if utils.is_scale(ds):
                Dimension(ds, self, ds[()])
            else:
                Dataset(ds, self)


    def datasets(self):
        for name in self:
            if isinstance(self[name], Dataset):
                yield name

    def dimensions(self):
        for name in self:
            if isinstance(self[name], Dimension):
                yield name

    def __bool__(self):
        """

        """
        self._file.__bool__()

    def __iter__(self):
        return self._file.__iter__()

    def __len__(self):
        return len(self._file)

    def __contains__(self, key):
        return key in self._file

    def __getitem__(self, key):
        if key in self._file:
            return getattr(self, key)
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if isinstance(value, (Dataset, Dimension)):
            setattr(self, key, value)
        else:
            raise TypeError('value must be a Dataset or Dimension object.')

    def __delitem__(self, key):
        try:
            if key not in self:
                raise KeyError(key)

            # Check if the object to delete is a dimension
            # And if it is, check that no datasets are attached to it
            if isinstance(self[key], Dimension):
                for ds_name in self.datasets:
                    if key in self[ds_name].dim_names:
                        raise ValueError(f'{key} is a dimension of {ds_name}. You must delete all datasets associated with a dimension before you can delete the dimension.')

            del self._file[key]
            delattr(self, key)
        except Exception as err:
            raise err

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._file.__exit__()

    def close(self):
        self._file.close()

    def flush(self):
        """

        """
        self._file.flush()


    def __repr__(self):
        """

        """
        return self._file.__repr__()

    def sel(self):
        """

        """

    def to_pandas(self):
        """

        """

    def to_xarray(self):
        """

        """

    def to_file(self):
        """

        """

    def copy(self, name: Union[str, pathlib.Path, io.BytesIO]=None, mode: str='r', compression: str='zstd', **kwargs):
        """
        Copy a file object. kwargs can be any parameter for File.
        """
        # kwargs.setdefault('mode', 'w')
        file = File(name, mode='w', compression=compression, **kwargs)

        ## Create dimensions
        for dim_name in self.dimensions():
            dim = self[dim_name]
            _ = file.create_dimension_like(dim.name, dim, include_data=True)

        ## Create datasets
        for ds_name in self.datasets():
            ds = self[ds_name]
            _ = file.create_dataset_like(ds.name, ds, include_data=True)

        return file


    def create_dimension(self, name, data=None, **kwargs):
        """

        """
        dimension, data = create_h5py_dimension(self, name, data=data, **kwargs)
        dim = Dimension(dimension, self, data)

        return dim


    def create_dataset(self, name, dims, shape=None, dtype=None, data=None, **kwargs):
        """

        """
        ds0 = create_h5py_dataset(self, name, dims, shape=shape, dtype=dtype, data=data, **kwargs)
        ds = Dataset(ds0, self)

        return ds


    def create_dataset_like(self, name, other, include_data=False, **kwargs):
        """ Create a dataset similar to `other`.

        name
            Name of the dataset (absolute or relative).  Provide None to make
            an anonymous dataset.
        other
            The dataset which the new dataset should mimic. All properties, such
            as shape, dtype, chunking, ... will be taken from it, but no data
            or attributes are being copied.

        Any dataset keywords (see create_dataset) may be provided, including
        shape and dtype, in which case the provided values take precedence over
        those from `other`.
        """
        if not isinstance(other, Dataset):
            raise TypeError('other must be a Dataset.')

        other1 = other._dataset
        for k in ('shape', 'dtype', 'chunks', 'compression',
                  'compression_opts', 'scaleoffset', 'shuffle', 'fletcher32',
                  'fillvalue'):
            kwargs.setdefault(k, getattr(other1, k))
        # TODO: more elegant way to pass these (dcpl to create_dataset?)
        dcpl = other1.id.get_create_plist()
        kwargs.setdefault('track_times', dcpl.get_obj_track_times())
        kwargs.setdefault('track_order', dcpl.get_attr_creation_order() > 0)

        # Special case: the maxshape property always exists, but if we pass it
        # to create_dataset, the new dataset will automatically get chunked
        # layout. So we copy it only if it is different from shape.
        if other1.maxshape != other1.shape:
            kwargs.setdefault('maxshape', other1.maxshape)

        if include_data:
            ds0 = create_h5py_dataset(self, name, tuple(dim.label for dim in other1.dims),  **kwargs)

            # Directly copy chunks using write_direct_chunk
            for chunk in ds0.iter_chunks():
                chunk_starts = tuple(c.start for c in chunk)
                _, data = other1.id.read_direct_chunk(chunk_starts)
                ds0.id.write_direct_chunk(chunk_starts, data)

        else:
            ds0 = create_h5py_dataset(self, name, tuple(dim.label for dim in other1.dims), **kwargs)

        ds = Dataset(ds0, self)

        return ds


    # def create_dimension_like(self, name, other, include_data=False, **kwargs):
    #     """ Create a dimension similar to `other`.

    #     name
    #         Name of the dataset (absolute or relative).  Provide None to make
    #         an anonymous dataset.
    #     other
    #         The dataset which the new dataset should mimic. All properties, such
    #         as shape, dtype, chunking, ... will be taken from it, but no data
    #         or attributes are being copied.

    #     Any dataset keywords (see create_dataset) may be provided, including
    #     shape and dtype, in which case the provided values take precedence over
    #     those from `other`.
    #     """
    #     if not isinstance(other, Dimension):
    #         raise TypeError('other must be a Dimension.')

    #     other1 = other._dataset
    #     for k in ('shape', 'dtype', 'chunks', 'compression',
    #               'compression_opts', 'scaleoffset', 'shuffle', 'fletcher32',
    #               'fillvalue'):
    #         kwargs.setdefault(k, getattr(other1, k))
    #     # TODO: more elegant way to pass these (dcpl to create_dataset?)
    #     dcpl = other1.id.get_create_plist()
    #     kwargs.setdefault('track_times', dcpl.get_obj_track_times())
    #     kwargs.setdefault('track_order', dcpl.get_attr_creation_order() > 0)

    #     # Special case: the maxshape property always exists, but if we pass it
    #     # to create_dataset, the new dataset will automatically get chunked
    #     # layout. So we copy it only if it is different from shape.
    #     if other1.maxshape != other1.shape:
    #         kwargs.setdefault('maxshape', other1.maxshape)

    #     if include_data:
    #         ds0 = create_h5py_dimension(self, name, **kwargs)

    #         # Directly copy chunks using write_direct_chunk
    #         for chunk in ds0.iter_chunks():
    #             chunk_starts = tuple(c.start for c in chunk)
    #             _, data = other1.id.read_direct_chunk(chunk_starts)
    #             ds0.id.write_direct_chunk(chunk_starts, data)

    #     else:
    #         ds0, data = create_h5py_dimension(self, name, **kwargs)

    #     ds = Dimension(ds0, self, data)

    #     return ds


def create_h5py_dataset(file: File, name, dims, shape=None, dtype=None, data=None, **kwargs):
    """

    """
    if data is None:
        if (shape is None) or (dtype is None):
            raise ValueError('shape and dtype must be passed or data must be passed.')
    else:
        shape = data.shape
        dtype = data.dtype

    ## Check if dims already exist and if the dim lengths match
    for i, dim in enumerate(dims):
        if dim not in file:
            raise ValueError(f'{dim} not in File-Group')

        dim_len = file._file[dim].shape[0]
        if dim_len != shape[i]:
            raise ValueError(f'{dim} does not have the same length as the input data/shape dim.')

    ## Make chunks
    if 'chunks' not in kwargs:
        if 'maxshape' in kwargs:
            maxshape = kwargs['maxshape']
        else:
            maxshape = shape
        kwargs.setdefault('chunks', utils.guess_chunk(shape, maxshape, dtype))

    ## Create dataset
    if data is None:
        ds = file._file.create_dataset(name, shape, dtype=dtype, **kwargs)
    else:
        ds = file._file.create_dataset(name, dtype=dtype, data=data, **kwargs)

    for i, dim in enumerate(dims):
        ds.dims[i].attach_scale(file._file[dim])
        ds.dims[i].label = dim

    return ds


def create_h5py_dimension(file: File, name, data, **kwargs):
    """

    """
    shape = data.shape
    if 'dtype' in kwargs:
        dtype = kwargs['dtype']
    else:
        dtype = data.dtype

    if len(shape) != 1:
        raise ValueError('The shape of a dimension must be 1-D.')

    ## Make chunks
    if 'chunks' not in kwargs:
        if 'maxshape' in kwargs:
            maxshape = kwargs['maxshape']
        else:
            maxshape = shape
        kwargs.setdefault('chunks', utils.guess_chunk(shape, maxshape, dtype))

    ds = file._file.create_dataset(name, dtype=dtype, data=data, **kwargs)

    ds.make_scale(name)
    ds.dims[0].label = name

    return ds, data


class Dataset:
    """

    """
    def __init__(self, dataset: h5py.Dataset, file: File):
        """

        """
        self._dataset = dataset
        self.dim_names = tuple(dim.label for dim in dataset.dims)
        self.name = dataset.name.split('/')[-1]
        self.file = file
        setattr(file, self.name, self)
        self.loc = LocationIndexer(self)


    def __getitem__(self, key):
        return self._dataset[key]


    def iter_chunks(self, sel=None):
        return self._dataset.iter_chunks(sel)


    def to_pandas(self):
        """

        """

    def to_xarray(self):
        """

        """

    def copy(self, name, include_data=True, **kwargs):
        """
        Copy a Dataset object. Same as create_dataset_like.
        """
        ds = self.file.create_dataset_like(name, self, include_data=True, **kwargs)

        return ds


    def __repr__(self):
        """

        """
        return self._dataset.__repr__()

    def sel(self):
        """

        """



class Dimension(Dataset):
    """

    """
    def __init__(self, dataset: h5py.Dataset, file: File, data: np.ndarray):
        """

        """
        self._dataset = dataset
        self.dim_names = tuple(dim.label for dim in dataset.dims)
        self.name = dataset.name.split('/')[-1]
        self.file = file
        self.data = data
        setattr(file, self.name, self)


    def copy(self, name, include_data=True, **kwargs):
        """
        Copy a Dataset object. Same as create_dimension_like.
        """
        ds = self.file.create_dimension_like(name, self, include_data=True, **kwargs)

        return ds


    # def iter_chunks(self, sel=None):
    #     return self._dataset.iter_chunks(sel)

    def __repr__(self):
        """

        """
        return self._dataset.__repr__()

    def sel(self):
        """

        """

    def to_pandas(self):
        """

        """

    def to_xarray(self):
        """

        """


class LocationIndexer:
    """

    """
    def __init__(self, dataset):
        """

        """
        self.dataset = dataset


    def __getitem__(self, key):
        """

        """
        if isinstance(key, slice):
            dim = self.dataset.file[self.dataset.dim_names[0]]
            slice_idx = index_slice(key, dim.data)

            return self.dataset[slice_idx]

        elif isinstance(key, tuple):
            types = tuple(type(k) for k in key)

        else:
            # Do your handling for a plain index
            print("plain", key)



def index_slice(slice_obj, dim_data):
    """

    """
    start = slice_obj.start
    stop = slice_obj.stop

    if start not in dim_data:
        raise ValueError(f'{start} not in dimension.')
    if stop not in dim_data:
        raise ValueError(f'{stop} not in dimension.')

    start_idx = np.argwhere(dim_data == start)[0][0]
    stop_idx = np.argwhere(dim_data == stop)[0][0]

    if start_idx > stop_idx:
        raise ValueError(f'start index at {start_idx} is after stop index at {stop_idx}.')

    # start_stop_idx = np.searchsorted(dim_data, [start, stop])

    # if start > dim_data[-1]:
    #     raise ValueError('start is greater than max dim data.')
    # if stop < dim_data[0]:
    #     raise ValueError('stop is less than min dim data.')

    # start_idx = np.searchsorted(dim_data, start)
    # stop_idx = (np.abs(dim_data - stop)).argmin()

    return slice(start_idx, stop_idx)

















d




##############################################
### Testing

nc1 = '/media/data01/cache/tethys/test/2m_temperature_1950-1957_reanalysis-era5-land.nc'
nc2 = '/media/data01/cache/tethys/test/test2.nc'
nc3 = '/media/data01/cache/tethys/test/test3.nc'

dims = ('dim1', 'dim2')
dim1_len = 100
dim2_len = 10000

dim1_data = np.arange(dim1_len, dim1_len*2, dtype='int16')
dim2_data = np.arange(dim2_len, dim2_len*2, dtype='int16')

self = File(name, 'w')

# if 'dim1' in self:
#     del self['dim1']
# if 'dim2' in self:
#     del self['dim2']
# if 'test1' in self:
#     del self['test1']

dim1_test = self.create_dimension('dim1', data=dim1_data)
dim2_test = self.create_dimension('dim2', data=dim2_data)

test1_data = np.arange(dim1_len*dim2_len).reshape(dim1_len, dim2_len)

test1_ds = self.create_dataset('test1', dims, data=test1_data)

test2_ds = self.create_dataset_like('test2', test1_ds, include_data=True)

self = h5py.File(name, 'w', userblock_size=512)
self = h5py.File(name, 'r')

with open(name, 'rb') as f:
    b1 = f.read()



self.close()


x1 = xr.open_dataset(nc1)
xr_to_hdf5(x1, nc2)

file = h5py.File(nc2, rdcc_nbytes=0)
ds = file['t2m']
chunk_size = ds.chunks
chunks = list(ds.iter_chunks())
chunks_lon = chunks[:8]

arr = np.zeros((4383, 17, 124), 'int16')

def get_data(ds, chunks, arr):
    """

    """
    for chunk in chunks:
        arr[chunk] = ds[chunk]



ds[h5py.MultiBlockSlice(start=0, count=4383, stride=1, block=1), h5py.MultiBlockSlice(start=0, count=17, stride=1, block=1), h5py.MultiBlockSlice(start=0, count=124, stride=1, block=1)].shape

ds[h5py.MultiBlockSlice(start=0, count=1, stride=1, block=4383), h5py.MultiBlockSlice(start=0, count=1, stride=1, block=17), h5py.MultiBlockSlice(start=0, count=1, stride=1, block=124)].shape




self = File(nc2)

file = self.copy(nc3)

file2 = self.copy()



self = File(name)
















