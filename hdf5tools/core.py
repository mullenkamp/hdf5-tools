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
import copy
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
            return self.key
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        try:
            if key not in self:
                raise KeyError(key)

            # Check if the object to delete is a dimension
            # And if it is, check that no datasets are attached to it
            if isinstance(self[key], Dimension):
                for ds_name in self.datasets:
                    if key in self[ds_name].dims:
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

    def copy(self, **kwargs):
        """
        Copy a file object. kwargs can be any parameter for File.
        """


    def create_dimension(self, name, shape=None, dtype=None, data=None, **kwargs):
        """

        """
        dimension = Dimension(self, name, shape=shape, dtype=dtype, data=data, **kwargs)
        self[name] = dimension

        return dimension


    def create_dataset(self, name, dims, shape=None, dtype=None, data=None, **kwargs):
        """

        """
        ds = Dataset(self, name, dims, shape=shape, dtype=dtype, data=data, **kwargs)
        self[name] = ds

        return ds


    def create_dataset_like(self, name, other, **kwargs):
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

        ds = Dataset(self, name, tuple(dim.label for dim in other1.dims), **kwargs)

        return ds



class Dataset:
    """

    """
    def __init__(self, file: File, name, dims, shape=None, dtype=None, data=None, **kwargs):
        """

        """
        if shape is None:
            if data is None:
                raise ValueError('shape and dtype must be passed or data must be passed.')
            shape = data.shape

        if dtype is None:
            dtype = data.dtype

        ## Check if dims already exist and if the dim lengths match
        for i, dim in enumerate(dims):
            if dim not in file:
                raise ValueError(f'{dim} not in File-Group')

            dim_len = file._file[dim].shape[0]
            if dim_len != shape[i]:
                raise ValueError(f'{dim} does not have the same length as the input data/shape dim.')

        ## Create dataset
        if data is None:
            ds = file._file.create_dataset(name, shape, dtype=dtype, data=data, **kwargs)
        else:
            ds = file._file.create_dataset(name, dtype=dtype, data=data, **kwargs)

        for i, dim in enumerate(dims):
            ds.dims[i].attach_scale(file._file[dim])
            ds.dims[i].label = dim

        self._dataset = ds
        self.dims = dims
        self.file = file


    def to_pandas(self):
        """

        """

    def to_xarray(self):
        """

        """

    def copy(self, name, include_data=False, **kwargs):
        """
        Copy a Dataset object. Same as create_dataset_like.
        """
        other1 = self._dataset
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
            ds = Dataset(self.file, name, tuple(dim.label for dim in other1.dims), data=other1, **kwargs)
        else:
            ds = Dataset(self.file, name, tuple(dim.label for dim in other1.dims), **kwargs)

        return ds


    def __repr__(self):
        """

        """

    def sel(self):
        """

        """



class Dimension(Dataset):
    """

    """
    def __init__(self, file: File, name, shape=None, dtype=None, data=None, **kwargs):
        """

        """
        if data is not None:
            ## Test to make sure it's a single dimension
            if data.ndim > 1:
                raise ValueError('The input array for the dimension must be a 1-D array.')
            if dtype is None:
                dtype = data.dtype
            ds = file._file.create_dataset(name, dtype=dtype, data=data, **kwargs)
        else:
            ds = file._file.create_dataset(name, shape, dtype=dtype, data=data, **kwargs)

        ds.make_scale(name)
        ds.dims[0].label = name

        self._dataset = ds
        self.file = file


class MultiIndex:
    """

    """



d




##############################################
### Testing

dims = ('dim1', 'dim2')

dim1_data = np.array([0, 1, 2], dtype='int16')
dim2_data = np.array([3, 4, 5, 6], dtype='int16')

self = File(name, 'w')

if 'dim1' in self:
    del self['dim1']
if 'dim2' in self:
    del self['dim2']
if 'test1' in self:
    del self['test1']

dim1_test = self.create_dimension('dim1', data=dim1_data)
dim2_test = self.create_dimension('dim2', data=dim2_data)

test1_data = np.arange(12).reshape(3, 4)

test1_ds = self.create_dataset('test1', dims, data=test1_data)

test2_ds = self.create_dataset_like('test2', test1_ds)

self = h5py.File(name, 'w', userblock_size=512)
self = h5py.File(name, 'r')

with open(name, 'rb') as f:
    b1 = f.read()



self.close()

















































