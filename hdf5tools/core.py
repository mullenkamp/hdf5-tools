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

name = '/home/mike/data/temp/test_file.h5'


###################################################
### Classes


class File(h5py.File):
    """

    """
    def __init__(self, name: Union[str, pathlib.Path, io.BytesIO]=None, mode: str='r', driver: str=None, **kwargs):
        """
        The top level object for managing hdf5 data. Is equivalent to the h5py.File object.

        Parameters
        ----------
        name : str, pathlib.Path, io.BytesIO, or None
            A str or pathlib.Path object to a file on disk, a BytesIO object, or None. If None, it will create an in-memory hdf5 File.
        mode : str
            The typical python open mode.
        driver : str or None
            File driver to use; see https://docs.h5py.org/en/stable/high/file.html#file-driver.
        **kwargs
            Any other kwargs that will be passed to the h5py.File object.
        """
        if name is None:
            name = tempfile.NamedTemporaryFile()
            driver = 'core'
            super().__init__(name=name.name, mode='w', driver=driver, backing_store=False, **kwargs)
        else:
            super().__init__(name=name, mode=mode, driver=driver, **kwargs)


    def create_dimension(self, name, shape=None, dtype=None, data=None, **kwargs):
        """

        """
        if data is not None:
            ## Test to make sure it's a single dimension
            if data.ndim > 1:
                raise ValueError('The input array for the dimension must be a 1-D array.')
            if dtype is None:
                dtype = data.dtype
            ds = super().create_dataset(name, dtype=dtype, data=data, **kwargs)
        else:
            ds = super().create_dataset(name, shape, dtype=dtype, data=data, **kwargs)

        ds.make_scale(name)
        ds.dims[0].label = name

        return ds


    def create_dataset(self, name, dims, shape=None, dtype=None, data=None, **kwargs):
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
            if dim not in self:
                raise ValueError(f'{dim} not in File-Group')

            dim_len = self[dim].shape[0]
            if dim_len != shape[i]:
                raise ValueError(f'{dim} does not have the same length as the input data/shape dim.')

        ## Create dataset
        if data is None:
            ds = super().create_dataset(name, shape, dtype=dtype, data=data, **kwargs)
        else:
            ds = super().create_dataset(name, dtype=dtype, data=data, **kwargs)

        for i, dim in enumerate(dims):
            ds.dims[i].attach_scale(self[dim])
            ds.dims[i].label = dim

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
        for k in ('shape', 'dtype', 'chunks', 'compression',
                  'compression_opts', 'scaleoffset', 'shuffle', 'fletcher32',
                  'fillvalue'):
            kwargs.setdefault(k, getattr(other, k))
        # TODO: more elegant way to pass these (dcpl to create_dataset?)
        dcpl = other.id.get_create_plist()
        kwargs.setdefault('track_times', dcpl.get_obj_track_times())
        kwargs.setdefault('track_order', dcpl.get_attr_creation_order() > 0)

        # Special case: the maxshape property always exists, but if we pass it
        # to create_dataset, the new dataset will automatically get chunked
        # layout. So we copy it only if it is different from shape.
        if other.maxshape != other.shape:
            kwargs.setdefault('maxshape', other.maxshape)

        ds = super().create_dataset(name, **kwargs)

        for i, dim in enumerate(other.dims):
            ds.dims[i].attach_scale(self[dim.label])
            ds.dims[i].label = dim.label

        return ds














##############################################
### Testing

dims = ('dim1', 'dim2')

dim1_data = np.array([0, 1, 2], dtype='int16')
dim2_data = np.array([3, 4, 5, 6], dtype='int16')

self = File(name, 'r+')

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




# self.close()

















































