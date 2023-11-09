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
import fcntl
import uuid

# from hdf5tools import utils

import utils

h5py.get_config().track_order = True

sup = np.testing.suppress_warnings()
sup.filter(FutureWarning)

###################################################
### Parameters

name = '/media/data01/cache/tethys/test/test1.h5'
name_indent = 4
value_indent = 20


###################################################
### Classes


class File:
    """

    """
    def __init__(self, name: Union[str, pathlib.Path, io.BytesIO]=None, mode: str='r', compression: str='zstd', write_lock=False, **kwargs):
        """
        The top level object for managing hdf5 data. Is equivalent to the h5py.File object.

        Parameters
        ----------
        name : str, pathlib.Path, io.BytesIO, or None
            A str or pathlib.Path object to a file on disk, a BytesIO object, or None. If None, it will create an in-memory hdf5 File.
        mode : str
            The typical python open mode.
        compression : str or None
            The default compression for all dimensiona dna datasets. These can be changed individually at dataset/dimension creation.
        write_lock : bool
            Lock the file (using fcntl.flock) during write operations. Only use this when using multithreading or multiprocessing and you want to write to the same file. You probably shouldn't perform read operations during the writes.
        **kwargs
            Any other kwargs that will be passed to the h5py.File object.
        """
        writable = True if (mode.lower() in ['r+', 'w', 'a', 'w-', 'x']) else False

        if 'rdcc_nbytes' not in kwargs:
            kwargs['rdcc_nbytes'] = 2**21
        lock_fileno = None
        if name is None:
            name = uuid.uuid4().hex[:16]
            kwargs.setdefault('driver', 'core')
            if 'backing_store' not in kwargs:
                kwargs.setdefault('backing_store', False)
            file = h5py.File(name=name, track_order=True, mode='w', **kwargs)
            writable = True
        else:
            if write_lock and writable:
                lock_fileno = os.open(name, os.O_RDONLY)
                fcntl.flock(lock_fileno, fcntl.LOCK_EX)

                file = h5py.File(name=name, mode=mode, track_order=True, locking=False, **kwargs)
            else:
                file = h5py.File(name=name, mode=mode, track_order=True, **kwargs)

        self._file = file
        self.mode = mode
        self.writable = writable
        self.filename = file.filename
        self.compression = compression
        self.lock_fileno = lock_fileno
        self.driver = file.driver

        for ds_name in file:
            ds = file[ds_name]
            if utils.is_scale(ds):
                Dimension(ds, self)
            else:
                Dataset(ds, self)

        self.attrs = Attributes(file.attrs)


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
        return self._file.__bool__()

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
            raise TypeError('Assigned value must be a Dataset or Dimension object.')

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
        # self._file.__exit__()
        self.close()

    def close(self):
        self._file.close()
        if self.lock_fileno is not None:
            fcntl.flock(self.lock_fileno, fcntl.LOCK_UN)
            os.close(self.lock_fileno)

    def flush(self):
        """

        """
        self._file.flush()


    def __repr__(self):
        """

        """
        return file_summary(self)

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

    def copy(self, name: Union[str, pathlib.Path, io.BytesIO]=None, compression: str='zstd', **kwargs):
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


    def create_dimension(self, name, data, scale_factor=None, add_offset=None, missing_value=None, units=None, calendar=None, dtype_decoded=None, encoding=None, **kwargs):
        """

        """
        if 'compression' not in kwargs:
            compression = self.compression
            compressor = utils.get_compressor(compression)
            kwargs.update({**compressor})
        else:
            compression = kwargs['compression']

        data = np.asarray(data)

        dtype, shape = get_dtype_shape(data, dtype=None, shape=None)

        encoding = prepare_encodings_for_datasets(dtype, scale_factor, add_offset, missing_value, units, calendar, dtype_decoded, encoding)

        dimension = create_h5py_dimension(self, name, data, dtype, shape, encoding, **kwargs)
        dim = Dimension(dimension, self)
        dim.encoding.update(encoding)
        dim.encoding['compression'] = str(compression)

        return dim


    def create_dataset(self, name, dims, shape=None, dtype=None, data=None, scale_factor=None, add_offset=None, missing_value=None, units=None, calendar=None, dtype_decoded=None, encoding=None, **kwargs):
        """

        """
        if 'compression' not in kwargs:
            compression = self.compression
            compressor = utils.get_compressor(compression)
            kwargs.update({**compressor})
        else:
            compression = kwargs['compression']

        if data is not None:
            data = np.asarray(data)

        dtype, shape = get_dtype_shape(data, dtype=None, shape=None)

        encoding = prepare_encodings_for_datasets(dtype, scale_factor, add_offset, missing_value, units, calendar, dtype_decoded, encoding)

        ds0 = create_h5py_dataset(self, name, dims, shape, dtype, data, encoding, **kwargs)
        ds = Dataset(ds0, self)
        ds.encoding.update(encoding)
        ds.encoding['compression'] = str(compression)

        return ds


    def create_dataset_like(self, name, other, include_data=False, include_attrs=False, **kwargs):
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

        if 'compression' in other1.attrs:
            compression = other1.attrs['compression']
            kwargs.update(**utils.get_compressor(compression))
        else:
            compression = kwargs['compression']

        # TODO: more elegant way to pass these (dcpl to create_dataset?)
        dcpl = other1.id.get_create_plist()
        kwargs.setdefault('track_times', dcpl.get_obj_track_times())
        # kwargs.setdefault('track_order', dcpl.get_attr_creation_order() > 0)

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
                filter_mask, data = other1.id.read_direct_chunk(chunk_starts)
                ds0.id.write_direct_chunk(chunk_starts, data, filter_mask)

        else:
            ds0 = create_h5py_dataset(self, name, tuple(dim.label for dim in other1.dims), **kwargs)

        ds = Dataset(ds0, self)
        ds.encoding.update(other.encoding._encoding)
        if include_attrs:
            ds.attrs.update(other.attrs)

        return ds


def get_dtype_shape(data=None, dtype=None, shape=None):
    """

    """
    if data is None:
        if (shape is None) or (dtype is None):
            raise ValueError('shape and dtype must be passed or data must be passed.')
        if not isinstance(dtype, str):
            dtype = dtype.name
    else:
        shape = data.shape
        dtype = data.dtype.name

    return dtype, shape


def prepare_encodings_for_datasets(dtype, scale_factor, add_offset, missing_value, units, calendar, dtype_decoded, encoding):
    """

    """
    if encoding is None:
        encoding = {'dtype': dtype, 'missing_value': missing_value, 'add_offset': add_offset, 'scale_factor': scale_factor, 'units': units, 'calendar': calendar}
        for key, value in copy.deepcopy(encoding).items():
            if value is None:
                del encoding[key]
    else:
        for key in encoding.keys():
            if key not in utils.enc_fields:
                raise ValueError(f'{key} is not a valid encoding parameter. They must be one or more of {utils.enc_fields}.')

    return encoding


def create_h5py_dataset(file: File, name, dims, shape=None, dtype=None, data=None,  encoding=None, **kwargs):
    """

    """
    if data is None:
        if (shape is None) or (dtype is None):
            raise ValueError('shape and dtype must be passed or data must be passed.')
        if not isinstance(dtype, str):
            dtype = dtype.name
    else:
        shape = data.shape
        dtype = data.dtype.name

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
        ds = file._file.create_dataset(name, shape, dtype=dtype, track_order=True, **kwargs)
    else:
        ## Encode data before creating dataset
        data = utils.encode_data(data, **encoding)

        ds = file._file.create_dataset(name, dtype=dtype, data=data, track_order=True, **kwargs)

    for i, dim in enumerate(dims):
        ds.dims[i].attach_scale(file._file[dim])
        ds.dims[i].label = dim

    return ds


def create_h5py_dimension(file: File, name, data, dtype, shape, encoding, **kwargs):
    """

    """
    if len(shape) != 1:
        raise ValueError('The shape of a dimension must be 1-D.')

    ## Make chunks
    if 'chunks' not in kwargs:
        if 'maxshape' in kwargs:
            maxshape = kwargs['maxshape']
        else:
            maxshape = shape
        kwargs.setdefault('chunks', utils.guess_chunk(shape, maxshape, dtype))

    ## Encode data before creating dataset/dimension
    data = utils.encode_data(data, **encoding)

    ## Make Dataset
    ds = file._file.create_dataset(name, dtype=dtype, data=data, track_order=True, **kwargs)

    ds.make_scale(name)
    ds.dims[0].label = name

    return ds


def format_value(value):
    """

    """
    if isinstance(value, (int, np.integer)):
        return str(value)
    elif isinstance(value, (float, np.floating)):
        return f'{value:.2f}'
    else:
        return value


def append_summary(summary, summ_dict):
    """

    """
    for key, value in summ_dict.items():
        spacing = value_indent - len(key)
        if spacing < 1:
            spacing = 1

        summary += f"""\n{key}""" + """ """ * spacing + value

    return summary


def dataset_summary(ds):
    """

    """
    if ds:
        # dims_shapes = [str(ds.dim_names[i]) + ': ' + str(s) for i, s in enumerate(ds.shape)]
        # dtype_name = ds.dtype.name
        # dtype_decoded = ds.encoding['dtype_decoded']

        # summ_dict = {'name': ds.name, 'dtype encoded': dtype_name, 'dtype decoded': dtype_decoded, 'dims order': '(' + ', '.join(ds.dim_names) + ')', 'chunk size': str(ds.chunks)}
        summ_dict = {'name': ds.name, 'dims order': '(' + ', '.join(ds.dim_names) + ')', 'chunk size': str(ds.chunks)}

        summary = """<hdf5tools.Dataset>"""

        summary = append_summary(summary, summ_dict)

        summary += """\nDimensions:"""

        for dim_name in ds.dim_names:
            dim = ds.file[dim_name]
            dtype_name = dim.encoding['dtype_decoded']
            dim_len = dim.shape[0]
            first_value = format_value(dim[0])
            spacing = value_indent - name_indent - len(dim_name)
            if spacing < 1:
                spacing = 1
            dim_str = f"""\n    {dim_name}""" + """ """ * spacing
            dim_str += f"""({dim_len}) {dtype_name} {first_value} ..."""
            summary += dim_str

        attrs_summary = make_attrs_repr(ds.attrs, name_indent, value_indent, 'Attributes')
        summary += """\n""" + attrs_summary

    else:
        summary = """Dataset is closed"""

    return summary


def dimension_summary(ds):
    """

    """
    if ds:
        name = ds.name
        dim_len = ds.shape[0]
        # dtype_name = ds.dtype.name
        # dtype_decoded = ds.encoding['dtype_decoded']

        first_value = format_value(ds.data[0])
        last_value = format_value(ds.data[-1])

        # summ_dict = {'name': name, 'dtype encoded': dtype_name, 'dtype decoded': dtype_decoded, 'chunk size': str(ds.chunks), 'dim length': str(dim_len), 'values': f"""{first_value} ... {last_value}"""}
        summ_dict = {'name': name, 'chunk size': str(ds.chunks), 'dim length': str(dim_len), 'values': f"""{first_value} ... {last_value}"""}

        summary = """<hdf5tools.Dataset>"""

        summary = append_summary(summary, summ_dict)

        attrs_summary = make_attrs_repr(ds.attrs, name_indent, value_indent, 'Attributes')
        summary += """\n""" + attrs_summary
    else:
        summary = """Dimension is closed"""

    return summary


def file_summary(file):
    """

    """
    if file:
        file_path = pathlib.Path(file.filename)
        if file_path.exists() and file_path.is_file():
            file_size = file_path.stat().st_size*0.000001
            file_size_str = """{file_size:.3f} MB""".format(file_size=file_size)
        else:
            file_size_str = """NA"""

        summ_dict = {'file name': file_path.name, 'file size': file_size_str, 'writable': str(file.writable)}

        summary = """<hdf5tools.File>"""

        summary = append_summary(summary, summ_dict)

        summary += """\nDimensions:"""

        for dim_name in file.dimensions():
            dim = file[dim_name]
            dtype_name = dim.encoding['dtype_decoded']
            dim_len = dim.shape[0]
            first_value = format_value(dim[0])
            spacing = value_indent - name_indent - len(dim_name)
            if spacing < 1:
                spacing = 1
            dim_str = f"""\n    {dim_name}""" + """ """ * spacing
            dim_str += f"""({dim_len}) {dtype_name} {first_value} ..."""
            summary += dim_str

        summary += """\nDatasets:"""

        for ds_name in file.datasets():
            ds = file[ds_name]
            dtype_name = ds.encoding['dtype_decoded']
            shape = ds.shape
            dim_names = ', '.join(ds.dim_names)
            first_value = format_value(ds[tuple(0 for i in range(len(shape)))])
            spacing = value_indent - name_indent - len(ds_name)
            if spacing < 1:
                spacing = 1
            ds_str = f"""\n    {ds_name}""" + """ """ * spacing
            ds_str += f"""({dim_names}) {dtype_name} {first_value} ..."""
            summary += ds_str

        attrs_summary = make_attrs_repr(file.attrs, name_indent, value_indent, 'Attributes')
        summary += """\n""" + attrs_summary
    else:
        summary = """File is closed"""

    return summary



class Dataset:
    """

    """
    def __init__(self, dataset: h5py.Dataset, file: File):
        """

        """
        self._dataset = dataset
        self.dim_names = tuple(dim.label for dim in dataset.dims)
        self.ndim = dataset.ndim
        self.shape = dataset.shape
        self.size = dataset.size
        self.dtype = dataset.dtype
        self.nbytes = dataset.nbytes
        self.chunks = dataset.chunks
        self.fillvalue = dataset.fillvalue
        self.name = dataset.name.split('/')[-1]
        self.file = file
        setattr(file, self.name, self)
        self.attrs = Attributes(dataset.attrs)
        self.encoding = Encoding(dataset.attrs, dataset.dtype, file.writable)
        self.loc = LocationIndexer(self)


    def __getitem__(self, key):
        return utils.decode_data(self._dataset[key], **self.encoding._encoding)


    def iter_chunks(self, sel=None):
        return self._dataset.iter_chunks(sel)

    def __bool__(self):
        return self._dataset.__bool__()

    def len(self):
        return self._dataset.len()

    def to_pandas(self):
        """

        """

    def to_xarray(self):
        """

        """

    def copy(self, name, include_data=True, include_attrs=True, **kwargs):
        """
        Copy a Dataset object. Same as create_dataset_like.
        """
        ds = self.file.create_dataset_like(name, self, include_data=include_data, include_attrs=include_attrs, **kwargs)

        return ds


    def __repr__(self):
        """

        """
        return dataset_summary(self)

    def sel(self):
        """

        """



class Dimension:
    """

    """
    def __init__(self, dataset: h5py.Dataset, file: File):
        """

        """
        self._dataset = dataset
        self.dim_names = tuple(dim.label for dim in dataset.dims)
        self.ndim = dataset.ndim
        self.shape = dataset.shape
        self.size = dataset.size
        self.dtype = dataset.dtype
        self.nbytes = dataset.nbytes
        self.chunks = dataset.chunks
        self.fillvalue = dataset.fillvalue
        self.name = dataset.name.split('/')[-1]
        self.file = file
        setattr(file, self.name, self)
        self.attrs = Attributes(dataset.attrs)
        self.encoding = Encoding(dataset.attrs, dataset.dtype, file.writable)
        self.loc = LocationIndexer(self)
        self.data = self[()]


    def copy(self, name, include_data=True, include_attrs=True, **kwargs):
        """
        Copy a Dataset object. Same as create_dimension_like.
        """
        ds = self.file.create_dimension_like(name, self, include_data=include_data, include_attrs=include_attrs, **kwargs)

        return ds

    def __getitem__(self, key):
        return utils.decode_data(self._dataset[key], **self.encoding._encoding)

    def iter_chunks(self, sel=None):
        return self._dataset.iter_chunks(sel)

    def __bool__(self):
        return self._dataset.__bool__()

    def len(self):
        return self._dataset.len()

    def __repr__(self):
        """

        """
        return dimension_summary(self)

    def sel(self):
        """

        """

    def to_pandas(self):
        """

        """

    def to_xarray(self):
        """

        """


class Attributes:
    """

    """
    def __init__(self, attrs: h5py.AttributeManager):
        self._attrs = attrs

    def get(self, key, default=None):
        return self._attrs.get(key, default)

    def __getitem__(self, key):
        return self._attrs[key]

    def __setitem__(self, key, value):
        self._attrs[key] = value

    def clear(self):
        self._attrs.clear()

    def keys(self):
        for key in self._attrs.keys():
            if key not in utils.ignore_attrs:
                yield key

    def values(self):
        for key, value in self._attrs.items():
            if key not in utils.ignore_attrs:
                yield value

    def items(self):
        for key, value in self._attrs.items():
            if key not in utils.ignore_attrs:
                yield key, value

    def pop(self, key, default=None):
        return self._attrs.pop(key, default)

    def update(self, other=()):
        self._attrs.update(other)

    def create(self, key, data, shape=None, dtype=None):
        self._attrs.create(key, data, shape, dtype)

    def modify(self, key, value):
        self._attrs.modify(key, value)

    def __delitem__(self, key):
        del self._attrs[key]

    def __contains__(self, key):
        return key in self._attrs

    def __iter__(self):
        return self._attrs.__iter__()

    def __repr__(self):
        return make_attrs_repr(self, name_indent, value_indent, 'Attributes')


class Encoding:
    """

    """
    def __init__(self, attrs: h5py.AttributeManager, dtype, writable):
        enc = utils.get_encoding_data_from_h5py_attrs(attrs)
        enc = utils.process_encoding(enc, dtype)
        enc = utils.assign_dtype_decoded(enc)
        self._encoding = enc
        if writable:
            attrs.update(enc)
        self._attrs = attrs
        self._writable = writable

    def get(self, key, default=None):
        return self._encoding.get(key, default)

    def __getitem__(self, key):
        return self._encoding[key]

    def __setitem__(self, key, value):
        if key in utils.enc_fields:
            self._encoding[key] = value
            if self._writable:
                self._attrs[key] = value
        else:
            raise ValueError(f'key must be one of {utils.enc_fields}.')

    def clear(self):
        keys = list(self._encoding.keys())
        self._encoding.clear()
        if self._writable:
            for key in keys:
                del self._attrs[key]

    def keys(self):
        return self._encoding.keys()

    def values(self):
        return self._encoding.values()

    def items(self):
        return self._encoding.items()

    def pop(self, key, default=None):
        if self._writable:
            if key in self._attrs:
                del self._attrs[key]
        return self._encoding.pop(key, default)

    def update(self, other=()):
        key_values = {**other}
        for key, value in key_values.items():
            if key in utils.enc_fields:
                self._encoding[key] = value
                if self._writable:
                    self._attrs[key] = value

    def __delitem__(self, key):
        del self._encoding[key]
        if self._writable:
            del self._attrs[key]

    def __contains__(self, key):
        return key in self._encoding

    def __iter__(self):
        return self._encoding.__iter__()

    def __repr__(self):
        return make_attrs_repr(self, name_indent, value_indent, 'Encodings')


class EncodeDecode:
    """

    """
    def __init__(self, encoding):
        """

        """
        self.encoding

    def encode(self, values):
        return utils.encode_data(values, **self.encoding._encoding)

    def decode(self, values):
        return utils.decode_data(values, **self.encoding._encoding)



def make_attrs_repr(attrs, name_indent, value_indent, header):
    summary = f"""{header}:"""
    for key, value in attrs.items():
        spacing = value_indent - name_indent - len(key)
        if spacing < 1:
            spacing = 1
        line_str = f"""\n    {key}""" + """ """ * spacing + f"""{value}"""
        summary += line_str

    return summary



class LocationIndexer:
    """

    """
    def __init__(self, dataset: Dataset):
        """

        """
        self.dataset = dataset


    def __getitem__(self, key):
        """

        """
        if isinstance(key, (int, float, str, slice, list, np.ndarray)):
            index = index_all(key, self.dataset, 0)

            return self.dataset[index]

        elif isinstance(key, tuple):
            key_len = len(key)

            if key_len == 0:
                return self.dataset[()]

            elif key_len > self.dataset.ndim:
                raise ValueError('input must have <= ndims.')

            index = []
            for i, k in enumerate(key):
                index_i = index_all(k, self.dataset, i)
                index.append(index_i)

            return self.dataset[tuple(index)]

        else:
            raise ValueError('You passed a strange object to index...')


def index_slice(slice_obj, dim_data):
    """

    """
    start = slice_obj.start
    stop = slice_obj.stop

    ## If the np.nonzero finds nothing, then it fails
    if start is None:
        start_idx = None
    else:
        try:
            start_idx = np.nonzero(dim_data == start)[0][0]
        except IndexError:
            try:
                start_time = np.datetime64(start)
                start_idx = np.nonzero(dim_data == start_time)[0][0]
            except IndexError:
                raise ValueError(f'{start} not in dimension.')

    ## stop_idx should include the stop label as per pandas
    if stop is None:
        stop_idx = None
    else:
        try:
            stop_idx = np.nonzero(dim_data == stop)[0][0] + 1
        except IndexError:
            try:
                stop_time = np.datetime64(stop)
                stop_idx = np.nonzero(dim_data == stop_time)[0][0] + 1
            except IndexError:
                raise ValueError(f'{stop} not in dimension.')

    if (stop_idx is not None) and (start_idx is not None):
        if start_idx > stop_idx:
            raise ValueError(f'start index at {start_idx} is after stop index at {stop_idx}.')

    return slice(start_idx, stop_idx)


def index_label(label, dim_data):
    """

    """
    try:
        label_idx = np.nonzero(dim_data == label)[0][0]
    except IndexError:
        try:
            label_time = np.datetime64(label)
            label_idx = np.nonzero(dim_data == label_time)[0][0]
        except IndexError:
            raise ValueError(f'{label} not in dimension.')

    return label_idx


def index_array(values, dim_data):
    """

    """
    values = np.asarray(values)

    val_len = len(values)
    if val_len == 0:
        raise ValueError('The array is empty...')
    elif val_len == 1:
        index = index_label(values[0], dim_data)

    ## check if regular
    elif utils.is_regular_index(values):
        index = index_slice(slice(values[0], values[-1]), dim_data)

    # TODO I might need to do something more fancy here...
    else:
        index = values

    return index


@sup
def index_all(key, dataset, pos):
    """

    """
    if isinstance(key, (int, float, str)):
        dim_data = dataset.file[dataset.dim_names[pos]].data
        label_idx = index_label(key, dim_data)

        return label_idx

    elif isinstance(key, slice):
        dim_data = dataset.file[dataset.dim_names[pos]].data
        slice_idx = index_slice(key, dim_data)

        return slice_idx

    elif key is None:
         return slice(None, None)

    elif isinstance(key, (list, np.ndarray)):
        key = np.asarray(key)

        dim_data = dataset.file[dataset.dim_names[pos]].data

        if key.dtype.name == 'bool':
            if len(key) != len(dim_data):
                raise ValueError('If the input is a bool array, then it must be the same length as the dimension.')

            return key
        else:
            idx = index_array(key, dim_data)

            return idx












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

self = File(name, 'w', compression='zstd')

# if 'dim1' in self:
#     del self['dim1']
# if 'dim2' in self:
#     del self['dim2']
# if 'test1' in self:
#     del self['test1']

dim1_test = self.create_dimension('dim1', data=dim1_data)
dim2_test = self.create_dimension('dim2', data=dim2_data)

test1_data = np.arange(dim1_len*dim2_len, dtype='int32').reshape(dim1_len, dim2_len)

test1_ds = self.create_dataset('test1', dims, data=test1_data)

test2_ds = self.create_dataset_like('test2', test1_ds, include_data=True)

self = h5py.File(name, 'w', userblock_size=512)
self = h5py.File(name, 'r+')

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




self = File(nc2)

file = self.copy(nc3)

file2 = self.copy()



self = File(name, 'r+', write_lock=True)
f1 = File(name)
f2 = File(nc2)

slice_obj = slice(110, 130)

f1 = open('/media/data01/cache/tethys/test/test_lock.lock', 'wb')

fcntl.flock(f1.fileno(), fcntl.LOCK_EX)
fcntl.flock(f1.fileno(), fcntl.LOCK_UN)

f2 = open('/media/data01/cache/tethys/test/test_lock.lock', 'rb')

fcntl.flock(f2.fileno(), fcntl.LOCK_EX)





