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

try:
    import pandas as pd
    import_pandas = True
except ImportError:
    import_pandas = False

try:
    import xarray as xr
    import_xarray = True
except ImportError:
    import_xarray = False

# from hdf5tools import utils

import utils

h5py.get_config().track_order = True

sup = np.testing.suppress_warnings()
sup.filter(FutureWarning)

###################################################
### Parameters

name_indent = 4
value_indent = 20


###################################################
### Classes


class File:
    """

    """
    def __init__(self, name: Union[str, pathlib.Path, io.BytesIO]=None, mode: str='r', compression: str='zstd', write_lock=False, **kwargs):
        """
        The top level file object for managing cf conventions data.

        Parameters
        ----------
        name : str, pathlib.Path, io.BytesIO, or None
            A str or pathlib.Path object to a file on disk, a BytesIO object, or None. If None, it will create an in-memory hdf5 File.
        mode : str
            The typical python open mode.
        compression : str or None
            The default compression for all coordinatea and variables. These can be changed individually at variable/coordinate creation.
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
                Coordinate(ds, self)
            else:
                Variable(ds, self)

        self.attrs = Attributes(file.attrs)


    def variables(self):
        for name in self:
            if isinstance(self[name], Variable):
                yield name

    def coords(self):
        for name in self:
            if isinstance(self[name], Coordinate):
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
        if isinstance(key, str):
            if key in self._file:
                return getattr(self, key)
            else:
                raise KeyError(key)
        else:
            raise TypeError('key must be a string.')

    def __setitem__(self, key, value):
        if isinstance(value, (Variable, Coordinate)):
            setattr(self, key, value)
        else:
            raise TypeError('Assigned value must be a Variable or Coordinate object.')

    def __delitem__(self, key):
        try:
            if key not in self:
                raise KeyError(key)

            # Check if the object to delete is a coordinate
            # And if it is, check that no variables are attached to it
            if isinstance(self[key], Coordinate):
                for ds_name in self.variables:
                    if key in self[ds_name].dims:
                        raise ValueError(f'{key} is a coordinate of {ds_name}. You must delete all variables associated with a coordinate before you can delete the coordinate.')

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

    def sel(self, selection: dict=None, include_dims: list=None, exclude_dims: list=None, include_variables: list=None, exclude_variables: list=None, **file_kwargs):
        """

        """
        ## Check for coordinate names in input
        dims = np.array(list(self.coords()))

        if selection is not None:
            keys = tuple(selection.keys())
            for key in keys:
                if key not in dims:
                    raise KeyError(f'{key} is not in the coordinates.')

        if include_dims is not None:
            include_dims_check = np.isin(include_dims, dims)
            if not include_dims_check.all():
                no_dims = ', '.join(include_dims[np.where(include_dims_check)[0].tolist()])
                raise KeyError(f'{no_dims} are not in dims.')

        if exclude_dims is not None:
            exclude_dims_check = np.isin(exclude_dims, dims)
            if not exclude_dims_check.all():
                no_dims = ', '.join(exclude_dims[np.where(exclude_dims_check)[0].tolist()])
                raise KeyError(f'{no_dims} are not in dims.')

        ## Check if variables exist
        variables = np.array(list(self.variables()))

        if include_variables is not None:
            include_variables_check = np.isin(include_variables, variables)
            if not include_variables_check.all():
                no_variables = ', '.join(include_variables[np.where(include_variables_check)[0].tolist()])
                raise KeyError(f'{no_variables} are not in variables.')

        if exclude_variables is not None:
            exclude_variables_check = np.isin(exclude_variables, variables)
            if not exclude_variables_check.all():
                no_variables = ', '.join(exclude_variables[np.where(exclude_variables_check)[0].tolist()])
                raise KeyError(f'{no_variables} are not in variables.')

        ## Filter dims
        if include_dims is not None:
            dims = dims[np.isin(dims, include_dims)]
        if exclude_dims is not None:
            dims = dims[~np.isin(dims, exclude_dims)]

        ## Filter variables
        if include_variables is not None:
            variables = variables[np.isin(variables, include_variables)]
        if exclude_variables is not None:
            variables = variables[~np.isin(variables, exclude_variables)]

        for ds_name in copy.deepcopy(variables):
            ds = self[ds_name]
            ds_dims = np.array(ds.dims)
            dims_check = np.isin(ds_dims, dims).all()
            if not dims_check:
                variables.remove(ds_name)

        ## Create file
        file_kwargs['mode'] = 'w'
        new_file = File(**file_kwargs)

        ## Iterate through the coordinates
        for dim_name in dims:
            old_dim = self[dim_name]

            if selection is not None:
                if dim_name in selection:
                    data = old_dim.loc[selection[dim_name]]
                else:
                    data = old_dim.data
            else:
                data = old_dim.data

            new_dim = new_file.create_coordinate(dim_name, data, encoding=old_dim.encoding._encoding)
            new_dim.attrs.update(old_dim.attrs)

        ## Iterate through the old variables
        # TODO: Make the variable copy when doing a selection more RAM efficient
        for ds_name in variables:
            old_ds = self[ds_name]

            if selection is not None:
                ds_dims = old_ds.dims

                ds_sel = []
                for dim in ds_dims:
                    if dim in keys:
                        ds_sel.append(selection[key])
                    else:
                        ds_sel.append(None)

                data = old_ds.loc[tuple(ds_sel)]
                new_ds = new_file.create_variable(ds_name, old_ds.dims, data=data, encoding=old_ds.encoding._encoding)
                new_ds.attrs.update(old_ds.attrs)
            else:
                new_ds = old_ds.copy(new_file)

        ## Add global attrs
        new_file.attrs.update(self.attrs)

        return new_file


    def to_pandas(self):
        """

        """
        if not import_pandas:
            raise ImportError('pandas could not be imported.')

        # TODO: This feels wrong...
        result = None
        for var_name in self.variables():
            if result is None:
                result = self[var_name].to_pandas().to_frame()
            else:
                result = result.join(self[var_name].to_pandas().to_frame(), how='outer')

        return result


    def to_xarray(self, **kwargs):
        """
        Closes the file and opens it in xarray.
        """
        if not import_xarray:
            raise ImportError('xarray could not be imported.')

        filename = pathlib.Path(self.filename)

        if filename.is_file():
            self.close()
        else:
            temp_file = tempfile.NamedTemporaryFile()
            filename = temp_file.name
            self.to_file(filename)
            self.close()

        x1 = xr.open_dataset(filename, engine='h5netcdf', **kwargs)

        return x1


    def to_file(self, name: Union[str, pathlib.Path, io.BytesIO], compression: str='zstd', **file_kwargs):
        """
        Like copy, but must be a file path and will not be returned.
        """
        file = self.copy(name, compression, **file_kwargs)
        file.close()


    def copy(self, name: Union[str, pathlib.Path, io.BytesIO]=None, compression: str='zstd', **file_kwargs):
        """
        Copy a file object. kwargs can be any parameter for File.
        """
        # kwargs.setdefault('mode', 'w')
        file = File(name, mode='w', compression=compression, **file_kwargs)

        ## Create coordinates
        for dim_name in self.coords():
            dim = self[dim_name]
            _ = copy_coordinate(file, dim, dim_name)

        ## Create variables
        for ds_name in self.variables():
            ds = self[ds_name]
            _ = copy_variable(file, ds, ds_name)

        return file


    def create_coordinate(self, name, data, scale_factor=None, add_offset=None, missing_value=None, units=None, calendar=None, dtype_decoded=None, encoding=None, **kwargs):
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

        encoding = prepare_encodings_for_variables(dtype, scale_factor, add_offset, missing_value, units, calendar, dtype_decoded, encoding)

        coordinate = create_h5py_coordinate(self, name, data, shape, encoding, **kwargs)
        dim = Coordinate(coordinate, self, encoding)
        dim.encoding['compression'] = str(compression)

        return dim


    def create_variable(self, name: str, dims: (str, tuple, list), shape: (tuple, list)=None, dtype: np.dtype=None, data=None, scale_factor=None, add_offset=None, missing_value=None, units=None, calendar=None, dtype_decoded=None, encoding=None, **kwargs):
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

        dtype, shape = get_dtype_shape(data, dtype, shape)

        encoding = prepare_encodings_for_variables(dtype, scale_factor, add_offset, missing_value, units, calendar, dtype_decoded, encoding)

        ds0 = create_h5py_variable(self, name, dims, shape, encoding, data, **kwargs)
        ds = Variable(ds0, self, encoding)
        ds.encoding['compression'] = str(compression)

        return ds


    def create_variable_like(self, from_variable, name, include_data=False, include_attrs=False, **kwargs):
        """ Create a variable similar to `other`.

        name
            Name of the variable (absolute or relative).  Provide None to make
            an anonymous variable.
        from_variable
            The variable which the new variable should mimic. All properties, such
            as shape, dtype, chunking, ... will be taken from it, but no data
            or attributes are being copied.

        Any variable keywords (see create_variable) may be provided, including
        shape and dtype, in which case the provided values take precedence over
        those from `other`.
        """
        ds = copy_variable(self, from_variable, name, include_data, include_attrs, **kwargs)

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


def prepare_encodings_for_variables(dtype, scale_factor, add_offset, missing_value, units, calendar, dtype_decoded, encoding):
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

    if 'datetime64' in dtype:
        if 'units' not in encoding:
            encoding['units'] = 'seconds since 1970-01-01'
        if 'calendar' not in encoding:
            encoding['calendar'] = 'gregorian'
        encoding['dtype'] = 'int64'

    return encoding


def create_h5py_variable(file: File, name: str, dims: (str, tuple, list), shape: (tuple, list), encoding: dict, data=None, **kwargs):
    """

    """
    dtype = encoding['dtype']

    ## Check if dims already exist and if the dim lengths match
    if isinstance(dims, str):
        dims = [dims]

    for i, dim in enumerate(dims):
        if dim not in file:
            raise ValueError(f'{dim} not in File')

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

    ## Create variable
    if data is None:
        ds = file._file.create_dataset(name, shape, dtype=dtype, track_order=True, **kwargs)
    else:
        ## Encode data before creating variable
        data = utils.encode_data(data, **encoding)

        ds = file._file.create_dataset(name, dtype=dtype, data=data, track_order=True, **kwargs)

    for i, dim in enumerate(dims):
        ds.dims[i].attach_scale(file._file[dim])
        ds.dims[i].label = dim

    return ds


def create_h5py_coordinate(file: File, name: str, data, shape: (tuple, list), encoding: dict, **kwargs):
    """

    """
    if len(shape) != 1:
        raise ValueError('The shape of a coordinate must be 1-D.')

    dtype = encoding['dtype']

    ## Make chunks
    if 'chunks' not in kwargs:
        if 'maxshape' in kwargs:
            maxshape = kwargs['maxshape']
        else:
            maxshape = shape
        kwargs.setdefault('chunks', utils.guess_chunk(shape, maxshape, dtype))

    ## Encode data before creating variable/coordinate
    # print(encoding)
    data = utils.encode_data(data, **encoding)

    # print(data)
    # print(dtype)

    ## Make Variable
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


def variable_summary(ds):
    """

    """
    if ds:
        # dims_shapes = [str(ds.dims[i]) + ': ' + str(s) for i, s in enumerate(ds.shape)]
        # dtype_name = ds.dtype.name
        # dtype_decoded = ds.encoding['dtype_decoded']

        # summ_dict = {'name': ds.name, 'dtype encoded': dtype_name, 'dtype decoded': dtype_decoded, 'dims order': '(' + ', '.join(ds.dims) + ')', 'chunk size': str(ds.chunks)}
        summ_dict = {'name': ds.name, 'dims order': '(' + ', '.join(ds.dims) + ')', 'chunk size': str(ds.chunks)}

        summary = """<hdf5tools.Variable>"""

        summary = append_summary(summary, summ_dict)

        summary += """\nCoordinates:"""

        for dim_name in ds.dims:
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
        summary = """Variable is closed"""

    return summary


def coordinate_summary(ds):
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

        summary = """<hdf5tools.Coordinate>"""

        summary = append_summary(summary, summ_dict)

        attrs_summary = make_attrs_repr(ds.attrs, name_indent, value_indent, 'Attributes')
        summary += """\n""" + attrs_summary
    else:
        summary = """Coordinate is closed"""

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

        summary += """\nCoordinates:"""

        for dim_name in file.coords():
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

        summary += """\nVariables:"""

        for ds_name in file.variables():
            ds = file[ds_name]
            dtype_name = ds.encoding['dtype_decoded']
            shape = ds.shape
            dims = ', '.join(ds.dims)
            first_value = format_value(ds[tuple(0 for i in range(len(shape)))])
            spacing = value_indent - name_indent - len(ds_name)
            if spacing < 1:
                spacing = 1
            ds_str = f"""\n    {ds_name}""" + """ """ * spacing
            ds_str += f"""({dims}) {dtype_name} {first_value} ..."""
            summary += ds_str

        attrs_summary = make_attrs_repr(file.attrs, name_indent, value_indent, 'Attributes')
        summary += """\n""" + attrs_summary
    else:
        summary = """File is closed"""

    return summary



class Variable:
    """

    """
    def __init__(self, dataset: h5py.Dataset, file: File, encoding: dict=None):
        """

        """
        self._dataset = dataset
        self.dims = tuple(dim.label for dim in dataset.dims)
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
        self.encoding = Encoding(dataset.attrs, dataset.dtype, file.writable, encoding)
        self.loc = LocationIndexer(self)


    def __getitem__(self, key):
        return self.encoding.decode(self._dataset[key])

    def __setitem__(self, key, value):
        self._dataset[key] = self.encoding.encode(value)

    def iter_chunks(self, sel=None):
        return self._dataset.iter_chunks(sel)

    def __bool__(self):
        return self._dataset.__bool__()

    def len(self):
        return self._dataset.len()

    def to_pandas(self):
        """

        """
        if not import_pandas:
            raise ImportError('pandas could not be imported.')

        indexes = []
        for dim in self.dims:
            coord = self.file[dim]
            indexes.append(coord.data)

        pd_index = pd.MultiIndex.from_product(indexes, names=self.dims)

        series = pd.Series(self[()].flatten(), index=pd_index)
        series.name = self.name

        return series


    def to_xarray(self, **kwargs):
        """

        """
        if not import_xarray:
            raise ImportError('xarray could not be imported.')

        da = xr.DataArray(data=self[()], coords=[self.file[dim].data for dim in self.dims], dims=self.dims, name=self.name, attrs=self.attrs)

        return da


    def copy(self, to_file: File=None, name: str=None, include_data=True, include_attrs=True, **kwargs):
        """
        Copy a Variable object. Same as create_variable_like.
        """
        if (to_file is None) and (name is None):
            raise ValueError('If to_file is None, then a name must be passed.')

        if to_file is None:
            to_file = self.file

        if name is None:
            name = self.name

        ds = copy_variable(to_file, self, name, include_data=include_data, include_attrs=include_attrs, **kwargs)

        return ds


    def __repr__(self):
        """

        """
        return variable_summary(self)


class Coordinate:
    """

    """
    def __init__(self, dataset: h5py.Dataset, file: File, encoding: dict=None):
        """

        """
        self._dataset = dataset
        self.dims = tuple(dim.label for dim in dataset.dims)
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
        self.encoding = Encoding(dataset.attrs, dataset.dtype, file.writable, encoding)
        self.loc = LocationIndexer(self)
        self.data = self[()]


    def copy(self, to_file: File=None, name: str=None, include_attrs=True, **kwargs):
        """
        Copy a Coordinate object.
        """
        if (to_file is None) and (name is None):
            raise ValueError('If to_file is None, then a name must be passed.')

        if to_file is None:
            to_file = self.file

        if name is None:
            name = self.name

        ds = copy_coordinate(to_file, self, name, include_attrs=include_attrs, **kwargs)

        return ds

    def __getitem__(self, key):
        return self.encoding.decode(self._dataset[key])

    def __setitem__(self, key, value):
        """

        """
        self._dataset[key] = self.encoding.encode(value)
        self.data = self[()]

    def iter_chunks(self, sel=None):
        return self._dataset.iter_chunks(sel)

    def __bool__(self):
        return self._dataset.__bool__()

    def len(self):
        return self._dataset.len()

    def __repr__(self):
        """

        """
        return coordinate_summary(self)


    def to_pandas(self):
        """

        """
        if not import_pandas:
            raise ImportError('pandas could not be imported.')

        return pd.Index(self.data, name=self.name)


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
    def __init__(self, attrs: h5py.AttributeManager, dtype, writable, encoding: dict=None):
        if encoding is None:
            enc = utils.get_encoding_data_from_attrs(attrs)
        else:
            enc = utils.get_encoding_data_from_attrs(encoding)
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

    def encode(self, values):
        return utils.encode_data(values, **self._encoding)

    def decode(self, values):
        return utils.decode_data(values, **self._encoding)


# class EncodeDecode:
#     """

#     """
#     def __init__(self, encoding):
#         """

#         """
#         self.encoding

#     def encode(self, values):
#         return utils.encode_data(values, **self.encoding._encoding)

#     def decode(self, values):
#         return utils.decode_data(values, **self.encoding._encoding)



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
    def __init__(self, variable: Variable):
        """

        """
        self.variable = variable


    def __getitem__(self, key):
        """

        """
        if isinstance(key, (int, float, str, slice, list, np.ndarray)):
            index = index_combo_one(key, self.variable, 0)

            return self.variable.encoding.decode(self.variable[index])

        elif isinstance(key, tuple):
            key_len = len(key)

            if key_len == 0:
                return self.variable.encoding.decode(self.variable[()])

            elif key_len > self.variable.ndim:
                raise ValueError('input must have <= ndims.')

            index = []
            for i, k in enumerate(key):
                index_i = index_combo_one(k, self.variable, i)
                index.append(index_i)

            return self.variable.encoding.decode(self.variable[tuple(index)])

        else:
            raise ValueError('You passed a strange object to index...')


    def __setitem__(self, key, value):
        """

        """
        if isinstance(key, (int, float, str, slice, list, np.ndarray)):
            index = index_combo_one(key, self.variable, 0)

            self.variable[index] = self.variable.encoding.encode(value)

        elif isinstance(key, tuple):
            key_len = len(key)

            if key_len == 0:
                self.variable[()] = self.variable.encoding.encode(value)

            elif key_len > self.variable.ndim:
                raise ValueError('input must have <= ndims.')

            index = []
            for i, k in enumerate(key):
                index_i = index_combo_one(k, self.variable, i)
                index.append(index_i)

            self.variable[tuple(index)] = self.variable.encoding.encode(value)

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
                raise ValueError(f'{start} not in coordinate.')

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
                raise ValueError(f'{stop} not in coordinate.')

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
            raise ValueError(f'{label} not in coordinate.')

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
def index_combo_one(key, variable, pos):
    """

    """
    if isinstance(key, (int, float, str)):
        dim_data = variable.file[variable.dims[pos]].data
        label_idx = index_label(key, dim_data)

        return label_idx

    elif isinstance(key, slice):
        dim_data = variable.file[variable.dims[pos]].data
        slice_idx = index_slice(key, dim_data)

        return slice_idx

    elif key is None:
         return slice(None, None)

    elif isinstance(key, (list, np.ndarray)):
        key = np.asarray(key)

        dim_data = variable.file[variable.dims[pos]].data

        if key.dtype.name == 'bool':
            if len(key) != len(dim_data):
                raise ValueError('If the input is a bool array, then it must be the same length as the coordinate.')

            return key
        else:
            idx = index_array(key, dim_data)

            return idx



def copy_variable(to_file: File, from_variable: Variable, name, include_data=True, include_attrs=True, **kwargs):
    """

    """
    other1 = from_variable._dataset
    for k in ('chunks', 'compression',
              'compression_opts', 'scaleoffset', 'shuffle', 'fletcher32',
              'fillvalue'):
        kwargs.setdefault(k, getattr(other1, k))

    if 'compression' in other1.attrs:
        compression = other1.attrs['compression']
        kwargs.update(**utils.get_compressor(compression))
    else:
        compression = kwargs['compression']

    # TODO: more elegant way to pass these (dcpl to create_variable?)
    dcpl = other1.id.get_create_plist()
    kwargs.setdefault('track_times', dcpl.get_obj_track_times())
    # kwargs.setdefault('track_order', dcpl.get_attr_creation_order() > 0)

    # Special case: the maxshape property always exists, but if we pass it
    # to create_variable, the new variable will automatically get chunked
    # layout. So we copy it only if it is different from shape.
    if other1.maxshape != other1.shape:
        kwargs.setdefault('maxshape', other1.maxshape)

    encoding = from_variable.encoding._encoding.copy()
    shape = from_variable.shape

    if include_data:
        ds0 = create_h5py_variable(to_file, name, tuple(dim.label for dim in other1.dims), shape, encoding, **kwargs)

        # Directly copy chunks using write_direct_chunk
        for chunk in ds0.iter_chunks():
            chunk_starts = tuple(c.start for c in chunk)
            filter_mask, data = other1.id.read_direct_chunk(chunk_starts)
            ds0.id.write_direct_chunk(chunk_starts, data, filter_mask)

    else:
        ds0 = create_h5py_variable(to_file, name, tuple(dim.label for dim in other1.dims), shape, encoding, **kwargs)

    ds = Variable(ds0, to_file, encoding)
    if include_attrs:
        ds.attrs.update(from_variable.attrs)

    return ds


def copy_coordinate(to_file: File, from_coordinate: Coordinate, name, include_attrs=True, **kwargs):
    """

    """
    other1 = from_coordinate._dataset
    for k in ('chunks', 'compression',
              'compression_opts', 'scaleoffset', 'shuffle', 'fletcher32',
              'fillvalue'):
        kwargs.setdefault(k, getattr(other1, k))

    if 'compression' in other1.attrs:
        compression = other1.attrs['compression']
        kwargs.update(**utils.get_compressor(compression))
    else:
        compression = kwargs['compression']

    # TODO: more elegant way to pass these (dcpl to create_variable?)
    dcpl = other1.id.get_create_plist()
    kwargs.setdefault('track_times', dcpl.get_obj_track_times())
    # kwargs.setdefault('track_order', dcpl.get_attr_creation_order() > 0)

    # Special case: the maxshape property always exists, but if we pass it
    # to create_variable, the new variable will automatically get chunked
    # layout. So we copy it only if it is different from shape.
    if other1.maxshape != other1.shape:
        kwargs.setdefault('maxshape', other1.maxshape)

    encoding = from_coordinate.encoding._encoding.copy()
    shape = from_coordinate.shape

    ds0 = create_h5py_coordinate(to_file, name, from_coordinate.data, shape, encoding, **kwargs)

    ds = Coordinate(ds0, to_file, encoding)
    if include_attrs:
        ds.attrs.update(from_coordinate.attrs)

    return ds





d




##############################################
### Testing

name = '/media/data01/cache/tethys/test/test1.h5'
nc1 = '/media/data01/cache/tethys/test/2m_temperature_1950-1957_reanalysis-era5-land.nc'
nc2 = '/media/data01/cache/tethys/test/test2.nc'
nc3 = '/media/data01/cache/tethys/test/test3.nc'

dims = ('dim1', 'dim2')
dim1_len = 100
dim2_len = 10000

dim1_data = np.arange(dim1_len, dim1_len*2, dtype='int16')
dim1_data = np.arange(dim1_len, dim1_len*2, dtype='datetime64[h]')
dim2_data = np.arange(dim2_len, dim2_len*2, dtype='int16')

self = File(name, mode='w', compression='zstd')

# if 'dim1' in self:
#     del self['dim1']
# if 'dim2' in self:
#     del self['dim2']
# if 'test1' in self:
#     del self['test1']

dim1_test = self.create_coordinate('dim1', data=dim1_data)
dim2_test = self.create_coordinate('dim2', data=dim2_data)

test1_data = np.arange(len(dim1_data)*len(dim2_data), dtype='int32').reshape(len(dim1_data), len(dim2_data))
test4_data = np.arange(len(dim1_data), dtype='int32')

test1_ds = self.create_variable('test1', dims, data=test1_data)
test2_ds = self.create_variable_like(test1_ds, 'test2', include_data=True)
test3_ds = test2_ds.copy(name='test3')
test4_ds = self.create_variable('test4', dims[0], data=test4_data)

self = h5py.File(name, 'w', userblock_size=512)
self = h5py.File(name, 'r+')

with open(name, 'rb') as f:
    b1 = f.read()



self.close()


x1 = xr.open_variable(nc1)
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

self = File(nc2)



