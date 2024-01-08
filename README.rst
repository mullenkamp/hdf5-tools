hdf5-tools
==================================

This git repository contains a python package with an H5 class to load and combine one or more HDF5 data files (or xarray datasets) with optional filters. The class will then export the combined data to an HDF5 file, file object, or xr.Dataset. This class is designed to be fast and safe on memory. This means that files of any size can be combined and saved even on a PC with low memory (unlike xarray).

Installation
------------
Using pip:

.. code::

    pip install hdf5tools


Or using conda/mamba from conda-forge:

.. code::

    conda install -c conda-forge hdf5tools


Usage
-------
Currently, only the **Combine** class is recommended for other to use.

First, initiate the class with one or many: paths to netcdf3, netcdf4, or hdf5 files; xr.Dataset objects (opened via xr.open_dataset); or h5py.File objects.

.. code:: python

    from hdf5tools import Combine

    ###############################
    ### Parameters

    path1 = '/path/to/file1.nc'
    path2 = '/path/to/file2.nc'

    ##############################
    ### Combine files

    c1 = Combine([path1, path2])



If you want to do some kind of selection via the coordinates or only select some of the data variables/coordinates then use the **.sel** method (like in xarray). Be sure to read the docstrings for additional info about the input parameters.

.. code:: python

    c2 = c1.sel({'time': slice('2001-01-01', '2020-01-01'), 'latitude': slice(0, 10)}, include_data_vars=['precipitation'])


And finally, save the combined data to a single hdf5/netcdf4 file using the **.to_hdf5** method. The only additional parameters that are important include the output which should be a path or a io.Bytes object, and the compression parameter. If you plan on using this file outside of the python environment, use gzip for compression, otherwise use lzf. The docstrings have more details.


.. code:: python

    output = '/path/to/output.nc'

    c2.to_hdf5(output, compression='gzip')


If you've passed xr.Dataset objects to Combine, it will be slower than passing the file as a path on disk. Only pass xr.Dataset objects to Combine if you don't want to write the intermediate file to disk before reading it into Combine.

The package also comes with a bonus function called **xr_to_hdf5**. It is a convenience function to convert a single xr.Dataset to an hdf5/netcdf4 file.

.. code:: python

    from hdf5tools import xr_to_hdf5

    xr_to_hdf5(xr_dataset, output, compression='gzip')

