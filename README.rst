hdf5-tools
==================================

This git repository contains a python package with an H5 class to load and combine one or more HDF5 data files (or xarray datasets) with optional filters. The class will then export the combined data to an HDF5 file, file object, or xr.Dataset. This class is designed to be fast and safe on memory. This means that files of any size can be combined and saved even on a PC with low memory (unlike xarray).
