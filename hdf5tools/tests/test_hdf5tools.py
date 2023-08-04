"""
Created on 2021-04-27.

@author: Mike K
"""
from hdf5tools import H5
import os
import pytest
from glob import glob
import xarray as xr

##############################################
### Parameters

base_path = os.path.join(os.path.split(os.path.realpath(os.path.dirname(__file__)))[0], 'datasets')


######################################
### Testing

files = glob(base_path + '/*.nc')
files.sort()

ds_ids = set([os.path.split(f)[-1].split('_')[0] for f in files])

# for ds_id in ds_ids:
#     ds_files = [xr.open_dataset(f, engine='h5netcdf') for f in files if ds_id in f]
#     h1 = H5(ds_files)
#     print(h1)
#     new_path = os.path.join(base_path, ds_id + '_test1.h5')
#     h1.to_hdf5(new_path)
#     x1 = xr.open_dataset(new_path, engine='h5netcdf')
#     print(x1)

#     first_times = x1.time.values[0:5]
#     x1.close()
#     h2 = h1.sel({'time': slice(first_times[0], first_times[-1])})
#     print(h2)
#     h2.to_hdf5(new_path)
#     x1 = xr.open_dataset(new_path, engine='h5netcdf')
#     print(x1.load())
#     assert x1.time.shape[0] == 4

#     main_vars = [v for v in list(x1.data_vars) if set(x1[v].dims) == set(x1.dims)]
#     x1.close()
#     h2 = h1.sel(include_data_vars=main_vars)
#     print(h2)
#     h2.to_hdf5(new_path)
#     x1 = xr.open_dataset(new_path, engine='h5netcdf')
#     print(x1.load())
#     assert set(x1.data_vars) == set(main_vars)
#     x1.close()

#     os.remove(new_path)

# for ds_id in ds_ids:
#     ds_files = []

#     for i, f in enumerate(files):
#         if ds_id in f:
#             if i == 0:
#                 ds_files.append(f)
#             else:
#                 ds_files.append(xr.open_dataset(f, engine='h5netcdf'))

#     h1 = H5(ds_files)
#     print(h1)
#     new_path = os.path.join(base_path, ds_id + '_test1.h5')
#     h1.to_hdf5(new_path)
#     x1 = xr.open_dataset(new_path, engine='h5netcdf')
#     print(x1)

#     first_times = x1.time.values[0:5]
#     x1.close()
#     h2 = h1.sel({'time': slice(first_times[0], first_times[-1])})
#     print(h2)
#     h2.to_hdf5(new_path)
#     x1 = xr.open_dataset(new_path, engine='h5netcdf')
#     print(x1.load())
#     assert x1.time.shape[0] == 4

#     main_vars = [v for v in list(x1.data_vars) if set(x1[v].dims) == set(x1.dims)]
#     x1.close()
#     h2 = h1.sel(include_data_vars=main_vars)
#     print(h2)
#     h2.to_hdf5(new_path)
#     x1 = xr.open_dataset(new_path, engine='h5netcdf')
#     print(x1.load())
#     assert set(x1.data_vars) == set(main_vars)
#     x1.close()

#     os.remove(new_path)


@pytest.mark.parametrize('ds_id', ds_ids)
def test_H5_xr(ds_id):
    """

    """
    ds_files = [xr.open_dataset(f, engine='h5netcdf') for f in files if ds_id in f]
    h1 = H5(ds_files)
    print(h1)
    new_path = os.path.join(base_path, ds_id + '_test1.h5')
    h1.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    print(x1)

    first_times = x1.time.values[0:5]
    x1.close()
    h2 = h1.sel({'time': slice(first_times[0], first_times[-1])})
    print(h2)
    h2.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    print(x1.load())
    assert x1.time.shape[0] == 4

    main_vars = [v for v in list(x1.data_vars) if set(x1[v].dims) == set(x1.dims)]
    x1.close()
    h2 = h1.sel(include_data_vars=main_vars)
    print(h2)
    h2.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    print(x1.load())
    assert set(x1.data_vars) == set(main_vars)
    x1.close()

    os.remove(new_path)


@pytest.mark.parametrize('ds_id', ds_ids)
def test_H5_hdf5(ds_id):
    """

    """
    ds_files = [f for f in files if ds_id in f]
    h1 = H5(ds_files)
    print(h1)
    new_path = os.path.join(base_path, ds_id + '_test1.h5')
    h1.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    print(x1)

    first_times = x1.time.values[0:5]
    x1.close()
    h2 = h1.sel({'time': slice(first_times[0], first_times[-1])})
    print(h2)
    h2.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    print(x1.load())
    assert x1.time.shape[0] == 4

    main_vars = [v for v in list(x1.data_vars) if set(x1[v].dims) == set(x1.dims)]
    x1.close()
    h2 = h1.sel(include_data_vars=main_vars)
    print(h2)
    h2.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    print(x1.load())
    assert set(x1.data_vars) == set(main_vars)
    x1.close()

    os.remove(new_path)


@pytest.mark.parametrize('ds_id', ds_ids)
def test_H5_mix(ds_id):
    """

    """
    ds_files = []

    for i, f in enumerate(files):
        if ds_id in f:
            if i == 0:
                ds_files.append(f)
            else:
                ds_files.append(xr.open_dataset(f, engine='h5netcdf'))

    h1 = H5(ds_files)
    print(h1)
    new_path = os.path.join(base_path, ds_id + '_test1.h5')
    h1.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    print(x1)

    first_times = x1.time.values[0:5]
    x1.close()
    h2 = h1.sel({'time': slice(first_times[0], first_times[-1])})
    print(h2)
    h2.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    print(x1.load())
    assert x1.time.shape[0] == 4

    main_vars = [v for v in list(x1.data_vars) if set(x1[v].dims) == set(x1.dims)]
    x1.close()
    h2 = h1.sel(include_data_vars=main_vars)
    print(h2)
    h2.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    print(x1.load())
    assert set(x1.data_vars) == set(main_vars)
    x1.close()

    os.remove(new_path)



def min_required_for_netcdf4():
    """
    The minimum requirements for making the hdf5 file netcdf4 compatible is the  libver='v110' (or earlier), all the track_order=True, and the scale assignments and labels.
    """
    import h5py
    import numpy as np

    output = '/media/data01/cache/hdf5tools/test0.h5'

    conc = np.arange(1, 101, dtype='int8')
    n_samples = np.arange(1, 10000, dtype='int32')

    dims = {'conc': conc, 'n_samples': n_samples}

    data = np.zeros((len(conc), len(n_samples)), dtype='int8')

    with h5py.File(output, 'w', libver='v110', track_order=True) as nf:
        for name, val in dims.items():
            dim_ds = nf.create_dataset(name, val.shape, dtype=val.dtype, track_order=True)
            dim_ds[:] = val
            dim_ds.make_scale(name)
            dim_ds.dims[0].label = name

        data_ds = nf.create_dataset('data', data.shape, dtype=data.dtype, track_order=True)
        data_ds[:] = data
        data_ds.dims[0].attach_scale(nf['conc'])
        data_ds.dims[0].label = 'conc'
        data_ds.dims[1].attach_scale(nf['n_samples'])
        data_ds.dims[1].label = 'n_samples'
