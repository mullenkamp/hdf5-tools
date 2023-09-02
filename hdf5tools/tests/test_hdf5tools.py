"""
Created on 2021-04-27.

@author: Mike K
"""
from hdf5tools import H5
import numpy as np
import os
import pytest
from glob import glob
import xarray as xr

##############################################
### Parameters

base_path = os.path.join(os.path.split(os.path.realpath(os.path.dirname(__file__)))[0], 'datasets')


#############################################
### Functions


def xr_concat(datasets):
    """
    A much more efficient concat/combine of xarray datasets. It's also much safer on memory.
    """
    # Get variables for the creation of blank dataset
    coords_list = []
    chunk_dict = {}

    for chunk in datasets:
        coords_list.append(chunk.coords.to_dataset())
        for var in chunk.data_vars:
            if var not in chunk_dict:
                dims = tuple(chunk[var].dims)
                enc = chunk[var].encoding.copy()
                dtype = chunk[var].dtype
                _ = [enc.pop(d) for d in ['original_shape', 'source'] if d in enc]
                var_dict = {'dims': dims, 'enc': enc, 'dtype': dtype, 'attrs': chunk[var].attrs}
                chunk_dict[var] = var_dict

    try:
        xr3 = xr.combine_by_coords(coords_list, compat='override', data_vars='minimal', coords='all', combine_attrs='override')
    except:
        xr3 = xr.merge(coords_list, compat='override', combine_attrs='override')

    # Run checks - requires psutil which I don't want to make it a dep yet...
    # available_memory = getattr(psutil.virtual_memory(), 'available')
    # dims_dict = dict(xr3.coords.dims)
    # size = 0
    # for var, var_dict in chunk_dict.items():
    #     dims = var_dict['dims']
    #     dtype_size = var_dict['dtype'].itemsize
    #     n_dims = np.prod([dims_dict[dim] for dim in dims])
    #     size = size + (n_dims*dtype_size)

    # if size >= available_memory:
    #     raise MemoryError('Trying to create a dataset of size {}MB, while there is only {}MB available.'.format(int(size*10**-6), int(available_memory*10**-6)))

    # Create the blank dataset
    for var, var_dict in chunk_dict.items():
        dims = var_dict['dims']
        shape = tuple(xr3[c].shape[0] for c in dims)
        xr3[var] = (dims, np.full(shape, np.nan, var_dict['dtype']))
        xr3[var].attrs = var_dict['attrs']
        xr3[var].encoding = var_dict['enc']

    # Update the attributes in the coords from the first ds
    for coord in xr3.coords:
        xr3[coord].encoding = datasets[0][coord].encoding
        xr3[coord].attrs = datasets[0][coord].attrs

    # Fill the dataset with data
    for chunk in datasets:
        for var in chunk.data_vars:
            if isinstance(chunk[var].variable._data, np.ndarray):
                xr3[var].loc[chunk[var].transpose(*chunk_dict[var]['dims']).coords.indexes] = chunk[var].transpose(*chunk_dict[var]['dims']).values
            elif isinstance(chunk[var].variable._data, xr.core.indexing.MemoryCachedArray):
                c1 = chunk[var].copy().load().transpose(*chunk_dict[var]['dims'])
                xr3[var].loc[c1.coords.indexes] = c1.values
                c1.close()
                del c1
            else:
                raise TypeError('Dataset data should be either an ndarray or a MemoryCachedArray.')

    return xr3


######################################
### Testing

files = glob(base_path + '/*.nc')
files.sort()

ds_ids = set([os.path.split(f)[-1].split('_')[0] for f in files])

## Test data
before_dict = {}
for ds_id in ds_ids:
    before = xr_concat([xr.open_dataset(f, engine='h5netcdf') for f in files if ds_id in f])
    before_dict[ds_id] = before


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
    # print(h1)
    new_path = os.path.join(base_path, ds_id + '_test1.h5')
    h1.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    # print(x1)

    ## Compare before and after
    before = before_dict[ds_id]
    assert before.equals(x1)

    first_times = x1.time.values[0:5]
    x1.close()
    h2 = h1.sel({'time': slice(first_times[0], first_times[-1])})
    # print(h2)
    h2.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    x1.load()
    before2 = before.sel({'time': slice(first_times[0], first_times[-2])})
    # print(x1)
    assert before2.equals(x1)

    main_vars = [v for v in list(x1.data_vars) if set(x1[v].dims) == set(x1.dims)]
    x1.close()
    h2 = h1.sel(include_data_vars=main_vars)
    # print(h2)
    h2.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    x1.load()
    # print(x1)
    assert before[main_vars].equals(x1)
    x1.close()

    os.remove(new_path)


@pytest.mark.parametrize('ds_id', ds_ids)
def test_H5_hdf5(ds_id):
    """

    """
    ds_files = [f for f in files if ds_id in f]
    h1 = H5(ds_files)
    # print(h1)
    new_path = os.path.join(base_path, ds_id + '_test1.h5')
    h1.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    # print(x1)

    ## Compare before and after
    before = before_dict[ds_id]
    assert before.equals(x1)

    first_times = x1.time.values[0:5]
    x1.close()
    h2 = h1.sel({'time': slice(first_times[0], first_times[-1])})
    # print(h2)
    h2.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    x1.load()
    before2 = before.sel({'time': slice(first_times[0], first_times[-2])})
    # print(x1)
    assert before2.equals(x1)

    main_vars = [v for v in list(x1.data_vars) if set(x1[v].dims) == set(x1.dims)]
    x1.close()
    h2 = h1.sel(include_data_vars=main_vars)
    # print(h2)
    h2.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    x1.load()
    # print(x1)
    assert before[main_vars].equals(x1)
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
    # print(h1)
    new_path = os.path.join(base_path, ds_id + '_test1.h5')
    h1.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    # print(x1)

    ## Compare before and after
    before = before_dict[ds_id]
    assert before.equals(x1)

    first_times = x1.time.values[0:5]
    x1.close()
    h2 = h1.sel({'time': slice(first_times[0], first_times[-1])})
    # print(h2)
    h2.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    x1.load()
    before2 = before.sel({'time': slice(first_times[0], first_times[-2])})
    # print(x1)
    assert before2.equals(x1)

    main_vars = [v for v in list(x1.data_vars) if set(x1[v].dims) == set(x1.dims)]
    x1.close()
    h2 = h1.sel(include_data_vars=main_vars)
    # print(h2)
    h2.to_hdf5(new_path)
    x1 = xr.open_dataset(new_path, engine='h5netcdf')
    x1.load()
    # print(x1)
    assert before[main_vars].equals(x1)

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
