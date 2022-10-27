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

files = glob(base_path + '/*.h5')
files.sort()

ds_ids = set([os.path.split(f)[-1].split('_')[0] for f in files])

for ds_id in ds_ids:
    ds_files = [xr.open_dataset(f, engine='h5netcdf') for f in files if ds_id in f]
    h1 = H5(ds_files)
    new_path = os.path.join(base_path, ds_id + '_test1.h5')
    h1.to_hdf5(new_path)


@pytest.mark.parametrize('remote', remotes)
def test_tethys(remote):
    """

    """
    t1 = Tethys([remote['remote']])

    ## Datasets
    datasets = t1.datasets
    assert len(datasets) > remote['assert']['datasets']

    ## Stations
    stn_list1 = t1.get_stations(remote['dataset_id'])
    assert len(stn_list1) > remote['assert']['stations']

    ## Versions
    rv1 = t1.get_versions(remote['dataset_id'])
    assert len(rv1) > remote['assert']['versions']

    ## Results
    data1 = t1.get_results(remote['dataset_id'], remote['station_ids'])
    assert len(data1) > remote['assert']['results']


## initialise for the rest of the tests
t1 = Tethys([remote3])


@pytest.mark.parametrize('output', outputs)
def test_get_results(output):
    data1 = t1.get_results(dataset_id, station_ids, squeeze_dims=True, output=output)

    if output == 'xarray':
        assert len(data1.time) > 90
    elif output == 'dict':
        assert len(data1['coords']['time']['data']) > 90
    elif output == 'json':
        assert len(data1) > 90


def test_get_nearest_station1():
    s1 = t1.get_stations(dataset_id, geometry1)

    assert len(s1) == 1


def test_get_nearest_station2():
    s2 = t1.get_stations(dataset_id, lat=lat, lon=lon)

    assert len(s2) == 1


def test_get_intersection_stations1():
    s3 = t1.get_stations(dataset_id, lat=lat, lon=lon, distance=distance)

    assert len(s3) >= 2


def test_get_nearest_results1():
    s1 = t1.get_results(dataset_id, geometry=geometry1)

    assert len(s1) > 0


def test_get_nearest_results2():
    s2 = t1.get_results(dataset_id, lat=lat, lon=lon)

    assert len(s2) > 0


# def test_get_intersection_results1():
#     s3 = t1.get_results(dataset_id, lat=lat, lon=lon, distance=distance)
#
#     assert len(s3) > 1
