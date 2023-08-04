#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:07:35 2022

@author: mike
"""
from tethysts import Tethys
import os

#############################################
### Parameters

base_path = os.path.join(os.path.split(os.path.realpath(os.path.dirname(__file__)))[0], 'datasets')

remotes = [{'bucket': 'nz-open-modelling-consortium', 'public_url': 'https://b2.nzrivers.xyz/file/', 'version': 4},
           {'bucket': 'fire-emergency-nz', 'public_url': 'https://b2.tethys-ts.xyz/file/', 'version': 4},
           {'bucket': 'ecan-env-monitoring', 'public_url': 'https://b2.tethys-ts.xyz/file', 'version': 4},
           {'bucket': 'gwrc-env', 'public_url': 'https://b2.tethys-ts.xyz/file', 'version': 4}
           ]

dataset_ids = {'7751c5f1bf47867fb109d7eb': [10], '0b2bd62cc42f3096136f11e9': None, 'f16774ea29f024a306c7fc7a': None, '9568f663d566aabb62a8e98e': None}

############################################
### Get data

t1 = Tethys(remotes)

for ds_id, heights in dataset_ids.items():
    stns1 = t1.get_stations(ds_id)
    station_ids = [s['station_id'] for s in stns1[:2]]
    for stn_id in station_ids:
        file_name = '{ds_id}_{stn_id}.nc'.format(ds_id=ds_id, stn_id=stn_id)
        file_path = os.path.join(base_path, file_name)
        results = t1.get_results(ds_id, stn_id, heights=heights, output_path=file_path, compression='zstd')
