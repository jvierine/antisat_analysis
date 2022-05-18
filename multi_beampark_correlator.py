#!/usr/bin/env python

import argparse
import pathlib
import pickle

import matplotlib.pyplot as plt
from tabulate import tabulate

import numpy as np
import h5py
from astropy.time import Time

import sorts

'''
Example execution

python multi_beampark_correlator.py \
    ~/data/spade/beamparks/uhf/2021.11.23/space-track.tles \
    -o ~/data/spade/beamparks/uhf/2021.11.23/dual_correlations.npy \
    ~/data/spade/beamparks/{uhf,esr}/2021.11.23/correlation.h5

python multi_beampark_correlator.py \
    /home/danielk/git/antisat_analysis/projects/output/russian_asat/2021_11_23_spacetrack.tle \
    -o /home/danielk/git/antisat_analysis/projects/output/russian_asat/orbit_determination/2021-11-23_dual_correlations.npy \
    /home/danielk/git/antisat_analysis/projects/output/russian_asat/2021.11.23_uhf_correlation.pickle \
    /home/danielk/git/antisat_analysis/projects/output/russian_asat/2021.11.23_uhf_correlation_plots/eiscat_uhf_selected_correlations.npy \
    /home/danielk/git/antisat_analysis/projects/output/russian_asat/2021.11.23_esr_correlation.pickle \
    /home/danielk/git/antisat_analysis/projects/output/russian_asat/2021.11.23_esr_correlation_plots/eiscat_esr_selected_correlations.npy

python multi_beampark_correlator.py \
    /home/danielk/git/antisat_analysis/projects/output/russian_asat/2021_11_23_spacetrack.tle \
    -o /home/danielk/git/antisat_analysis/projects/output/russian_asat/orbit_determination/2021-11-23_dual_correlations_v2.npy \
    /home/danielk/git/antisat_analysis/projects/output/russian_asat/2021.11.23_uhf_correlation_v2.pickle \
    /home/danielk/git/antisat_analysis/projects/output/russian_asat/2021.11.23_uhf_correlation_plots_v2/eiscat_uhf_selected_correlations.npy \
    /home/danielk/git/antisat_analysis/projects/output/russian_asat/2021.11.23_esr_correlation_v2.pickle \
    /home/danielk/git/antisat_analysis/projects/output/russian_asat/2021.11.23_esr_correlation_plots_v2/eiscat_esr_selected_correlations.npy
'''


def get_matches(data, bp, index):
    return data[bp['beampark'][index]]['metric'][bp['measurement_id'][index]]


def main(input_args=None):
    parser = argparse.ArgumentParser(description='Analyse beampark correlation for a beampark')
    parser.add_argument('catalog', type=str, help='Catalog to which data was correlated')
    parser.add_argument('inputs', type=str, nargs='+', help='Input correlation data files and correlation selection files (input as pairs of "correlation_file selection_file")')
    parser.add_argument('-o', '--output', type=str, default='', help='Output OID save location')
    parser.add_argument('--target-epoch', type=str, default=None, help='When filtering unique TLEs use this target epoch [ISO]')

    # This assumes correlation was done using the exact same TLE population so that
    #  object indecies match up in each correlation file

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    assert len(args.inputs) % 2 == 0

    tle_pth = pathlib.Path(args.catalog).resolve()
    input_pths = [pathlib.Path(x).resolve() for x in args.inputs[0::2]]
    select_pths = [pathlib.Path(x).resolve() for x in args.inputs[1::2]]

    if len(input_pths) < 2:
        raise ValueError('Need at least 2 correlation files')

    match_data = {}
    objects = {}

    for ind, (input_pth, select_pth) in enumerate(zip(input_pths, select_pths)):
        print(f'Loading: {input_pth.name} and {select_pth.name}')
        with h5py.File(input_pth, 'r') as ds:
            indecies = ds['matched_object_index'][()]
            results = ds['matched_object_metric'][()]
            unix_times = ds['matched_object_time'][()]
            name = ds.attrs['radar_name']

        select = np.load(select_pth)

        print(f'finished loading {name} [{results.shape} records]')

        # Pick only best matches in row 0
        match_data[ind] = {
            'match': indecies[0, :], 
            'metric': results[0, :],
            'times': unix_times,
            'correlated': select,
        }

        for oid in np.unique(match_data[ind]['match']):
            indecies = np.where(np.logical_and(
                match_data[ind]['match'] == oid,
                match_data[ind]['correlated'],
            ))[0]
            if indecies.size == 0:
                continue

            if oid not in objects:
                objects[oid] = {
                    'beampark': [ind], 
                    'measurement_id': [indecies],
                    'measurement_time': [unix_times[indecies]],
                    'num': 1,
                }
            else:
                objects[oid]['beampark'].append(ind)
                objects[oid]['measurement_id'].append(indecies)
                objects[oid]['measurement_time'].append(unix_times[indecies])
                objects[oid]['num'] += 1
    

    possible_multi = 0
    multi_match_oids = []
    multi_match_mids = []
    multi_match_datas = []
    multi_match_times = []
    for oid, bp in objects.items():
        # skip all objects only seen in one beampark
        if bp['num'] < 2:
            continue

        mids = [np.nan]*len(input_pths)
        for bpid, mid in zip(bp['beampark'], bp['measurement_id']):
            # Assume only once in each beampark, this will have to be updated 
            # if the same object is detected twice in one experiment
            mids[bpid] = mid[0]
        multi_match_mids.append(mids)
        multi_match_oids.append(oid)

        match_datas = {}
        match_times = {}
        for index in range(bp['num']):
            match = get_matches(match_data, bp, index)
            ID = bp['beampark'][index]
            match_datas[ID] = match[0] # Assume only one

            # Assume only one
            _t = Time(bp["measurement_time"][index][0], format='unix')
            match_times[ID] = _t

            print(f'Catalog-index (oid) - {oid:<6}: Beampark-{ID} @ {_t.iso} -> \
                residuals=({match["dr"][0]: .3e} m, {match["dv"][0]: .3e} m/s) -> \
                combined={match["metric"][0]:.3e}')
        multi_match_datas.append(match_datas)
        multi_match_times.append(match_times)

        possible_multi += 1

    multi_match_mids = np.array(multi_match_mids)

    print(f'Possible multi-beampark observed objects: {possible_multi}')

    print('Loading TLE population')
    pop = sorts.population.tle_catalog(tle_pth, cartesian=False)
    pop.out_frame = 'ITRS'

    if args.target_epoch is not None:
        args.target_epoch = Time(args.target_epoch, format='iso', scale='utc').mjd
    pop.unique(target_epoch=args.target_epoch)

    print(f'Population size: {len(pop)}')

    get_fields = ['oid', 'mjd0', 'line1', 'line2']

    _dtype = [(f'mid{ind}', np.int64) for ind in range(len(input_pths))]
    _dtype += [('cid', np.int64)]
    for key in get_fields:
        _dtype += [(key, pop.dtypes[pop.fields.index(key)])]
    _dtype += [(f'dr{ind}', np.float64) for ind in range(len(input_pths))]
    _dtype += [(f'dv{ind}', np.float64) for ind in range(len(input_pths))]
    _dtype += [(f'metric{ind}', np.float64) for ind in range(len(input_pths))]
    _dtype += [(f'jitter{ind}', np.int64) for ind in range(len(input_pths))]

    _data = pop.data[multi_match_oids][get_fields]
    _data_m = np.empty((len(_data),), dtype=_dtype)
    _data_m['cid'] = multi_match_oids
    for ind in range(len(input_pths)):
        _data_m[f'mid{ind}'] = multi_match_mids[:, ind]
        print(f'mid{ind}=', multi_match_mids[:, ind])
    for key in get_fields:
        _data_m[key] = _data[key]
    for dind, match_datas in enumerate(multi_match_datas):
        for index, mdata in match_datas.items():
            _data_m[f'dr{index}'][dind] = mdata['dr']
            _data_m[f'dv{index}'][dind] = mdata['dv']
            _data_m[f'metric{index}'][dind] = mdata['metric']
            _data_m[f'jitter{index}'][dind] = mdata['jitter_index']

    if len(args.output) > 0:
        output = pathlib.Path(args.output)
        if not output.parent.is_dir():
            output.parent.mkdir(exist_ok=True, parents=True)
        np.save(args.output, _data_m)

    print(tabulate(_data_m, headers=[f'MID-{ind}' for ind in range(len(input_pths))] + ['OID', 'NORAD-ID', 'mjd0', 'line1', 'line2']))


if __name__ == '__main__':
    main()
