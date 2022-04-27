#!/usr/bin/env python

import pathlib
import datetime
import argparse

import numpy as np
import pandas as pd
import h5py
from astropy.time import Time

'''Example usage:

python convert_spade_events_to_h5.py ./leo_bpark_2.1u_CN@uhf/ ./h5/leo.h5
'''

MIN_SNR = 33

names_v1_4 = [
    'YYYY', 'MM', 'DD', 
    'hh', 'mm', 'ss.s', 
    'TX', 'AZ', 'EL', 
    'RT', 'RG', 'RR', 
    'VD', 'AD', 'DI', 
    'CS', 'TS', 'EN', 
    'ED', 'TP', 'MT',
]

names_v1_6 = [
    'YYYY', 'MM', 'DD', 
    'hh', 'mm', 'ss.s', 
    'ID', 'TX', 
    'ST', 'AZ', 'EL', 
    'HT', 'RT', 'RG', 
    'RR', 'VD', 'AD', 
    'DI', 'CS', 'TS', 
    'EN', 'ED', 'TP', 
    'MT', 
]


def date2unix(year, month, day, hour, minute, second):
    dt = datetime.datetime(year, month, day, hour, minute, second)
    timestamp = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return timestamp


def get_spade_data(files, verbose=False):
    data = None
    for file in files:
        with open(file, 'r') as fh:
            first_line = fh.readline().strip().split()

        if len(first_line) < 3:
            next_data = None
        elif first_line[2] == '1.4':
            next_data = pd.read_csv(
                file, 
                sep=r'[ ]+', 
                comment='%', 
                skip_blank_lines=True, 
                names=names_v1_4,
                engine='python',
            )
        elif first_line[2] == '1.6' or first_line[2] == '1.5':
            next_data = pd.read_csv(
                file, 
                sep=r'[ ]+', 
                comment='%', 
                skip_blank_lines=True, 
                names=names_v1_6,
                engine='python',
            )
        else:
            next_data = None

        if next_data is None:
            if verbose:
                print(f'{file}: File not known SPADE-file...')
            continue
        else:
            if verbose:
                print(f'{file}: Detected SPADE-file version {first_line[2]}...')

        if data is None:
            data = next_data
        else:
            data = pd.concat([data, next_data])

    return data


def read_extended_event_data(input_summaries, args):
    _event_datas = []
    for events_file in input_summaries:
        _event_data = get_spade_data([events_file], verbose=args.v)

        _extended_files = []
        with open(events_file, 'r') as fh:
            for ind, line in enumerate(fh):
                if ind < 33 or len(line.strip()) == 0:
                    continue
                ev_name = line[181:].strip()
                _extended_files.append(ev_name)
        _event_data['event_name'] = _extended_files

        # Filter like the h5 generator filters
        disc = np.argwhere(_event_data['RT'].values**2.0 <= MIN_SNR).flatten()
        _event_data.drop(disc, inplace=True)
        _event_datas.append(_event_data)

    event_data = pd.concat(_event_datas)

    dts = []
    for ind in range(len(event_data)):
        t0 = date2unix(
            int(event_data['YYYY'].values[ind]), 
            int(event_data['MM'].values[ind]), 
            int(event_data['DD'].values[ind]), 
            int(event_data['hh'].values[ind]), 
            int(event_data['mm'].values[ind]),
            0,
        )
        t0 += event_data['ss.s'].values[ind]
        dts.append(t0)
    event_data['t'] = dts

    return event_data


def load_spade_extended(path):
    names = [f'{x}' for x in range(20)]
    names[0] = 'hit'
    names[1] = 'Y'
    names[2] = 'M'
    names[3] = 'D'
    names[4] = 'h'
    names[5] = 'm'
    names[6] = 's'
    names[7] = 'us'
    
    names[8] = 'r'
    names[10] = 'SNR'
    names[14] = 't'
    names[9] = 'v'

    data = pd.read_csv(
        path, 
        sep=r'[ ]+', 
        comment='%', 
        skip_blank_lines=True, 
        names=names, 
        skiprows=45,
        engine='python',
    )
    data['SNR'] = data['SNR']**2
    for ti in range(len(data['t']) - 1):
        if data['t'].values[ti] > data['t'].values[ti + 1]:
            data['t'].values[(ti + 1):] += data['t'].values[ti]
    data['t'] = (data['t'] - np.min(data['t']))*1e-6
    dts = []
    for ind in range(len(data)):
        t0 = date2unix(
            int(data['Y'].values[ind]), 
            int(data['M'].values[ind]), 
            int(data['D'].values[ind]), 
            int(data['h'].values[ind]), 
            int(data['m'].values[ind]),
            int(data['s'].values[ind]),
        )
        t0 += float(data['us'].values[ind])*1e-6
        dts.append(t0)
    unix0 = Time(dts, format='unix', scale='utc')
    data['unix'] = unix0.unix
    data['r'] = data['r']*1e3

    return data


def read_spade(target_dir, output_h5, SNR_lim=MIN_SNR, verbose=False):

    files = list(target_dir.glob('**/*.txt'))
    files.sort()

    t0s = []
    rgs = []
    vdops = []
    rts = []
    durs = []
    diams = []
    accs = []
    
    data = get_spade_data(files, verbose=verbose)

    if data is None:
        raise ValueError('No valid files found!')

    for ind, row in data.iterrows():

        sn = row['RT']**2.0

        if sn <= SNR_lim:
            if verbose:
                print(f'Skipping row {ind}: rt**2.0 = sn = {sn} <= {SNR_lim}')
            continue
        else:
            if verbose:
                print(f"Adding row {ind}")

        t0 = date2unix(
            int(row['YYYY']), 
            int(row['MM']), 
            int(row['DD']), 
            int(row['hh']), 
            int(row['mm']),
            0,
        )
        t0 += row['ss.s']

        t0s.append(t0)
        rgs.append(row['RG'])
        vdops.append(row['RR'])
        rts.append(row['RT'])
        durs.append(row['ED'])
        diams.append(row['DI'])
        accs.append(row['AD'])

    t = np.array(t0s)
    r = np.array(rgs)
    v = np.array(vdops)
    snr = np.array(rts)**2.0
    dur = np.array(durs)
    diams = np.array(diams)
    accs = np.array(accs)
    
    # store in hdf5 format
    if output_h5 is not None:
        ho = h5py.File(output_h5, "w")
        ho["t"] = t    # t
        ho["r"] = r    # range
        ho["v"] = v    # velocity
        ho["snr"] = snr  # snr
        ho["dur"] = dur  # duration
        ho["diams"] = diams  # minimum diameter
        ho["a"] = accs  # acceleration
        ho.close()

    return t, r, v, snr, dur, diams


def main(input_args=None):
    parser = argparse.ArgumentParser(
        description='Convert EISCAT spade experiment results to a single h5 file',
    )
    parser.add_argument(
        'input_directory', 
        type=str, help='Observation data location',
    )
    parser.add_argument(
        'output_h5', 
        type=str, help='Results output location',
    )
    parser.add_argument(
        '-m', '--min-snr', 
        default=MIN_SNR,
        type=float, help=f'Minimum (linear) SNR of events to keep, defaults to {MIN_SNR}.',
    )

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    output_pth = pathlib.Path(args.output_h5).resolve()
    input_pth = pathlib.Path(args.input_directory).resolve()

    read_spade(input_pth, output_pth, args.min_snr, verbose=True)


if __name__ == '__main__':
    main()
