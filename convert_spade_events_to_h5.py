import pathlib
import datetime
import argparse

import numpy as np
import pandas as pd
import h5py

'''Example usage:

python convert_spade_events_to_h5.py ./leo_bpark_2.1u_CN@uhf/ ./h5/leo.h5
'''


def date2unix(year, month, day, hour, minute, second):
    dt = datetime.datetime(year, month, day, hour, minute, second)
    timestamp = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return timestamp


def read_spade(target_dir, output_h5):

    files = list(target_dir.glob('**/*.txt'))
    files.sort()

    t0s = []
    rgs = []
    vdops = []
    rts = []
    durs = []
    diams = []
    
    names = [
        'YYYY', 'MM', 'DD', 
        'hh', 'mm', 'ss.s', 
        'TX', 'AZ', 'EL', 
        'RT', 'RG', 'RR', 
        'VD', 'AD', 'DI', 
        'CS', 'TS', 'EN', 
        'ED', 'TP', 'MT',
    ]

    data = None
    for file in files:
        next_data = pd.read_csv(file, sep=r'[ ]+', comment='%', skip_blank_lines=True, names=names)
        if data is None:
            data = next_data
        else:
            data.append(next_data)

    for ind, row in data.iterrows():

        sn = row['RT']**2.0

        if sn <= 33:
            print(f'Skipping row {ind}: rt**2.0 = sn = {sn} <= 33')
            continue
        else:
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

    t = np.array(t0s)
    r = np.array(rgs)
    v = np.array(vdops)
    snr = np.array(rts)**2.0
    dur = np.array(durs)
    diams = np.array(diams)
    
    # store in hdf5 format
    ho = h5py.File(output_h5, "w")
    ho["t"] = t    # t
    ho["r"] = r    # r  
    ho["v"] = v    # vel
    ho["snr"] = snr  # snr
    ho["dur"] = dur  # duration
    ho["diams"] = diams  # minimum diameter
    ho.close()

    return t, r, v, snr, dur, diams


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Convert EISCAT spade experiment results to a single h5 file',
    )
    parser.add_argument('input_directory', type=str, help='Observation data location')
    parser.add_argument('output_h5', type=str, help='Results output location')

    args = parser.parse_args()

    output_pth = pathlib.Path(args.output_h5).resolve()
    input_pth = pathlib.Path(args.input_directory).resolve()

    read_spade(input_pth, output_pth)
