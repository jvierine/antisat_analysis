import getpass
import argparse
import pathlib
from datetime import datetime

import spacetrack 


parser = argparse.ArgumentParser(description='Download tle snapshot from space-track')
parser.add_argument('start_date', type=str, help='Start date of snapshot [ISO]')
parser.add_argument('end_date', type=str, help='End date of snapshot [ISO]')
parser.add_argument('output', type=str, help='TLE target output file')

args = parser.parse_args()

_fmt = '%Y-%m-%d'

output = pathlib.Path(args.output).resolve()
dt0 = datetime.strptime(args.start_date, _fmt)
dt1 = datetime.strptime(args.end_date, _fmt)

print(f'Getting TLEs for the range [{dt0} -> {dt1}]')

drange = spacetrack.operators.inclusive_range(dt0, dt1)

user = input("Username for space-track.org:")
passwd = getpass.getpass("Password for " + user + ":")

st = spacetrack.SpaceTrackClient(user, passwd)

lines = st.tle_publish(
    iter_lines=True, 
    publish_epoch=drange, 
    orderby='TLE_LINE1', 
    format='tle',
)
lineno = 0
with open(output, 'w') as fp:
    for line in lines:
        fp.write(line + '\n')
        lineno += 1

print(f'Wrote {lineno} lines to {output}')
