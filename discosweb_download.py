#!/usr/bin/env python
import sys
import getpass
import argparse
import json
import subprocess
import requests
from pprint import pprint

URL = 'https://discosweb.esoc.esa.int'

_ID_MAP = {
    'NORAD': 'satno',
    'COSPAR': 'cosparId',
    'DISCOS': 'id',
}


def main(input_args=None):
    parser = argparse.ArgumentParser(
        description='Download size information from discosweb',
    )
    parser.add_argument(
        'object_id', metavar='ID', type=str, 
        help='Input ID, ID variant can be set with "--type"',
    )
    parser.add_argument(
        '-t', '--type', type=str, default='NORAD',
        choices = ['NORAD', 'COSPAR', 'DISCOS'],
        help='Input ID, ID variant can be set with "--type"',
    )
    parser.add_argument(
        'output', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
        help = 'Path to file where output should be written.'
    )
    parser.add_argument('--secret-tool-key', '-k', nargs=1)
    parser.add_argument(
        '--credentials', '-c', nargs=1, 
        help='File containing DISCOSweb token',
    )
    parser.add_argument(
        '--name', '-n', default=None, 
        help='Name of the object to match with the "like" operator',
    )

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    if args.type == 'NORAD':
        if args.object_id[-1].isalpha():
            args.object_id = args.object_id[:-1]
    elif args.type == 'COSPAR':
        args.object_id = f"'{args.object_id}'"
    
    if args.secret_tool_key is not None:
        res = subprocess.run(
            ['secret-tool', 'lookup', 'token'] + args.secret_tool_key, 
            capture_output=True, text=True,
        )
        token = res.stdout
    elif args.credentials is not None:
        raise NotImplementedError('Add input of username/password from file')
    else:
        token = getpass.getpass("API token for:")

    filt_str = f"{_ID_MAP[args.type]}={args.object_id}"
    print(f'Fetching data for "{filt_str}"')

    response = requests.get(
        f'{URL}/api/objects',
        headers={
            'Authorization': f'Bearer {token}',
            'DiscosWeb-Api-Version': '2',
        },
        params={
            'filter': filt_str,
        },
    )

    # Mass
    # 3964.32 kg
    # Shape
    # Cyl
    # Width
    # –
    # Height
    # 28 m
    # Depth
    # –
    # Diameter
    # 2.6 m
    # Span
    # 28 m
    # Max. cross section
    # 72.9933461154505 m²
    # Min. cross section
    # 5.30929158456675 m²
    # Avg. cross section
    # 59.8316320876176 m²

    doc = response.json()
    if response.ok:
        print(f'{len(doc)} Entries found...')
        json.dump(doc['data'], args.output, indent=2)
    else:
        print('Error...')
        pprint(doc['errors'])


if __name__ == '__main__':
    main()
