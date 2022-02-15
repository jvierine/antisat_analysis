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
        '-t', '--type', type=str, default='NORAD',
        choices = ['NORAD', 'COSPAR', 'DISCOS'],
        help='Input ID, ID variant can be set with "--type"',
    )
    parser.add_argument(
        '-o', '--output', 
        nargs='?', type=argparse.FileType('w'), default=sys.stdout,
        help = 'Path to file where output should be written.'
    )
    parser.add_argument(
        '--secret-tool-key', '-k', nargs=1, type=str, metavar='KEY',
        help='Attribute value (key) to fetch secret from',
    )
    parser.add_argument(
        '--credentials', '-c', nargs=1, metavar='FILE',
        help='File containing DISCOSweb token',
    )
    parser.add_argument(
        'object_id', metavar='ID', type=str, nargs='+',
        help='Input ID(s), ID variant can be set with "--type"',
    )

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

    if args.type == 'NORAD':
        tmp_oids = []
        for oid in args.object_id:
            oid = oid[:-1] if oid[-1].isalpha() else oid
            tmp_oids.append(oid)
        args.object_id = tmp_oids

    elif args.type == 'COSPAR':
        args.object_id = [f"'{oid}'" for oid in args.object_id]
    
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

    oids = ','.join(args.object_id)
    if len(args.object_id) == 1:
        oids += ','

    filt_str = f"in({_ID_MAP[args.type]},({oids}))"
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

    doc = response.json()
    if response.ok:
        print(f'{len(doc)} Entries found...')
        json.dump(doc['data'], args.output, indent=2)
    else:
        print('Error...')
        pprint(doc['errors'])


if __name__ == '__main__':
    main()
