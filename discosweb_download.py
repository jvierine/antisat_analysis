#!/usr/bin/env python
import sys
import re
import getpass
import argparse
import pathlib
import subprocess
from pprint import pprint
import requests

URL = 'https://discosweb.esoc.esa.int'

def main(input_args=None):
    parser = argparse.ArgumentParser(description='Download size information from discosweb')
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('--secret-tool-key', '-k', nargs=1)
    parser.add_argument('--credentials', '-c', nargs=1, help='File containing username and password for space-track.org')
    parser.add_argument('--name', '-n', default=None, help='Name of the object to match with the "like" operator')

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)

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

    response = requests.get(
        f'{URL}/api/objects',
        headers={
            'Authorization': f'Bearer {token}',
            'DiscosWeb-Api-Version': '2',
        },
        params={
            'filter': "eq(objectClass,Payload)&gt(reentry.epoch,epoch:'2020-01-01')",
            'sort': '-reentry.epoch',
        },
    )

    doc = response.json()
    if response.ok:
        pprint(doc['data'])
    else:
        pprint(doc['errors'])


if __name__ == '__main__':
    main()