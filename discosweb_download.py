#!/usr/bin/env python
import sys
import time
import getpass
import argparse
import json
import subprocess
import requests

URL = "https://discosweb.esoc.esa.int"

_ID_MAP = {
    "NORAD": "satno",
    "COSPAR": "cosparId",
    "DISCOS": "id",
}


def main(input_args=None):
    parser = argparse.ArgumentParser(
        description="Download size information from discosweb",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="NORAD",
        choices=["NORAD", "COSPAR", "DISCOS"],
        help='Input ID, ID variant can be set with "--type"',
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Path to file where output should be written.",
    )
    parser.add_argument(
        "--secret-tool-key",
        "-k",
        nargs=1,
        type=str,
        metavar="KEY",
        help='Attribute [named "token"] value [key] to fetch secret from',
    )
    parser.add_argument(
        "--credentials",
        "-c",
        nargs=1,
        metavar="FILE",
        help="File containing DISCOSweb token",
    )
    parser.add_argument(
        "object_id",
        metavar="ID",
        type=str,
        nargs="+",
        help='Input ID(s), ID variant can be set with "--type"',
    )

    if input_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input_args)
    if isinstance(args.object_id, str):
        args.object_id = [args.object_id]

    if args.type == "NORAD":
        tmp_oids = []
        for oid in args.object_id:
            oid = oid[:-1] if oid[-1].isalpha() else oid
            tmp_oids.append(oid)
        args.object_id = tmp_oids

    elif args.type == "COSPAR":
        args.object_id = [f"'{oid}'" for oid in args.object_id]

    if args.secret_tool_key is not None:
        res = subprocess.run(
            ["secret-tool", "lookup", "token"] + args.secret_tool_key,
            capture_output=True,
            text=True,
        )
        token = res.stdout
    elif args.credentials is not None:
        raise NotImplementedError("Add input of username/password from file")
    else:
        token = getpass.getpass("API token for:")

    current_page = 1
    params = {"sort": "id"}
    if len(args.object_id) > 0:
        oids = ",".join(args.object_id)
        if len(args.object_id) == 1:
            filt_str = f"{_ID_MAP[args.type]}={oids}"
        else:
            filt_str = f"in({_ID_MAP[args.type]},({oids}))"
        params["filter"] = filt_str
        print(f'Fetching data for "{filt_str}"')
    else:
        print("Fetching all data")

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
        print(f"{len(doc['data'])} Entries found...")
        json.dump(doc['data'], args.output, indent=2)
    else:
        print('Error...')
        print(doc['errors'])

    # session = requests.Session()
    # req = requests.Request(
    #     "GET",
    #     f"{URL}/api/objects",
    #     headers={
    #         "Authorization": f"Bearer {token}",
    #         "DiscosWeb-Api-Version": "2",
    #     },
    #     params=params,
    # )
    # objects = []
    # while True:
    #     print("getting page ", current_page)
    #     prepped = session.prepare_request(req)
    #     connector = "?" if len(params) == 0 else "&"
    #     prepped.url += f"{connector}page[number]={current_page}&page[size]=100"

    #     response = session.send(prepped)

    #     if response.status_code == 429:
    #         retry_interval = int(response.headers["Retry-After"])
    #         print(f"API requests exceeded, sleeping for {retry_interval} s")
    #         time.sleep(retry_interval + 1)
    #         continue
    #     else:
    #         current_page += 1

    #     if response.status_code != 200:
    #         response.raise_for_status()

    #     result = response.json()
    #     for obj in result["data"]:
    #         objects.append(obj)

    #     if "next" not in result["links"]:
    #         break


    # # print object data
    # print(
    #     "SATNO  COSPAR ID    DISCOS ID  Name                       "
    #     "Object class  Mass     Shape                     Length  "
    #     "Height  Depth   Min Xsect  Avg Xsect  Max Xsect  "
    #     "Launch Date  Re-entry Date  Country"
    # )
    # for object_ in objects:
    #     if object_["reentryEpoch"] is None:
    #         object_["reentryEpoch"] = "-"
    #     print(
    #         "{satno:5d}  {cosparId:11s}  {discosId:9d}  {name:25s}  "
    #         "{objectClass:12s}  {mass:7.1f}  {shape:24s}  "
    #         "{length:6.1f}  {height:6.1f}  {depth:6.1f}  "
    #         "{xSectMin:9.1f}  {xSectAvg:9.1f}  {xSectMax:9.1f}  "
    #         "{launchDate:11s}  {reentryEpoch:13s}  {country}".format(**object_)
    #     )

    # print(f"{len(objects)} Entries found...")
    # json.dump(objects, args.output, indent=2)


if __name__ == "__main__":
    main()
