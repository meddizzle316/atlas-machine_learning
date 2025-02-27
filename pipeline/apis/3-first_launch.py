#!/usr/bin/env python3
"""gives first upcoming rocket launches"""
import requests
import datetime


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(url)

    body = response.json()

    sorted_launches = sorted(body, key=lambda x: x["date_unix"])

    first = sorted_launches[0]

    tz = datetime.timezone(datetime.timedelta(hours=-4))
    dt = datetime.datetime.fromtimestamp(first['date_unix'], tz)

    formated_date = dt.isoformat()

    launch_pad = requests.get(
        f"https://api.spacexdata.com/v4/launchpads/{first['launchpad']}")
    # print(launch_pad)
    launch_pad = launch_pad.json()
    launch_pad_name = launch_pad['name']
    launch_pad_local = launch_pad['locality']

    rocket_id = first['rocket']

    rocket = requests.get(f"https://api.spacexdata.com/v4/rockets/{rocket_id}")
    rocket = rocket.json()
    rocket_name = rocket['name']

    print(
        f"{first['name']} "
        f"({formated_date}) "
        f"{rocket_name} - "
        f"{launch_pad_name} "
        f"({launch_pad_local})")
    # print(body)
