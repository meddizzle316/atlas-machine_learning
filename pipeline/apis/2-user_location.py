#!/usr/bin/env python3
import requests
import sys
import datetime


if __name__ == '__main__':
    user_url = None

    user_url = "".join(sys.argv[1:])

    if user_url is None:
        raise ValueError("no user_url")
    response = requests.get(user_url)
    reset_timestamp = response.headers.get('X-RateLimit-Reset')

    reset_time = datetime.datetime.fromtimestamp(int(reset_timestamp), datetime.UTC)

    now = datetime.datetime.now(datetime.timezone.utc)

    time_diff = reset_time - now

    seconds = time_diff.seconds

    minutes  = divmod(seconds, 60)[0]

    if response.status_code == 200:
        pass
    elif response.status_code == 403:
        print(f"Reset in {minutes} min")
    else:
        print("Not found")
        sys.exit()

    json_body = response.json()
    print(json_body['location'])
