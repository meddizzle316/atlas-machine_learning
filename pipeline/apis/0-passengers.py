#!/usr/bin/env python3
"""gets ships that have passenger count"""
import requests


def availableShips(passengerCount):
    """returns ships that can hold passenger count"""
    url = "https://swapi-api.hbtn.io/api/starships"
    starships = requests.get(url)
    starships_with_passengerCount = []

    if starships.status_code == 200:
        pass
    else:
        print(f"Error: {starships.status_code}")

    try:
        data = starships.json()
    except ValueError:
        print("Response is not in JSON format")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    # print(json.dumps(data['results'], indent=4))

    for ship in data['results']:
        try:
            passengers = int(ship['passengers'].replace(",", ""))
            if passengers >= int(passengerCount):
                starships_with_passengerCount.append(ship['name'])
        except ValueError:
            continue

    return starships_with_passengerCount
