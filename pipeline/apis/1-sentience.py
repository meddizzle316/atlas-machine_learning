#!/usr/bin/env python3
"""uses SWAG api to planets of sentient beings"""
import requests


def sentientPlanets():
    """returns names of all home planets
    of sentient species"""

    url = "https://swapi-api.hbtn.io/api/species"
    species = requests.get(url)
    sentient_species = []

    if species.status_code == 200:
        pass
    else:
        print(f"Error: {species.status_code}")

    try:
        data = species.json()
    except ValueError:
        print("Response is not in JSON format")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    # print(json.dumps(data['results'], indent=4))

    for species in data['results']:
        if species['homeworld'] is None:
            # print(f"species: {species['name']} has null homeworld, skipping")
            continue
        if "sentient" in species['classification'] or 'sentient' in species['designation']:
            homeworld_dict = requests.get(species['homeworld']).json()
            homeworld_name = homeworld_dict['name']

            # print(f"species: {species['name']}'s homeworld {homeworld_name} added to sentient list")
            sentient_species.append(homeworld_name)

    return sentient_species
