#!/usr/bin/env python3
"""lists all documents in pymongo"""


def list_all(mongo_collection):
    """
    Lists all documents in the given pymongo collection.

    Parameters:
        mongo_collection: A pymongo collection object.

    Returns:
        A list of all documents in the collection.
        If there are no documents, returns an empty list.
    """
    return list(mongo_collection.find({}))
    