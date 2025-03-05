#!/usr/bin/env python3
"""Returns a list of school documents that have the specified topic."""


def schools_by_topic(mongo_collection, topic):
    """
    Returns a list of school documents that have the specified topic.

    Parameters:
        mongo_collection: The pymongo collection object.
        topic (str): The topic to search for in the school's topics field.

    Returns:
        A list of documents (dicts) representing schools that have the specified topic.
        If no document is found, returns an empty list.
    """
    # Query to find documents where the "topics" array contains the provided topic.
    cursor = mongo_collection.find({"topics": topic})
    return list(cursor)
