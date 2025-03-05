#!/usr/bin/env python3
"""updates all documents in pymongo"""


def update_topics(mongo_collection, name, topics):
    """
    Updates the topics field for all documents in the collection with the given school name.

    Parameters:
        mongo_collection: The pymongo collection object.
        name (str): The name of the school whose topics should be updated.
        topics (list of str): A list of strings representing the new topics for the school.

    Returns:
        The UpdateResult object from pymongo, which contains details about the operation.
    """
    result = mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}}
    )
    return result
