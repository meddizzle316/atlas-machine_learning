#!/usr/bin/env python3
"""lists all documents in pymongo"""


def insert_school(mongo_collection, **kwargs):
    """
    Inserts a new document into the provided pymongo collection using keyword arguments.
    
    Parameters:
        mongo_collection: The pymongo collection object.
        **kwargs: Arbitrary keyword arguments that form the document to be inserted.
        
    Returns:
        The _id of the newly inserted document.
    """
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
    