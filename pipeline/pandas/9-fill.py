#!/usr/bin/env python3
"""fills nan columns"""


def fill(df):
    """fills nan columns"""
    df = df.drop("Weighted_Price", axis=1)
    df["Close"] = df['Close'].fillna(method='ffill')  # fills
    df["High"].fillna(df["Close"], inplace=True)
    df["Low"].fillna(df["Close"], inplace=True)
    df["Open"].fillna(df["Close"], inplace=True)
    df[["Volume_(BTC)", "Volume_(Currency)"]] = df[[
        "Volume_(BTC)", "Volume_(Currency)"]].fillna(0)
    return df
