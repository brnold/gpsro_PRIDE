# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 17:18:49 2025

@author: Benjamin Nold
"""
import datetime
import numpy as np
from typing import Union

def gps_to_utc(
    week: Union[int, np.ndarray],
    tow: Union[float, np.ndarray],
    leap_seconds: Union[int, np.ndarray]
) -> Union[datetime.datetime, np.ndarray]:
    """
    Converts GPS week and time-of-week to UTC datetime(s), applying leap seconds.
    - Returns a datetime.datetime for scalar input.
    - Returns a NumPy array of datetime.datetime for vector input.
    """
    gps_epoch = datetime.datetime(1980, 1, 6)

    if isinstance(week, (int, float)) and isinstance(tow, (int, float)) and isinstance(leap_seconds, (int, float)):
        gps_time = gps_epoch + datetime.timedelta(weeks=week, seconds=tow)
        return gps_time - datetime.timedelta(seconds=leap_seconds)

    # Convert to arrays in case user passes lists
    week = np.asarray(week)
    tow = np.asarray(tow)
    leap_seconds = np.asarray(leap_seconds)

    return np.array([
        gps_epoch + datetime.timedelta(weeks=int(w), seconds=float(t)) - datetime.timedelta(seconds=float(ls))
        for w, t, ls in zip(week, tow, leap_seconds)
    ])


def gps_week_tow_gps_dt(
    week: Union[int, np.ndarray],
    tow: Union[float, np.ndarray]
) -> Union[datetime.datetime, np.ndarray]:
    """
    Convert GPS week and time-of-week to datetime(s) since GPS epoch.
    - Returns a datetime.datetime for scalar input.
    - Returns a NumPy array of datetime.datetime for vector input.
    """
    gps_epoch = datetime.datetime(1980, 1, 6)

    if isinstance(week, (int, float)) and isinstance(tow, (int, float)):
        return gps_epoch + datetime.timedelta(weeks=week, seconds=tow)

    # Convert to arrays (in case someone passes lists)
    week = np.asarray(week)
    tow = np.asarray(tow)

    # Vectorized datetime generation using list comprehension
    return np.array([
        gps_epoch + datetime.timedelta(weeks=int(w), seconds=float(t))
        for w, t in zip(week, tow)
    ])

def datetime_to_gps_seconds(dt: datetime.datetime) -> float:
    gps_epoch = datetime.datetime(1980, 1, 6, tzinfo=dt.tzinfo)
    return (dt - gps_epoch).total_seconds()

