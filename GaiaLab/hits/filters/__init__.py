#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Filters for the datasets. Natively implemented are the Kalman filter and
a single-pole low-pass filter.

There is also a base filter class created for subclassing to allow the
creation of custom filters.
"""
from . import kalman
from . import lowpass
__author__ = "Toby James and Alex Bombrun"
__version__ = "0.1.0"
