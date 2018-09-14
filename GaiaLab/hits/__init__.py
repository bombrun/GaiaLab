#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Micrometeoroid hit detection and simulation programs for Gaia data.

Functions for hit detection and simulation. Detection functions are
packaged in hits.hitdetector. Simulation functions are packaged in
hits.hitsimulator.

hits.filters contains an implementation of the Kalman filter and a
single-pole low-pass filter, as well as a base filter class for
subclassing to create custom filters.

hits.response contains functions for characterising the response of the
satellite to micrometeoroid hits.

For further information on imported functions, run help(function). For
more importable functions available in each package, run help(package).
"""
from . import hitdetector
from . import response
from . import hitsimulator
from . import filters


__all__ = ['filters', 'response', 'hitdetector', 'hitsimulator']
__author__ = "Toby James and Alex Bombrun"
__version__ = "0.1"
