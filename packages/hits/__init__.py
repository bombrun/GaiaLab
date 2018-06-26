#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hit detection and simulation programs for Gaia data.

Functions for hit detection and simulation.
Detection functions are packaged in hits.hitDetector.
Simulation functions are packaged in hits.hitSimulator.

Included functions with this import are generateData()
from hits.hitSimulator and plotAnomaly() from hits.hitDetector.

A linear array of appropriate masses is also imported as 
hits.masses.

For further information on imported functions,
run help(function). For more importable functions 
packaged in each module, run help(hits.module).
"""

from hits.hitDetector import plotAnomaly
from hits.hitSimulator import generateData, masses

__author__ = "Toby James and Alex Bombrun"
__version__ = "0.1"
