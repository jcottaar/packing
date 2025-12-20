#!/usr/bin/env python
"""
Wrapper script to run pack_test.py without displaying figures.
Uses matplotlib's Agg backend to prevent GUI windows from appearing.
"""

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot

import pack_test

if __name__ == "__main__":
    pack_test.run_all_tests()
