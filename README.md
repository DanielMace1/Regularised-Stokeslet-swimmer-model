# Regularised-Stokeslet-swimmer-model
Regularised Stokeslet implementation for sperm-like microswimmer, produced for MMath project at Durham University.

## Overview

The code implements a free-swimming model of a microorganism with a rigid prolate spheroid head, with a deformable flagellum driven by a prescribed waveform. There is an option of including multiple swimmers.

## Contents
- Waveform classes for sinusoidal, helical and elliptical flagellar beats, with an arclength cutoff imposed to ensure a constant flagellum length.
- A prolate spheroid head component with Fibonacci-lattice distribution and weighting.
- Composite swimmer class combining the head and flagellum
- Regularised Stokeslet solver using Cortez(2005) blob function, imposing force and torque free conditions
- Time integration via Euler translation and Rodrigues rotation

## Requirements
Python with numpy and matplotlib

## Usage
'run_simulation' function takes a list of 'CompositeSwimmer' instances and evolves them through time.
