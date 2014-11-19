#!/usr/bin/env python
"""
Example illustrating a free energy calculation where a pair of ions fixed in 
space is grown in a box of TIP3P waters using OpenMM and the 'alchemy' module.

AUTHOR

John D. Chodera <jchodera@berkeley.edu>
14 Sep 2012

DESCRIPTION

In this example, we create a System object containing a pair of ions fixed
in space and solvate it in a box of TIP3P water.

We then create a series alchemically-modified  System objects in which the 
electrostatics and Lennard-Jones interations for one water molecule are 
annihilated.

Finally, we run a molecular dynamics simulation for each alchemically-modified 
state, writing the data out to a NetCDF4 file for analysis.

The corresponding 'analyze.py' script is used to analyze the NetCDF4 file and 
compute the free energy of this transformation.

PREREQUISITES

* OpenMM 
http://simtk.org/home/openmm

* The 'alchemy.py' module, included with this example.

* NumPy
http://numpy.scipy.org/

* netcdf4-python and NetCDF4
http://code.google.com/p/netcdf4-python/
http://www.unidata.ucar.edu/downloads/netcdf/netcdf-4_0/index.jsp

Note that the Enthought Python Distribution (EPD) is a convenient way to install
NumPy, NetCDF4, netcdf4-python, and many more packages useful for scientific computing:

http://www.enthought.com/products/epd.php

"""

#==============================================================================
# IMPORTS
#==============================================================================

# Import OpenMM and units packages.
import simtk.openmm as openmm
import simtk.unit as units

# Import factory to generate alchemically modified System objects.
import alchemy

# We need NumPy and NetCDF for writing out data in a platform-portable format.
import numpy 
import netCDF4 as netcdf 

#==============================================================================
# PARAMETERS
#==============================================================================

temperature = 298.0 * units.kelvin # simulation temperature
collision_rate = 9.1 / units.picosecond # collision rate for Langevin integrator
pressure = 1.0 * units.atmospheres # simulation pressure
barostat_frequency = 25 # number of steps between Monte Carlo barostat volume updates
timestep = 2.0 * units.femtoseconds # integration timestep
niterations = 5 # number of samples to collect from each state
nsteps = 50 # number of MD timesteps per sample
filename = 'output.nc' # name of NetCDF4 file for output

# Alchemical 'lambda' values for generating intermediate states.
# lambda = 1 means the system is fully interacting; lambda = 0 means noninteracting
vdw_lambdas = [0.0, 0.25, 0.5, 0.75, 1.0] # alchemical 'lambda' values for turning on soft-core Lennard-Jones interactions
coulomb_lambdas = [0.0, 0.05, 0.25, 0.5, 0.75, 1.0] # alchemical 'lambda' values for charging

kT = temperature * units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA

#==============================================================================
# Use OpenMM application layer to read in ion pair.
#==============================================================================

# Import the OpenMM application layer.
import simtk.openmm.app as app

# Read a forcefield with Lennard-Jones and solvent parameters.
# This is cheating a bit, since your complexes won't have parameters defined in amber99sb.xml, 
# but we can work around this when the time comes with a specially-constructed XML file or
# having you construct the system manually.
forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml')

# Load in the atoms we'll be fixing in space and charging up.
pdbfile = app.PDBFile('NaCl.pdb')

# Determine number of fixed ions.
fixed_particle_positions = pdbfile.getPositions(asNumpy=True)
nfixed = fixed_particle_positions.shape[0]

# Solvate the structure.
modeller = app.Modeller(pdbfile.topology, pdbfile.positions)
modeller.addSolvent(forcefield, model='tip3p', padding=1.0*units.nanometer)

# Create OpenMM System object.
settings = { 'nonbondedMethod' : app.PME, 'constraints' : app.HBonds, 'nonbondedCutoff' : 0.9*units.nanometer, 'vdwCutoff' : 0.9*units.nanometer, 'useDispersionCorrection' : True, 'rigidWater' : True }
reference_system = forcefield.createSystem(modeller.topology, **settings)

# Add a barostat.
barostat = openmm.MonteCarloBarostat(pressure, temperature, barostat_frequency)
reference_system.addForce(barostat)

# Fix the particle positions for the ions by setting masses to zero.
for index in range(nfixed):
    reference_system.setParticleMass(index, 0.0 * units.amu)

# Get initial positions.
positions = modeller.getPositions()

#==============================================================================
# Create alchemical intermediates.
#==============================================================================

from alchemy import AlchemicalState, AbsoluteAlchemicalFactory

print "Creating alchemical intermediates..."
alchemical_atoms = range(nfixed) # atoms to be alchemically modified

alchemical_states = list() # alchemical_states[istate] is the alchemical state lambda specification for alchemical state 'istate'

# Create alchemical states where we turn on Lennard-Jones (via softcore) with zero charge.
for vdw_lambda in vdw_lambdas:
    alchemical_states.append( AlchemicalState(coulomb_lambda=0.0, vdw_lambda=vdw_lambda, annihilate_coulomb=True, annihilate_vdw=True) )

# Create alchemical states where we turn on charges with full Lennard-Jones.
for coulomb_lambda in coulomb_lambdas:
    alchemical_states.append( AlchemicalState(coulomb_lambda=coulomb_lambda, vdw_lambda=1.0, annihilate_coulomb=True, annihilate_vdw=True) )

alchemical_factory = AbsoluteAlchemicalFactory(reference_system, alchemical_atoms=alchemical_atoms)
systems = alchemical_factory.createPerturbedSystems(alchemical_states) # systems[istate] will be the System object corresponding to alchemical intermediate state index 'istate'
nstates = len(systems)

#==============================================================================
# Run simulation.
#==============================================================================

# Initialize NetCDF file to store data.
import netCDF4 as netcdf 
ncfile = netcdf.Dataset(filename, 'w', version='NETCDF4')
ncfile.createDimension('iteration', 0) # unlimited number of iterations
ncfile.createDimension('state', nstates) # number of replicas
ncfile.createDimension('atom', reference_system.getNumParticles()) # number of atoms in system
ncfile.createDimension('spatial', 3) # number of spatial dimensions
ncfile.createVariable('positions', 'f', ('iteration','state','atom','spatial')) # positions (in A)
ncfile.createVariable('box_vectors', 'f', ('iteration','state','spatial','spatial')) # box vectors (in A)
ncfile.createVariable('energies', 'f', ('iteration','state','state')) # reduced potential energies (in kT)

# Run simulation at each intermediate.
for istate in range(nstates):
    print "Simulating state %d / %d..." % (istate, nstates)

    # Select system corresponding to this alchemical state.
    system = systems[istate]

    # Initialize integrator and context.
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)

    # Minimize energy.
    print "Minimizing energy..."
    tolerance = 10.0 * units.kilojoules_per_mole / units.nanometers
    maxIterations = 20
    openmm.LocalEnergyMinimizer.minimize(context, tolerance, maxIterations)

    for iteration in range(niterations):
        print "  iteration %d / %d" % (iteration, niterations)

        # Run dynamics.
        integrator.step(nsteps)
        
        # Get current configuration.
        state = context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)

        # Store positions and box vectors.
        ncfile.variables['positions'][iteration,istate,:,:] = positions[:,:] / units.angstroms
        ncfile.variables['box_vectors'][iteration,istate,:,:] = box_vectors[:,:] / units.angstroms

        # Report potential energy.
        state = context.getState(getEnergy=True)
        print "    %8.3f kT" % (state.getPotentialEnergy() / kT)

    # Sync up NetCDF file.
    ncfile.sync()
        
    # Clean up.
    del context, integrator

# Compute energies of all snapshots at all states.
for jstate in range(nstates):
    print "Computing energies for state %d / %d..." % (jstate, nstates)

    # Select system corresponding to this alchemical state.
    system = systems[jstate]

    # Initialize integrator and context.
    integrator = openmm.VerletIntegrator(timestep)
    context = openmm.Context(system, integrator)

    for iteration in range(niterations):
        print "  iteration %d / %d" % (iteration, niterations)

        for istate in range(nstates):
            # Set configuration.
            positions = units.Quantity(ncfile.variables['positions'][iteration,istate,:,:], units.angstroms)
            box_vectors = units.Quantity(ncfile.variables['box_vectors'][iteration,istate,:,:], units.angstroms)
            context.setPositions(positions)
            context.setPeriodicBoxVectors(box_vectors[0,:], box_vectors[1,:], box_vectors[2,:])

            # Get energy.
            state = context.getState(getEnergy=True)
            ncfile.variables['energies'][iteration,istate,jstate] = state.getPotentialEnergy() / kT

    # Sync up NetCDF file.
    ncfile.sync()
        
    # Clean up.
    del context, integrator

# Clean up.
ncfile.close()
