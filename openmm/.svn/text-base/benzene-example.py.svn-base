#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""

DESCRIPTION

Alchemical free energy calculation driver using soft-core bonds.

COPYRIGHT

@author John D. Chodera <jchodera@gmail.com>

All code in this repository is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

TODO

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

# Python
import os
import os.path
import sys
import math
import copy
import time

# Numerical libraries
import numpy

# OpenMM
import simtk
import simtk.unit as units
import simtk.openmm as openmm

# Replica-exchange molecular dynamics
import repex

#=============================================================================================
# SUBROUTINES
#=============================================================================================

#=============================================================================================
# ALCHEMICAL MODIFICATIONS
#=============================================================================================

def create_softcore_bond(iatom, jatom, r0, K, kT, bond_lambda):
    """
    Create a CustomBondForce object containing a single soft-core bond.

    ARGUMENTS
    
    iatom (int) - first bond atom
    jatom (int) - second bond atom
    r0 (simtk.unit.Quantity with units compatible with simtk.unit.angstrom) - equilibrium bond length
    K (simtk.unit.Quantity with units compatible with simtk.unit.kilocalories_per_mole / simtk.unit.angstrom**2) - equilibrium spring constant
    bond_lambda (float) - lambda value between 0 (dissociated) and 1 (approximates Harmonic bond)

    RETURNS

    force (simtk.openmm.CustomBondForce) - the CustomBondForce object

    """

    # Define a "softcore" bond via a Morse potential.
    # TODO: Later, we may want to interpolate between a harmonic bond when lambda = 1 and a Morse bond when lambda ~ 0.5.
    energy_function = "lambda*D_e*(1 - exp(-a*(r-r0)))^2;" # Morse potential
    energy_function += "a = sqrt((K*lambda) / (2*D_e*lambda));" # compute 'a' parameter to try to keep bond curvature close to expected from harmonic bond
    force = simtk.openmm.CustomBondForce(energy_function)
    force.addPerBondParameter('lambda') # alchemical bond state: 1 is fully made, 0 is broken
    force.addPerBondParameter('D_e') # Morse bond dissociation energy
    force.addPerBondParameter('K') # spring constant for harmonic bond
    force.addPerBondParameter('r0') # equilibrium bond length for harmonic bond

    # Compute Morse parameters to match harmonic equilibrium bond length and spring constant, scaled by lambda value
    D_e = 5.0 * kT

    # Add the Morse bond.
    force.addBond(iatom, jatom, [bond_lambda, D_e, K, r0])

    return force


def create_alchemical_intermediates(reference_system, bond_atoms, bond_lambda, kT, annihilate=False):
    """
    Build alchemically-modified system where ligand is decoupled or annihilated using Custom*Force classes.

    ARGUMENTS

    reference_system (simtk.openmm.System) - reference System object from which alchemical derivatives will be made (will not be modified)
    bond_atoms (list of int) - atoms spanning bond to be eliminated    
    bond_lambda (float) - lambda value for bond breaking (lambda = 1 is original system, lambda = 0 is broken-bond system)
    kT (simtk.unit.Quantity with units compatible with simtk.unit.kilocalories_per_mole) - thermal energy, used in constructing alchemical intermediates

    RETURNS

    system (simtk.openmm.System) - alchemical intermediate copy

    """

    # Create new system.
    system = openmm.System()

    # Set periodic box vectors.
    [a,b,c] = reference_system.getDefaultPeriodicBoxVectors()
    system.setDefaultPeriodicBoxVectors(a,b,c)

    # Add atoms.
    for atom_index in range(reference_system.getNumParticles()):
        mass = reference_system.getParticleMass(atom_index)
        system.addParticle(mass)

    # Add constraints
    for constraint_index in range(reference_system.getNumConstraints()):
        [iatom, jatom, r0] = reference_system.getConstraintParameters(constraint_index)
        # Raise an exception if the specified bond_atoms are part of a constrained bond; we can't handle that.
        if (iatom in bond_atoms) and (jatom in bond_atoms):
            raise Exception("Bond to be broken is part of a constraint.")
        system.addConstraint(iatom, jatom, r0)    

    # Perturb force terms.
    for force_index in range(reference_system.getNumForces()):
        # Dispatch forces based on reference force type.
        reference_force = reference_system.getForce(force_index)
        
        if bond_lambda == 1.0:
            # Just make a copy of the force if lambda = 1.
            force = copy.deepcopy(reference_force)
            system.addForce(force)            
            continue

        if isinstance(reference_force, openmm.HarmonicBondForce):
            force = openmm.HarmonicBondForce()
            for bond_index in range(reference_force.getNumBonds()):
                # Retrieve parameters.
                [iatom, jatom, r0, K] = reference_force.getBondParameters(bond_index)
                if (iatom in bond_atoms) and (jatom in bond_atoms):
                    if bond_lambda == 0.0: continue # eliminate this bond if broken
                    # Replace this bond with a soft-core (Morse) bond.
                    softcore_bond_force = create_softcore_bond(iatom, jatom, r0, K, kT, bond_lambda)
                    system.addForce(softcore_bond_force)
                else:
                    # Add bond parameters.
                    force.addBond(iatom, jatom, r0, K)
                    
            # Add force to new system.
            system.addForce(force)

        elif isinstance(reference_force, openmm.HarmonicAngleForce):
            force = openmm.HarmonicAngleForce()
            for angle_index in range(reference_force.getNumAngles()):
                # Retrieve parameters.
                [iatom, jatom, katom, theta0, Ktheta] = reference_force.getAngleParameters(angle_index)
                # Turn off angle terms that span bond.
                if ((iatom in bond_atoms) and (jatom in bond_atoms)) or ((jatom in bond_atoms) and (katom in bond_atoms)):
                    if bond_lambda == 0.0: continue # eliminate this angle if bond broken
                    Ktheta *= bond_lambda
                # Add parameters.
                force.addAngle(iatom, jatom, katom, theta0, Ktheta)
            # Add force to system.                
            system.addForce(force)

        elif isinstance(reference_force, openmm.PeriodicTorsionForce):
            force = openmm.PeriodicTorsionForce()
            for torsion_index in range(reference_force.getNumTorsions()):
                # Retrieve parmaeters.
                [particle1, particle2, particle3, particle4, periodicity, phase, k] = reference_force.getTorsionParameters(torsion_index)
                # Annihilate if torsion spans bond.
                if ((particle1 in bond_atoms) and (particle2 in bond_atoms)) or ((particle2 in bond_atoms) and (particle3 in bond_atoms)) or ((particle3 in bond_atoms) and (particle4 in bond_atoms)):
                    if bond_lambda == 0.0: continue # eliminate this torsion if bond broken
                    k *= bond_lambda
                # Add parameters.
                force.addTorsion(particle1, particle2, particle3, particle4, periodicity, phase, k)
            # Add force to system.
            system.addForce(force)            

        elif isinstance(reference_force, openmm.NonbondedForce):
            # NonbondedForce will handle charges and exception interactions.
            force = openmm.NonbondedForce()
            for particle_index in range(reference_force.getNumParticles()):
                # Retrieve parameters.
                [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)
                # Lennard-Jones and electrostatic interactions involving atoms in bond will be handled by CustomNonbondedForce except at lambda = 0 or 1.
                if ((bond_lambda > 0) and (bond_lambda < 1)) and (particle_index in bond_atoms):                    
                    # TODO: We have to also add softcore electrostatics.
                    epsilon *= 0.0             
                # Add modified particle parameters.
                force.addParticle(charge, sigma, epsilon)
            for exception_index in range(reference_force.getNumExceptions()):
                # Retrieve parameters.
                [iatom, jatom, chargeprod, sigma, epsilon] = reference_force.getExceptionParameters(exception_index)
                # Modify exception for bond atoms.
                if ((iatom in bond_atoms) and (jatom in bond_atoms)):
                    if (bond_lambda == 0.0): continue # Omit exception if bond has been turned off.
                    # Alchemically modify epsilon and chargeprod.
                    if (iatom in bond_atoms) and (jatom in bond_atoms):
                        # Attenuate exception interaction (since it will be covered by CustomNonbondedForce interactions).
                        epsilon *= bond_lambda
                        chargeprod *= bond_lambda
                    # TODO: Compute restored (1,3) and (1,4) interactions across modified bond.
                # Add modified exception parameters.
                force.addException(iatom, jatom, chargeprod, sigma, epsilon)
            # Set parameters.
            force.setNonbondedMethod( reference_force.getNonbondedMethod() )
            force.setCutoffDistance( reference_force.getCutoffDistance() )
            force.setReactionFieldDielectric( reference_force.getReactionFieldDielectric() )
            force.setEwaldErrorTolerance( reference_force.getEwaldErrorTolerance() )
            # Add force to new system.
            system.addForce(force)

            if (bond_lambda == 0.0) or (bond_lambda == 1.0): continue # don't need softcore if bond is turned off

            # CustomNonbondedForce will handle the softcore interactions with and among alchemically-modified atoms.
            # Softcore potential.
            # TODO: Add coulomb interaction.
            energy_expression = "4*epsilon*compute*x*(x-1.0);"
            energy_expression += "x = 1.0/(alpha*(bond_lambda*(1.0-bond_lambda)/0.25) + (r/sigma)^6);"
            energy_expression += "epsilon = sqrt(epsilon1*epsilon2);"
            energy_expression += "sigma = 0.5*(sigma1 + sigma2);"
            energy_expression += "compute = (1-bond_lambda)*alchemical1*alchemical2 + (alchemical1*(1-alchemical2) + (1-alchemical1)*alchemical2);" # only compute interactions with or between alchemically-modified atoms 

            force = openmm.CustomNonbondedForce(energy_expression)            
            alpha = 0.5 # softcore parameter
            force.addGlobalParameter("alpha", alpha);
            force.addGlobalParameter("bond_lambda", bond_lambda);
            force.addPerParticleParameter("charge")
            force.addPerParticleParameter("sigma")
            force.addPerParticleParameter("epsilon")
            force.addPerParticleParameter("alchemical"); 
            for particle_index in range(reference_force.getNumParticles()):
                # Retrieve parameters.
                [charge, sigma, epsilon] = reference_force.getParticleParameters(particle_index)
                # Alchemically modify parameters.
                if particle_index in bond_atoms:
                    force.addParticle([charge, sigma, epsilon, 1])
                else:
                    force.addParticle([charge, sigma, epsilon, 0])
            for exception_index in range(reference_force.getNumExceptions()):
                # Retrieve parameters.
                [iatom, jatom, chargeprod, sigma, epsilon] = reference_force.getExceptionParameters(exception_index)
                # Exclude exception for bonded atoms.
                if (iatom in bond_atoms) and (jatom in bond_atoms): continue
                # All exceptions are handled by NonbondedForce, so we exclude all these here.
                force.addExclusion(iatom, jatom)
            if reference_force.getNonbondedMethod() in [openmm.NonbondedForce.Ewald, openmm.NonbondedForce.PME]:
                force.setNonbondedMethod( openmm.CustomNonbondedForce.CutoffPeriodic )
            else:
                force.setNonbondedMethod( reference_force.getNonbondedMethod() )
            force.setCutoffDistance( reference_force.getCutoffDistance() )
            system.addForce(force)

        else:
            # Add copy of force term.
            force = copy.deepcopy(reference_force)
            system.addForce(force)            

    return system

#=============================================================================================
# MAIN AND TESTS
#=============================================================================================

if __name__ == "__main__":
    verbose = True

    # PARAMETERS
    base_directory = 'examples/benzene'
    inpcrd_filename = os.path.join(base_directory, 'benzene.crd')
    prmtop_filename = os.path.join(base_directory, 'benzene.prmtop')

    temperature = 300.0 * units.kelvin
    nsteps = 500
    timestep = 1.0 * units.femtoseconds

    # Compute thermal energy.
    kB = units.AVOGADRO_CONSTANT_NA * units.BOLTZMANN_CONSTANT_kB 
    kT = kB * temperature

    # Select platform.
    platform_name = 'Reference'
    platform = openmm.Platform.getPlatformByName(platform_name)

    # Load standard systems.
    import simtk.openmm.app as app
    inpcrd = app.AmberInpcrdFile(inpcrd_filename)
    prmtop = app.AmberPrmtopFile(prmtop_filename)

    # Create System object.
    print "Creating reference System object..."
    reference_system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None, implicitSolvent=app.OBC2)

    # Load coordinates.
    coordinates = inpcrd.getPositions()

    # Define bond atoms to soften.
    bond_atoms = [0, 1] # atoms in bond

    # Construct alchemical systems.
    if verbose: print "Constructing alchemical states..."
    systems = list() # alchemically-modified systems
    bond_lambda = numpy.array([1.00, 0.75, 0.50, 0.15, 0.10, 0.075, 0.06, 0.05, 0.025, 0.00]) # lambda values for tranformation from A into B
    nlambda = len(bond_lambda)
    for lambda_index in range(nlambda):
        # Create alchemically-modified state.
        system = create_alchemical_intermediates(reference_system, bond_atoms, bond_lambda[lambda_index], kT)
        # Append system.
        systems.append(system)

    # Set up reference thermodynamic state.
    import thermodynamics
    reference_state = thermodynamics.ThermodynamicState(systems[0], temperature)

    # Create replica-exchange simulation.
    if verbose: print "Setting up replica-exchange simulation..."
    output_filename = 'repex.nc'
    simulation = repex.HamiltonianExchange(reference_state, systems, [coordinates], output_filename) # initialize the replica-exchange simulation
    simulation.verbose = True
    simulation.number_of_iterations = 1000
    simulation.timestep = timestep
    simulation.nsteps_per_iteration = nsteps
    simulation.minimize = True
    simulation.show_mixing_statistics = True
    simulation.number_of_equilibration_iterations = 0
    simulation.platform = platform
    #simulation.replica_mixing_scheme = 'none' # don't do any swapping
    simulation.replica_mixing_scheme = 'swap-all' # swap normally
    
    if verbose: print "Running..."
    simulation.run() # run the simulation
                                

