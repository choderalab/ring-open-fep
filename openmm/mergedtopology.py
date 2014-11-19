#!/usr/bin/python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Factory for generating alchemically modified 'merged topology' OpenMM System objects for relative
free energy calculations in which one small molecule is transformed into another.

DESCRIPTION

This module provides a factor ```MergedTopologyFactory``` for transforming one small molecule
into another, given a list of pairs of corresponding atoms between the two molecules.

Each molecule is specified by an OpenMM System object, and the alchemical progress coordinate
```alchemical_lambda``` varies from 0 (molecule A) to 1 (molecule B).

Rings may be opened and closed during the transformation.

EXAMPLES

>>> import molecules
>>> molecule_A = molecules.diethylbenzene()
>>> molecule_B = molecules.diethylbenzene()
>>> corresponding_atoms = [(0, 0)]

>>> import mergedtopology
>>> factory = mergedtopology.MergedTopologyFactory(moleculeA, moleculeB)

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

* Can we store serialized form of Force objects so that we can save time in reconstituting
  Force objects when we make copies?  We can even manipulate the XML representation directly.
* Allow protocols to automatically be resized to arbitrary number of states, to 
  allow number of states to be enlarged to be an integral multiple of number of GPUs.
* Add GBVI support to AlchemicalFactory.
* Add analytical dispersion correction to softcore Lennard-Jones, or find some other
  way to deal with it (such as simply omitting it from lambda < 1 states).
* Deep copy Force objects that don't need to be modified instead of using explicit 
  handling routines to copy data.  Eventually replace with removeForce once implemented?
* Can alchemically-modified System objects share unmodified Force objects to avoid overhead
  of duplicating Forces that are not modified?

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import numpy
import copy
import time

#=============================================================================================
# MergedTopologyFactory
#=============================================================================================

class MergedTopologyFactory(object):
    """
    Factory for generating OpenMM System object corresponding to merged toplogy between two molecules.

    EXAMPLES
    
    Create alchemical intermediates for transforming one molecule into another molecule.
    
    >>> # Create a reference system.
    >>> from simtk.pyopenmm.extras import testsystems
    >>> [reference_system, coordinates] = testsystems.WaterBox()
    >>> # Create a factory to produce alchemical intermediates.
    >>> factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=[0, 1, 2])
    >>> # Get the default protocol for 'denihilating' in solvent.
    >>> protocol = factory.defaultSolventProtocolExplicit()
    >>> # Create the perturbed systems using this protocol.
    >>> systems = factory.createPerturbedSystems(protocol)

    """    

    # Factory initialization.
    def __init__(self, system_A, system_B, corresponding_atoms, verbose=False):
        """
        Initialize factory for generating alchemical intermediates between two molecules.

        ARGUMENTS

        system_A (simtk.openmm.System) - System object for molecule A
        system_B (simtk.openmm.System) - System object for molecule B
        corresponding_atoms (list of tuples) - corresponding_atoms[i] is a tuple (i,j) relating atom i of molecule A with atom j of molecule B
        
        """

        # Store deep copies of both molecules.
        self.system_A = copy.deepcopy(system_A)
        self.system_B = copy.deepcopy(system_B)

        # Store copy of atom correspondence list.
        self.corresponding_atoms = copy.deepcopy(corresponding_atoms)
        
        return

    def createMergedTopology(self, alchemical_lambda, mm=None, verbose=False):
        """
        Create a merged topology file with the specified alchemical lambda value for interpolating between molecules A and B.

        ARGUMENTS

        alchemical_lambda (float) - the alchemical lambda in interval [0,1] for interpolating between molecule A (alchemical_lambda = 0) and molecule B (alchemical_lambda = 1),

        OPTIONAL ARGUMENTS

        mm (implements simtk.openmm interface) - OpenMM API implementation to use (default: simtk.openmm)

        TODO

        EXAMPLES

        NOTES

        Merged molecule will contain atom groups in this order:

        S_AB : [atoms in A and B]
        S_A  : [atoms in A and not B]
        S_B  : [atoms in B and not A]

        Masses in S_AB are the geometric product of their masses in A and B.

        """

        # Record timing statistics.
        if verbose: print "Creating merged topology corresponding to alchemical lamdba of %f..." % alchemical_lambda

        # Get local references to systems A and B.
        system_A = self.system_A
        system_B = self.system_B
        corresponding_atoms = self.corresponding_atoms

        #
        # Construct atom sets and correspondence lists for A, B, and merged topology.
        # 

        # Determine number of atoms in each system.
        natoms_A = system_A.getNumParticles()  # number of atoms in molecule A
        natoms_B = system_B.getNumParticles()  # number of atoms in molecule B
        natoms_AandB = len(corresponding_atoms)  # number of atoms in both A and B
        natoms_AnotB = natoms_A - natoms_AandB          # number of atoms in A and not B
        natoms_BnotA = natoms_B - natoms_AandB          # number of atoms in B and not A
        natoms_merged = natoms_AandB + natoms_AnotB + natoms_BnotA  # number of atoms in merged topology

        # Determine sets of atoms shared and not shared.
        atomset_A_AandB = set([ index_A for (index_A, index_B) in corresponding_atoms ]) # atoms in molecule A and B (A molecule numbering)
        atomset_A_AnotB = set(range(natoms_A)) - atomset_A_AandB       # atoms in molecule A and not B (A molecule numbering)
        atomset_A_BnotA = set()

        atomset_B_AandB = set([ index_B for (index_A, index_B) in corresponding_atoms ]) # atoms in molecule B and A (B molecule numbering)
        atomset_B_BnotA = set(range(natoms_B)) - atomset_B_AandB       # atoms in molecule B and not A (B molecule numbering)
        atomset_B_AnotB = set()

        atomset_merged_AandB = set(range(natoms_AandB))                                                               # atoms present in A and B (merged molecule numbering)
        atomset_merged_AnotB = set(range(natoms_AandB, natoms_AandB + natoms_AnotB))                                  # atoms present in A and not B (merged molcule numbering)
        atomset_merged_BnotA = set(range(natoms_AandB + natoms_AnotB, natoms_AandB + natoms_AnotB + natomsBnotA))     # atoms present in B and not A (merged molecule numbering)

        # Construct lists of corresponding atom indices.
        atom_index = dict()

        #
        # Construct merged OpenMM system.
        #

        import simtk.unit as units
        
        # Select OpenMM API implementation to use.
        if not mm:
            import simtk.openmm 
            mm = simtk.openmm

        # Create new System object.
        system = mm.System()

        # Populate merged sytem with atoms.
        # Masses of atoms in both A and B are geometric mean; otherwise standard mass.
        # Add particles in A and B.
        for (index_A, index_B) in corresponding_atoms:
            # Create particle with geometric mean of masses.
            mass_A = system_A.getParticleMass(index_A)
            mass_B = system_B.getParticleMass(index_B)
            mass = units.sqrt(mass_A * mass_B)
            system.addParticle(mass)
        for index_A in atomlist_A_AnotB:
            mass_A = system_A.getParticleMass(index_A)
            system.addParticle(mass_A)
        for index_B in atomlist_B_BnotA:
            mass_B = system_B.getParticleMass(index_B)
            system.addParticle(mass_B)
        
        # Define helper function.
        def find_force(system, classname):
            """
            Find the specified Force object in an OpenMM System by classname.

            ARGUMENTS

            system (simtk.openmm.System) - system containing forces to be searched
            classname (string) - classname of Force object to locate

            RETURNS

            force (simtk.openmm.Force) - the first Force object encountered with the specified classname, or None if one could not be found
            """
            nforces = system.getNumForces()
            force = None
            for index in range(nforces):
                if isinstance(system.getForce(index), getattr(mm, classname)):
                    force = system.getForce(index)
            return force

        # Add bonds.
        # NOTE: This does not currently deal with bonds that are broken or formed during the transformation.
        force_A = find_force(system_A, 'HarmonicBondForce')             
        for index in range(force_A.getNumBonds()):            
            # Get bond parameters from molecule A.
            [iatom_A, jatom_A, length_A, k_A] = force.getBondParameters(index)
            # Translate atom indices to merged atom indices.
            (iatom_merged, jatom_merged) = (atom_indices['A'][iatom_A]['merged'], atom_indices['B'][jatom_A]['merged']) 
            # Store bond parameters for random access.
            bonds[(iatom_merged, jatom_merged)] = (length-A, k_A)
        force_B = find_force(system_B, 'HarmonicBondForce')             
        for index in range(force_B.getNumBonds()):            
            # Get bond parameters from molecule A.
            [iatom_B, jatom_B, length_B, k_B] = force.getBondParameters(index)
            # Translate atom indices to merged atom indices.
            (iatom_merged, jatom_merged) = (atom_indices['B'][iatom_B]['merged'], atom_indices['B'][jatom_B]['merged']) 
            # Store bond parameters for random access.
            if (iatom_merged, jatom_merged) in bonds:
                # Mix bonds.
                (length_A, k_A) = bonds[(iatom_merged, jatom_merged)]
                (length, k) = ( (1.0-alchemical_lambda)*length_A + alchemical_lambda*length_B, (1.0-alchemical_lambda)*k_A + alchemical_lambda*k_B )
                bonds[(iatom_merged, jatom_merged)] = (length, k)
            else:
                bonds[(iatom_merged, jatom_merged)] = (length_B, k_B)
        # Add bonds to merged topology.
        force = mm.HarmonicBondForce()
        for (iatom, jatom) in bonds:
            # Retrieve bond parameters.
            (length, j) = bonds[(iatom, jatom)]
            # Add bond.
            force.addBond(iatom, jatom, length, k)
        # Add the Force to the merged topology sytem.
        system.addForce(force)
                


    return

if __name__ == "__main__":
    # Run doctests.
    import doctest
    doctest.testmod()


