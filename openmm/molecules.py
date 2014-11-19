
#!/usr/bin/python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Tools for generating small molecules.

DESCRIPTION

This module provides a set of tools for generating small molecules from IUPAC names.

DEPENDENCIES

* OpenEye OEChem toolkit
http://www.eyesopen.com/oechem-tk

* AmberTools
http://ambermd.org/#AmberTools

EXAMPLES

>>> import molecules
>>> # Create a molecule by name.
>>> molecule = molecules.createMoleculeByName('1,2-diethylbenzene')
>>> # Create OpenMM System using GAFF.
>>> [system, positions] = molecules.createSystem(molecule, forcefield='GAFF', charge_model='am1bcc')


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

# Import OpenEye tools on module import.
# TODO: Use lazy imports instead, since each method may only require a subset of tools?
import openeye.oechem
import openeye.oeiupac
import openeye.oeomega
import openeye.oequacpac
import openeye.oeszybki
import openeye.oeshape

#=============================================================================================
# METHODS FOR CREATING OR IMPORTING MOLECULES
#=============================================================================================

def createMoleculeByName(name, verbose=False, formal_charge=None, strict_typing=None):
   """
   Generate a small molecule by common or IUPAC name.

   ARGUMENTS

     name (string) - common or IUPAC name of molecule to generate

   OPTIONAL ARGUMENTS
     verbose (boolean) - if True, subprocess output is shown (default: False)
     formal_charge (int) - if specified, a form of this molecule with the desired formal charge state will be produced (default: None)
     strict_typing (boolean) - if set, passes specified value to omega (see documentation for expandConformations)

   RETURNS
     molecule (OEMol) - the molecule created

   NOTES
     OpenEye LexiChem's OEParseIUPACName is used to generate the molecle.
     The molecule is normalized by adding hydrogens.
     Omega is used to generate a single conformation.
     Also note that atom names will be blank coming from this molecule. They are assigned when the molecule is written, or one can assign using OETriposAtomNames for example.

   EXAMPLES

   Create a molecule of phenol.
   >>> molecule = createMoleculeByName('phenol')

   Create several small molecules.
   >>> molecule_names = ['benzene', 'toluene', 'ethanol', '1,2-dichlorobenzene']
   >>> molecules = [ createMoleculeByName(molecule_name) for molecule_name in molecule_names ]
     
   """

   # Create an empty molecule.
   molecule = openeye.oechem.OEMol() 

   # Create molecule topolgoy from name.
   # TODO: In future, try to retrieve the molecule from some sort of database if we can't build it this way.
   status = openeye.oeiupac.OEParseIUPACName(molecule, name) # create molecule topology from name

   # Normalize the molecule.
   normalizeMolecule(molecule)

   # Generate a conformation with Omega.
   molecule = expandConformations(molecule, maxconfs=1, include_original=False, verbose=verbose, strict_typing=strict_typing)

   # Select desired protonation state, if specified.
   if (formal_charge != None):
      molecule = selectProtonationState(molecule, formal_charge, verbose=verbose)
    
   # Return the molecule.
   return molecule

#=============================================================================================

def readMolecule(filename, normalize=False):
   """Read in a molecule from a file (such as .mol2).

   ARGUMENTS
     filename (string) - the name of the file containing a molecule, in a format that OpenEye autodetects (such as .mol2, .sdf)

   OPTIONAL ARGUMENTS
     normalize (boolean) - if True, molecule is normalized (renamed, aromaticity, protonated) after reading (default: True)

   RETURNS
     molecule (OEMol) - OEMol representation of molecule


   EXAMPLES

   Create a test molecule
   >>> molecule = createMoleculeByName('benzene')

   Write it to disk.
   >>> import tempfile
   >>> filename = tempfile.mktemp(suffix='.mol2')
   >>> writeMolecule(molecule, filename)

   Read it again.
   >>> molecule = readMolecule(filename)

   """

   # Open input stream.
   istream = openeye.oechem.oemolistream()
   istream.open(filename)

   # Create molecule.
   molecule = openeye.oechem.OEMol()   

   # Read the molecule.
   openeye.oechem.OEReadMolecule(istream, molecule)

   # Close the stream.
   istream.close()

   # Normalize molecule if desired.
   if normalize: 
      normalizeMolecule(molecule)

   return molecule

#=============================================================================================
# METHODS FOR INTERROGATING MOLECULES
#=============================================================================================

def formalCharge(molecule):
   """
   Report the net formal charge of a molecule.

   ARGUMENTS
     molecule (OEMol) - the molecule whose formal charge is to be determined

   RETURN VALUES
     formal_charge (integer) - the net formal charge of the molecule

   EXAMPLE
   
   Create a molecule.   
   >>> molecule = createMoleculeByName('benzene')

   Get its formal charge.
   >>> formal_charge = formalCharge(molecule)
     
   """

   # Create a copy of the molecule.
   molecule_copy = openeye.oechem.OEMol(molecule)

   # Assign formal charges.
   openeye.oechem.OEFormalPartialCharges(molecule_copy)

   # Compute net charge (floating point).
   net_charge = openeye.oechem.OENetCharge(molecule_copy)

   # Compute net formal charge.
   formal_charge = int(round(net_charge))

   # return formal charge
   return formal_charge

#=============================================================================================
# METHODS FOR MODIFYING MOLECULES
#=============================================================================================

def normalizeMolecule(molecule):
   """
   Normalize the molecule by checking aromaticity, adding explicit hydrogens, and renaming by IUPAC name.

   ARGUMENTS
     molecule (OEMol) - the molecule to be normalized.

   EXAMPLES

   Normalize a molecule (creating it first).

   >>> molecule = createMoleculeByName('benzene')
   >>> normalized_molecule = normalizeMolecule(molecule)

   """
   
   # Find ring atoms and bonds
   openeye.oechem.OEFindRingAtomsAndBonds(molecule) 
   
   # Assign aromaticity.
   openeye.oechem.OEAssignAromaticFlags(molecule, openeye.oechem.OEAroModelOpenEye)   

   # Add hydrogens.
   openeye.oechem.OEAddExplicitHydrogens(molecule)

   # Set title to IUPAC name.
   name = openeye.oeiupac.OECreateIUPACName(molecule)
   molecule.SetTitle(name)

   return molecule

#=============================================================================================

def expandConformations(molecule, maxconfs=None, threshold=None, include_original=False, torsionlib=None, verbose=False, strict_typing=None):   
   """
   Enumerate conformations of the molecule with OpenEye's Omega.

   ARGUMENTS
   molecule (OEMol) - molecule to enumerate conformations for

   OPTIONAL ARGUMENTS
     include_original (boolean) - if True, original conformation is included (default: False)
     maxconfs (integer) - if set to an integer, limits the maximum number of conformations to generated -- maximum of 120 (default: None)
     threshold (real) - threshold in RMSD (in Angstroms) for retaining conformers -- lower thresholds retain more conformers (default: None)
     torsionlib (string) - if a path to an Omega torsion library is given, this will be used instead (default: None)
     verbose (boolean) - if True, omega will print extra information
     strict_typing (boolean) -- if specified, pass option to SetStrictAtomTypes for Omega to control whether related MMFF types are allowed to be substituted for exact matches.

   RETURN VALUES
     expanded_molecule - molecule with expanded conformations

   EXAMPLES
     # create a new molecule with Omega-expanded conformations
     expanded_molecule = expandConformations(molecule)

     
   """

   # Initialize omega
   omega = openeye.oeomega.OEOmega()

   if strict_typing != None:
      # Set atom typing options
      omega.SetStrictAtomTypes( strict_typing)

   # Set maximum number of conformers.
   if maxconfs:
      omega.SetMaxConfs(maxconfs)
     
   # Set whether given conformer is to be included.
   omega.SetIncludeInput(include_original)
   
   # Set RMSD threshold for retaining conformations.
   if threshold:
      omega.SetRMSThreshold(threshold) 
 
   # If desired, do a torsion drive using the specified library.
   if torsionlib:
      omega.SetTorsionLibrary(torsionlib)

   # Create copy of molecule.
   expanded_molecule = openeye.oechem.OEMol(molecule)   

   # Enumerate conformations.
   omega(expanded_molecule)

   # verbose output
   if verbose: print "%d conformation(s) produced." % expanded_molecule.NumConfs()

   # return conformationally-expanded molecule
   return expanded_molecule

#=============================================================================================

def selectProtonationState(molecule, formal_charge, verbose=False):
   """
   Select the desired formal charge state of the specified molecule.
   
   ARGUMENTS
     molecule (OEMol) - the molecule of interest
     formal_charge (int) - the desired formal charge

   OPTIONAL ARGUMENTS
     verbose (boolean) - if True, verbose output will be printed (default: False)

   RETURNS
     molecule (OEMol) - molecule with the desired formal charge

   EXAMPLES

   Select the desired charge of a phosphate group.
   >>> molecule = createMoleculeByName('acetate')
   >>> molecule = selectProtonationState(molecule, -1)

   """

   # Enumerate protonation states.
   protonation_states = enumerateStates(molecule, enumerate="protonation", verbose=verbose)

   # Select a protonation state that matches the desired formal charge.
   for molecule in protonation_states:
      if formalCharge(molecule) == formal_charge:
         # Return the molecule if we've found one in the desired protonation state.
         return molecule
      
   # Raise an exception if we did not succeed.
   print "enumerateStates did not enumerate a molecule with desired formal charge."
   print "Options are:"
   for molecule in protonation_states:
      print "%s, formal charge %d" % (molecule.GetTitle(), formalCharge(molecule))
   raise Exception("Could not find desired formal charge.")

   return

#=============================================================================================

def assignPartialCharges(molecule, charge_model='am1bcc', multiconformer=False, minimize_contacts=False, verbose=False):
   """
   Assign partial charges to a molecule using OEChem oeproton.

   ARGUMENTS
     molecule (OEMol) - molecule for which charges are to be assigned

   OPTIONAL ARGUMENTS
     charge_model (string) - partial charge model, one of ['am1bcc'] (default: 'am1bcc')
     multiconformer (boolean) - if True, multiple conformations are enumerated and the resulting charges averaged (default: False)
     minimize_contacts (boolean) - if True, intramolecular contacts are eliminated by minimizing conformation with MMFF with all charges set to absolute values (default: False)
     verbose (boolean) - if True, information about the current calculation is printed

   RETURNS
     charged_molecule (OEMol) - the charged molecule with GAFF atom types

   NOTES
     multiconformer and minimize_contacts can be combined, but this can be slow

   EXAMPLES

   Assign AM1-BCC partial charges to phenol.
   >>> # create a molecule
   >>> molecule = createMoleculeByName('phenol')
   >>> # assign am1bcc charges
   >>> charged_molecule = assignPartialCharges(molecule, charge_model='am1bcc')
   """

   # Check that molecule has atom names; if not we need to assign them
   # TODO: Is this strictly necessary?  Can we assign names to only our internal working copy?
   assignNames = False
   for atom in molecule.GetAtoms():
      if atom.GetName() == '':
         assignNames = True #In this case we are missing an atom name and will need to assign
   if assignNames:
      if verbose: print "Assigning TRIPOS names to atoms"
      openeye.oechem.OETriposAtomNames(molecule)

   # Check input pameters.
   supported_charge_models  = ['am1bcc']
   if not (charge_model in supported_charge_models):
      raise Exception("Charge model %(charge_model)s not in supported set of %(supported_charge_models)s" % vars())

   # If a multiconformer fit is desired, expand conformations; otherwise make a copy of molecule.
   if multiconformer:
      expanded_molecule = expandConformations(molecule)
   else:
      expanded_molecule = openeye.oechem.OEMol(molecule)
   nconformers = expanded_molecule.NumConfs()
   if verbose: print 'assignPartialCharges: %(nconformers)d conformations will be used in charge determination.' % vars()
   
   # Set up storage for partial charges.
   # TODO: Is this why the atoms need names?
   partial_charges = dict()
   for atom in molecule.GetAtoms():
      name = atom.GetName()
      partial_charges[name] = 0.0

   # Assign partial charges for each conformation.
   conformer_index = 0
   for conformation in expanded_molecule.GetConfs():
      conformer_index += 1
      if verbose and multiconformer: print "assignPartialCharges: conformer %d / %d" % (conformer_index, expanded_molecule.NumConfs())

      # Assign partial charges to a copy of the molecule.
      if verbose: print "assignPartialCharges: determining partial charges..."
      charged_molecule = openeye.oechem.OEMol(conformation)   
      if charge_model == 'am1bcc':
         openeye.oequacpac.OEAssignPartialCharges(charged_molecule, openeye.oequacpac.OECharges_AM1BCC)
      
      # Minimize with positive charges to splay out fragments, if desired.
      if minimize_contacts:
         if verbose: print "assignPartialCharges: Minimizing conformation with MMFF and absolute value charges..." % vars()         
         # Set partial charges to absolute value.
         for atom in charged_molecule.GetAtoms():
            atom.SetPartialCharge(abs(atom.GetPartialCharge()))
         # Minimize in Cartesian space to splay out substructures.
         szybki = openeye.oeszybki.OESzybki() # create an instance of OESzybki
         szybki.SetRunType(openeye.oeszybki.OERunType_CartesiansOpt) # set minimization         
         szybki.SetUseCurrentCharges(True) # use charges for minimization
         results = openeye.oeszybki.szybki(charged_molecule)
         # Recompute charges;
         if verbose: print "assignPartialCharges: redetermining partial charges..."         
         openeye.oequacpac.OEAssignPartialCharges(charged_molecule, openeye.oequacpac.OECharges_AM1BCC)         
         
      # Accumulate partial charges.
      for atom in charged_molecule.GetAtoms():
         name = atom.GetName()
         partial_charges[name] += atom.GetPartialCharge()

   # Compute and store average partial charges in a copy of the original molecule.
   charged_molecule = openeye.oechem.OEMol(molecule)
   for atom in charged_molecule.GetAtoms():
      name = atom.GetName()
      atom.SetPartialCharge(partial_charges[name] / nconformers)

   # Return the charged molecule
   return charged_molecule

#=============================================================================================

def assignPartialChargesWithAntechamber(molecule, charge_model='bcc', judgetypes=None, cleanup=True, verbose=False, netcharge=None):
   """
   Assign partial charges to a molecule using antechamber from AmberTools.

   ARGUMENTS
     molecule (OEMol) - molecule for which charges are to be computed

   OPTIONAL ARGUMENTS
     charge_model (string) - antechamber partial charge model (default: 'bcc')
     judgetypes (integer) - if specified, this is provided as a -j argument to antechamber (default: None)
     cleanup (boolean) - clean up temporary files (default: True)
     verbose (boolean) - if True, verbose output of subprograms is displayed
     netcharge (integer) -- if given, give -nc (netcharge) option to antechamber in calculation of charges

   RETURNS
     charged_molecule (OEMol) - the charged molecule with GAFF atom types

   REQUIREMENTS
     antechamber (on PATH)

   WARNING
     This module is currently broken, as atom names get all jacked up during readMolecule() for these mol2 files.
     DLM 4/2/2009: I believe I have fixed the module by switching antechamber to use sybyl atom types. However please note that these will not work as input to tleap and must use GAFF/AMBER atom types there. 
     
   EXAMPLES
   
   Assign AM1-BCC charges to phenol with antechamber (creating it first).

   >>> molecule = createMoleculeByName('phenol')
   >>> charged_molecule = assignPartialChargesWithAntechamber(molecule, charge_model='bcc', netcharge=0)

   """

   # Create temporary working directory and move there.
   import os, tempfile
   old_directory = os.getcwd()
   working_directory = tempfile.mkdtemp()   
   os.chdir(working_directory)

   # Write input mol2 file to temporary directory.
   uncharged_molecule_filename = tempfile.mktemp(suffix = '.mol2', dir = working_directory)
   if verbose: print "Writing uncharged molecule to %(uncharged_molecule_filename)s" % vars()
   writeMolecule(molecule, uncharged_molecule_filename)

   # Create filename for output mol2 file.
   charged_molecule_filename = tempfile.mktemp(suffix ='.mol2', dir = working_directory)

   # Determine net charge of ligand from formal charges.
   formal_charge = formalCharge(molecule)

   # Run antechamber to assign GAFF atom types and charge ligand.
   command = 'antechamber -i %(uncharged_molecule_filename)s -fi mol2 -o %(charged_molecule_filename)s -fo mol2 -c %(charge_model)s -at sybyl' % vars()
   if netcharge:
       command +=' -nc %d' % netcharge
   if judgetypes: 
      command += ' -j %(judgetypes)d' % vars()
   if verbose: print command
   import commands
   output = commands.getoutput(command)
   if verbose: print output

   # Read new mol2 file.
   if verbose: print "Reading charged molecule from %(charged_molecule_filename)s" % vars()   
   charged_molecule = readMolecule(charged_molecule_filename)

   # Restore old working directory.
   os.chdir(old_directory)

   # Clean up temporary working directory.
   if cleanup:      
      import os, os.path
      try:
         for filename in os.listdir(working_directory):
            os.unlink(os.path.join(working_directory, filename))
         os.remove(working_directory)
      except:
         pass
   else:
      if verbose: print "Leaving working directory intact: %s" % working_directory
      
   # Return the charged molecule
   return charged_molecule

#=============================================================================================

def enumerateStates(molecules, enumerate='protonation', consider_aromaticity=True, maxstates=200, verbose=False):
    """
    Enumerate protonation or tautomer states for a list of molecules.

    ARGUMENTS
      molecules (OEMol or list of OEMol) - molecules for which states are to be enumerated

    OPTIONAL ARGUMENTS
      enumerate (boolean) - type of states to expand -- 'protonation' or 'tautomer' (default: 'protonation')
      verbose (boolean) - if True, will print out debug output (default: False)

    RETURNS
      states (list of OEMol) - molecules in different protonation or tautomeric states

    TODO
      Modify to use a single molecule or a list of molecules as input.
      Apply some regularization to molecule before enumerating states?
      Pick the most likely state?
      Add more optional arguments to control behavior.

      
    EXAMPLES
    
    Enumerate protonation states of phosphate.
    >>> molecule = createMoleculeByName('phosphate')
    >>> states = enumerateStates(molecule, enumerate='protonation')

    Enumerate tautomeric states of ribose.
    >>> molecule = createMoleculeByName('4-pyridone')
    >>> states = enumerateStates(molecule, enumerate='tautomer')

    """

    # If 'molecules' is not a list, promote it to a list.
    if type(molecules) != type(list()):
       molecules = [molecules]

    # Check input arguments.
    if not ((enumerate == "protonation") or (enumerate == "tautomer")):
        raise Exception("'enumerate' argument must be either 'protonation' or 'tautomer' -- instead got '%s'" % enumerate)

    # Create an internal output stream to expand states into.
    ostream = openeye.oechem.oemolostream()
    ostream.openstring()
    ostream.SetFormat(openeye.oechem.OEFormat_SDF)
    
    # Default parameters.
    only_count_states = False # enumerate states, don't just count them

    # Enumerate states for each molecule in the input list.
    states_enumerated = 0
    for molecule in molecules:
        if (verbose): print "Enumerating states for molecule %s." % molecule.GetTitle()
        
        # Dump enumerated states to output stream (ostream).
        if (enumerate == "protonation"): 
            # Create a functor associated with the output stream.
            functor = openeye.oequacpac.OETyperMolFunction(ostream, consider_aromaticity, False, maxstates)
            # Enumerate protonation states.
            if (verbose): print "Enumerating protonation states..."
            states_enumerated += openeye.oequacpac.OEEnumerateFormalCharges(molecule, functor, verbose)        
        elif (enumerate == "tautomer"):
            # Create a functor associated with the output stream.
            functor = openeye.oequacpac.OETautomerMolFunction(ostream, consider_aromaticity, False, maxstates)
            # Enumerate tautomeric states.
            if (verbose): print "Enumerating tautomer states..."
            states_enumerated += openeye.oequacpac.OEEnumerateTautomers(molecule, functor, verbose)    
    if verbose: print "Enumerated a total of %d states." % states_enumerated

    # Collect molecules from output stream into a list.
    states = list()
    if (states_enumerated > 0):    
        state = openeye.oechem.OEMol()
        istream = openeye.oechem.oemolistream()
        istream.openstring(ostream.GetString())
        istream.SetFormat(openeye.oechem.OEFormat_SDF)
        while openeye.oechem.OEReadMolecule(istream, state):
           states.append(openeye.oechem.OEMol(state)) # append a copy

    # Return the list of expanded states as a Python list of OEMol() molecules.
    return states

#=============================================================================================

def superimposeMolecule(fitmol, refmol, maxconfs=None, verbose=False):

   """
   Fit a multi-conformer target molecule to a reference molecule using OpenEye Shape tookit, and return an OE molecule with the top conformers of the resulting fit. Tanimoto scores also returned.

   ARGUMENTS
      fitmol (OEMol) -- the (multi-conformer) molecule to be fit.
      refmol (OEMol) -- the molecule to fit to

   OPTIONAL ARGUMENTS
      maxconfs -- Limit on number of conformations to return; default return all
      verbose -- Turn verbosity on/off

   RETURNS
      outmol (OEMol) -- output (fit) molecule resulting from fitmol
      scores

   NOTES
      Passing this a multi-conformer fitmol is recommended for any molecule with rotatable bonds as fitting only includes rotations and translations, so one of the provided conformers must already have right bond rotations.

   EXAMPLES

   Fit indene to indole.
   
   >>> indene = createMoleculeByName('indene')
   >>> indole = createMoleculeByName('indole')
   >>> indene_fit = superimposeMolecule(indene, indole)
      
   """

   # Set up storage for overlay
   best = openeye.oeshape.OEBestOverlay()
   # Set reference molecule
   best.SetRefMol(refmol)

   if verbose:
      print "Reference title: ", refmol.GetTitle()
      print "Fit title: ", fitmol.GetTitle()
      print "Num confs: ", fitmol.NumConfs()

   resCount = 0
   # Each conformer-conformer pair generates multiple scores since there are multiple possible overlays; we only want the best. Load the best score for each conformer-conformer pair into an iterator and loop over it
   scoreiter = openeye.oeshape.OEBestOverlayScoreIter()
   openeye.oeshape.OESortOverlayScores(scoreiter, best.Overlay(fitmol), openeye.oeshape.OEHighestTanimoto())
   tanimotos = [] # Storage for scores
   for score in scoreiter:
      # Get the particular conformation of this match and transform to overlay onto reference structure
      tmpmol = openeye.oechem.OEMol(fitmol.GetConf(openeye.oechem.OEHasConfIdx(score.fitconfidx)))
      score.Transform(tmpmol)
      # Store to output molecule
      try: 
         # If it already exists
         outmol.NewConf(tmpmol)
      except: 
         # Otherwise
         outmol = tmpmol

      # Print some info
      if verbose:
         print "FitConfIdx: %-4d" % score.fitconfidx,
         print "RefConfIdx: %-4d" % score.refconfidx,
         print "Tanimoto: %.2f" % score.tanimoto
      # Store score
      tanimotos.append(score.tanimoto)
      resCount+=1

      if resCount == maxconfs: break

   return ( outmol, tanimotos )

#=============================================================================================
# METHODS FOR WRITING OR EXPORTING MOLECULES
#=============================================================================================

def writeMolecule(molecule, filename, substructure_name='MOL', preserve_atomtypes=False):
   """
   Write a molecule to a file in any format OpenEye autodetects from filename (such as .mol2).
   WARNING: The molecule passed to this function will be modified (standardized) before writing by the high-level OEWriteMolecule function.
   OEWriteConstMolecule is used, to avoid changing the molecule you pass in.

   ARGUMENTS
     molecule (OEMol) - the molecule to be written
     filename (string) - the file to write the molecule to (type autodetected from filename)

   OPTIONAL ARGUMENTS
     substructure_name (String) - if a mol2 file is written, this is used for the substructure name (default: 'MOL')
     preserve_atomtypes (bool) - if True, a mol2 file will be written with atom types preserved

   RETURNS
     None

   NOTES
     Multiple conformers are written.

   EXAMPLES

   Write a molecule to a temporary file (creating one first).

   >>> # Create a molecule
   >>> molecule = createMoleculeByName('phenol')
   >>> # Write it to disk
   >>> import tempfile
   >>> filename = tempfile.mktemp(suffix='.mol2')
   >>> writeMolecule(molecule, filename)

   TODO   

   * I suspect OEWriteConstMolecule is used backwards here; fix docs too.

   """

   # Open output stream.
   ostream = openeye.oechem.oemolostream(filename)

   # Define internal function for writing multiple conformers to an output stream.
   def write_all_conformers(ostream, molecule):
      # write all conformers of each molecule
      for conformer in molecule.GetConfs():
         if preserve_atomtypes: 
            openeye.oechem.OEWriteMol2File(ostream, conformer)
         else: 
            openeye.oechem.OEWriteConstMolecule(ostream, conformer)
      return

   # If 'molecule' is actually a list of molecules, write them all.
   if type(molecule) == type(list()):
      for individual_molecule in molecule:
         write_all_conformers(ostream, individual_molecule)
   else:
      write_all_conformers(ostream, molecule)

   # Close the stream.
   ostream.close()

   # Replace substructure name if mol2 file.
   import os
   suffix = os.path.splitext(filename)[-1]
   if (suffix == '.mol2' and substructure_name != None):
      modifySubstructureName(filename, substructure_name)

   return

#=============================================================================================

def createSystem(molecule, forcefield='GAFF', charge_model=None, formal_charge=None, constraints=None, gbmodel=None, verbose=False):
   """
   Construct an OpenMM System object from a given molecule using Antechamber.

   ARGUMENTS
     molecule (OEMol) - molecule to parameterize (only the first configuration will be used if multiple are present)

   OPTIONAL ARGUMENTS
     forcefield (string) - forcefield to assign parameters from; options are ['GAFF'] (default: GAFF)
     charge_model (string) - if specified, Antechamber will be asked to assign charges; otherwise charges in mol2 will be used (default: None)
     formal_charge (float) - if specified, this net charge will be passed to Antechamber's charging routines
     constraints (string) - if specified, bond constraints will be imposed ['HBonds', 'HAngles'] (default: None)
     gbmodel (string) - if specified, GB parameters will be assigned for specified GB model in ['OBC1', 'OBC2'] (default: None)
     verbose (boolean) - if True, verbose output will be printed

   RETURNS
     system (simtk.openmm.System) - OpenMM System for small molecule in vacuum or implicit solvent
     positions (simtk.unit.Quantity natoms x 3 array with units of distance) - atomic positions

   REQUIREMENTS
     AmberTools installation (in PATH)

   EXAMPLES

   Assign GAFF parameters and AM1-BCC charges to phenol using AmberTools Antechamber.

   >>> molecule = createMoleculeByName('phenol')
   >>> [system, positions] = createSystem(molecule, forcefield='GAFF', charge_model='bcc')

   WARNINGS
   
   * gbmodel does not yet select appropriate GB parameters in LEaP

   TODO

   * Improve flexibility of defining nonbonded treatment
   
   """

   # Create temporary working directory.
   import os, tempfile
   working_directory = tempfile.mkdtemp()
   old_directory = os.getcwd()
   os.chdir(working_directory)
   if verbose: print "Working directory is %(working_directory)s" % vars()

   # Determine net formal charge if not specified.
   if not formal_charge:
      formal_charge = formalCharge(molecule)

   # Make a copy of molecule, so as not to modify original.
   molecule = openeye.oechem.OEMol(molecule)

   if not charge_model:
      # TODO: Truncate charges to four decimal places, ensuring that total charge is integral.
      # NOTE: OE tools only write charges out to four decimal places.
      pass
   
   # Write molecule to mol2 file.
   tripos_mol2_filename = os.path.join(working_directory, 'tripos.mol2')
   writeMolecule(molecule, tripos_mol2_filename)
   
   # Run antechamber to assign GAFF atom types (and optionally charges).
   judgetypes = None
   gaff_mol2_filename = os.path.join(working_directory, 'gaff.mol2')   
   command = 'antechamber -i %(tripos_mol2_filename)s -fi mol2 -o %(gaff_mol2_filename)s -fo mol2' % vars()
   if judgetypes: 
      command += ' -j %(judgetypes)d' % vars()
   if charge_model:
      command += ' -c %(charge_model)s -nc %(formal_charge)d' % vars()   
   if verbose: print command
   import commands
   output = commands.getoutput(command)
   if verbose or (output.find('Warning')>=0): print output   
   
   # Generate frcmod file for additional GAFF parameters.
   frcmod_filename_tmp = os.path.join(working_directory, 'gaff.frcmod')
   commands.getoutput('parmchk -i %(gaff_mol2_filename)s -f mol2 -o %(frcmod_filename_tmp)s' % vars())

   # Create AMBER topology/coordinate files using LEaP.
   # TODO: Incorporate GB model parameters here.
   leapscript = """\
source leaprc.gaff 
mods = loadAmberParams %(frcmod_filename_tmp)s
molecule = loadMol2 %(gaff_mol2_filename)s
saveAmberParm molecule amber.prmtop amber.crd
quit""" % vars()
   leapin_filename = os.path.join(working_directory, 'leap.in')
   outfile = open(leapin_filename, 'w')
   outfile.write(leapscript)
   outfile.close()

   tleapout = commands.getoutput('tleap -f %(leapin_filename)s' % vars())
   if verbose: print tleapout
   tleapout = tleapout.split('\n')
   # Shop any warnings.
   if verbose:
      fnd = False
      for line in tleapout:
         tmp = line.upper()
         if tmp.find('WARNING')>-1: 
            print line
            fnd = True
         if tmp.find('ERROR')>-1: 
            print line
            fnd = True
      if fnd:
         print "Any LEaP warnings/errors are above."

   # Read prmtop and inpcrd.
   import simtk.openmm.app as app
   prmtop = app.AmberPrmtopFile('amber.prmtop')
   inpcrd = app.AmberInpcrdFile('amber.crd')
   
   # Create OpenMM System.
   implicit_solvent = None
   try:
      implicit_solvent = getattr(app, gbmodel)
   except:
      pass
   openmm_constraints = None
   try:
      openmm_constraints = getattr(app, constraints)
   except:
      pass
   system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=openmm_constraints, implicitSolvent=implicit_solvent)
   positions = inpcrd.getPositions()

   # Restore old directory.
   os.chdir(old_directory)   

   # Clean up temporary files.
   os.chdir(old_directory)
   if verbose:
      print "Work done in %s..." % working_directory
   else:
      commands.getoutput('rm -r %s' % working_directory)

   return [system, positions]

#=============================================================================================

def modifySubstructureName(mol2file, name):
   """
   Replace the substructure name (subst_name) in a mol2 file.

   ARGUMENTS
     mol2file (string) - name of the mol2 file to modify
     name (string) - new substructure name

   NOTES
     This is useful becuase the OpenEye tools leave this name set to <0>.
     The transformation is only applied to the first molecule in the mol2 file.

   TODO
     This function is still difficult to read.  It should be rewritten to be comprehensible by humans.

   EXAMPLES

   Modify the substructure name of phenol (after writing it to a mol2 file).

   >>> molecule = createMoleculeByName('phenol')
   >>> import tempfile
   >>> filename = tempfile.mktemp(suffix='.mol2')
   >>> writeMolecule(molecule, filename)
   >>> modifySubstructureName(filename, 'MOL')

   """

   # Read mol2 file.
   file = open(mol2file, 'r')
   text = file.readlines()
   file.close()

   # Find the atom records.
   atomsec = []
   ct = 0
   while text[ct].find('<TRIPOS>ATOM')==-1:
     ct+=1
   ct+=1
   atomstart = ct
   while text[ct].find('<TRIPOS>BOND')==-1:
     ct+=1
   atomend = ct

   atomsec = text[atomstart:atomend]
   outtext=text[0:atomstart]
   repltext = atomsec[0].split()[7] # mol2 file uses space delimited, not fixed-width

   # Replace substructure name.
   for line in atomsec:
     # If we blindly search and replace, we'll tend to clobber stuff, as the subst_name might be "1" or something lame like that that will occur all over. 
     # If it only occurs once, just replace it.
     if line.count(repltext)==1:
       outtext.append( line.replace(repltext, name) )
     else:
       # Otherwise grab the string left and right of the subst_name and sandwich the new subst_name in between. This can probably be done easier in Python 2.5 with partition, but 2.4 is still used someplaces.
       # Loop through the line and tag locations of every non-space entry
       blockstart=[]
       ct=0
       c=' '
       for ct in range(len(line)):
         lastc = c
         c = line[ct]
         if lastc.isspace() and not c.isspace():
           blockstart.append(ct)
       line = line[0:blockstart[7]] + line[blockstart[7]:].replace(repltext, name, 1)
       outtext.append(line)
       
   # Append rest of file.
   for line in text[atomend:]:
     outtext.append(line)
     
   # Write out modified mol2 file, overwriting old one.
   file = open(mol2file,'w')
   file.writelines(outtext)
   file.close()

   return

#=============================================================================================
# MAIN
#=============================================================================================

if __name__ == '__main__':
   import doctest
   doctest.testmod()


