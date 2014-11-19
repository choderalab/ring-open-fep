#=============================================================================================
# Generate mol2 files for small molecules given IUPAC names.
#
# PROTOCOL
#
# * Construct molecule from IUPAC name (protonation and tautomeric states are heuristically guessed) [OpenEye OEMol]
# * Generate multiple likely conformations to use for replicates [OpenEye Omega]
# 
# Written by John D. Chodera <jchodera@gmail.com> 2013-01-28
#=============================================================================================

#=============================================================================================
# Imports
#=============================================================================================

import commands
import os    

import numpy

from openeye.oechem import *
from openeye.oeomega import *
from openeye.oeiupac import *
from openeye.oeshape import *
try:
   from openeye.oequacpac import * #DLM added 2/25/09 for OETyperMolFunction; replacing oeproton
except:
   from openeye.oeproton import * #GJR temporary fix because of old version of openeye tools
from openeye.oeiupac import *
from openeye.oeszybki import *

#=============================================================================================
# CHANGELOG
#============================================================================================

#=============================================================================================
# PARAMETERS
#=============================================================================================

#=============================================================================================
# SUBROUTINES
#=============================================================================================

def write_file(filename, contents):
    """Write the specified contents to a file.

    ARGUMENTS
      filename (string) - the file to be written
      contents (string) - the contents of the file to be written

    """

    outfile = open(filename, 'w')

    if type(contents) == list:
        for line in contents:
            outfile.write(line)
    elif type(contents) == str:
        outfile.write(contents)
    else:
        raise "Type for 'contents' not supported: " + repr(type(contents))

    outfile.close()

    return

def read_file(filename):
    """Read contents of the specified file.

    ARGUMENTS
      filename (string) - the name of the file to be read

    RETURNS
      lines (list of strings) - the contents of the file, split by line
    """

    infile = open(filename, 'r')
    lines = infile.readlines()
    infile.close()

    return lines

#=============================================================================================
def createMoleculeFromIUPAC(name, verbose = False, charge = None, strictTyping = None):
   """Generate a small molecule from its IUPAC name.

   ARGUMENTS
     IUPAC_name (string) - IUPAC name of molecule to generate

   OPTIONAL ARGUMENTS
     verbose (boolean) - if True, subprocess output is shown (default: False)
     charge (int) - if specified, a form of this molecule with the desired charge state will be produced (default: None)
     strictTyping (boolean) -- if set, passes specified value to omega (see documentation for expandConformations)

   RETURNS
     molecule (OEMol) - the molecule

   NOTES
     OpenEye LexiChem's OEParseIUPACName is used to generate the molecle.
     The molecule is normalized by adding hydrogens.
     Omega is used to generate a single conformation.
     Also note that atom names will be blank coming from this molecule. They are assigned when the molecule is written, or one can assign using OETriposAtomNames for example.

   EXAMPLES
     # Generate a mol2 file for phenol.
     molecule = createMoleculeFromIUPAC('phenol')
     
   """

   # Create an OEMol molecule from IUPAC name.
   molecule = OEMol() # create a molecule
   status = OEParseIUPACName(molecule, name) # populate the molecule from the IUPAC name

   # Normalize the molecule.
   normalizeMolecule(molecule)

   # Generate a conformation with Omega
   omega = OEOmega()
   #omega.SetVerbose(verbose)
   #DLM 2/27/09: Seems to be obsolete in current OEOmega
   if strictTyping != None:
     omega.SetStrictAtomTypes( strictTyping) 
   
   omega.SetIncludeInput(False) # don't include input
   omega.SetMaxConfs(1) # set maximum number of conformations to 1
   omega(molecule) # generate conformation      

   if (charge != None):
      # Enumerate protonation states and select desired state.
      protonation_states = enumerateStates(molecule, enumerate = "protonation", verbose = verbose)
      for molecule in protonation_states:
         if formalCharge(molecule) == charge:
            # Return the molecule if we've found one in the desired protonation state.
            return molecule
      if formalCharge(molecule) != charge:
         print "enumerateStates did not enumerate a molecule with desired formal charge."
         print "Options are:"
         for molecule in protonation_states:
            print "%s, formal charge %d" % (molecule.GetTitle(), formalCharge(molecule))
         raise "Could not find desired formal charge."
    
   # Return the molecule.
   return molecule
#=============================================================================================
def readMolecule(filename, normalize = False):
   """Read in a molecule from a file (such as .mol2).

   ARGUMENTS
     filename (string) - the name of the file containing a molecule, in a format that OpenEye autodetects (such as .mol2)

   OPTIONAL ARGUMENTS
     normalize (boolean) - if True, molecule is normalized (renamed, aromaticity, protonated) after reading (default: False)

   RETURNS
     molecule (OEMol) - OEMol representation of molecule

   EXAMPLES
     # read a mol2 file
     molecule = readMolecule('phenol.mol2')
     # works with any type of file that OpenEye autodetects
     molecule = readMolecule('phenol.sdf')
   """

   # Open input stream.
   istream = oemolistream()
   istream.open(filename)

   # Create molecule.
   molecule = OEMol()   

   # Read the molecule.
   OEReadMolecule(istream, molecule)

   # Close the stream.
   istream.close()

   # Normalize if desired.
   if normalize: normalizeMolecule(molecule)

   return molecule
#=============================================================================================
# METHODS FOR INTERROGATING MOLECULES
#=============================================================================================
def formalCharge(molecule):
   """Report the net formal charge of a molecule.

   ARGUMENTS
     molecule (OEMol) - the molecule whose formal charge is to be determined

   RETURN VALUES
     formal_charge (integer) - the net formal charge of the molecule

   EXAMPLE
     net_charge = formalCharge(molecule)
   """

   # Create a copy of the molecule.
   molecule_copy = OEMol(molecule)

   # Assign formal charges.
   OEFormalPartialCharges(molecule_copy)

   # Compute net formal charge.
   formal_charge = int(round(OENetCharge(molecule_copy)))

   # return formal charge
   return formal_charge
#=============================================================================================
# METHODS FOR MODIFYING MOLECULES
#=============================================================================================
def normalizeMolecule(molecule):
   """Normalize the molecule by checking aromaticity, adding explicit hydrogens, and renaming by IUPAC name.

   ARGUMENTS
     molecule (OEMol) - the molecule to be normalized.

   EXAMPLES
     # read a partial molecule and normalize it
     molecule = readMolecule('molecule.sdf')
     normalizeMolecule(molecule)
   """
   
   # Find ring atoms and bonds
   # OEFindRingAtomsAndBonds(molecule) 
   
   # Assign aromaticity.
   OEAssignAromaticFlags(molecule, OEAroModelOpenEye)   

   # Add hydrogens.
   OEAddExplicitHydrogens(molecule)

   # Set title to IUPAC name.
   name = OECreateIUPACName(molecule)
   molecule.SetTitle(name)

   return molecule
#=============================================================================================
def expandConformations(molecule, maxconfs = None, threshold = None, include_original = False, torsionlib = None, verbose = False, strictTyping = None):   
   """Enumerate conformations of the molecule with OpenEye's Omega after normalizing molecule. 

   ARGUMENTS
   molecule (OEMol) - molecule to enumerate conformations for

   OPTIONAL ARGUMENTS
     include_original (boolean) - if True, original conformation is included (default: False)
     maxconfs (integer) - if set to an integer, limits the maximum number of conformations to generated -- maximum of 120 (default: None)
     threshold (real) - threshold in RMSD (in Angstroms) for retaining conformers -- lower thresholds retain more conformers (default: None)
     torsionlib (string) - if a path to an Omega torsion library is given, this will be used instead (default: None)
     verbose (boolean) - if True, omega will print extra information
     strictTyping (boolean) -- if specified, pass option to SetStrictAtomTypes for Omega to control whether related MMFF types are allowed to be substituted for exact matches.

   RETURN VALUES
     expanded_molecule - molecule with expanded conformations

   EXAMPLES
     # create a new molecule with Omega-expanded conformations
     expanded_molecule = expandConformations(molecule)

     
   """
   # Initialize omega
   omega = OEOmega()
   if strictTyping != None:
     omega.SetStrictAtomTypes( strictTyping)
   #Set atom typing options

   # Set verbosity.
   #omega.SetVerbose(verbose)
   #DLM 2/27/09: Seems to be obsolete in current OEOmega

   # Set maximum number of conformers.
   if maxconfs:
      omega.SetMaxConfs(maxconfs)
     
   # Set whether given conformer is to be included.
   omega.SetIncludeInput(include_original)
   
   # Set RMSD threshold for retaining conformations.
   if threshold:
      omega.SetRMSThreshold(threshold) 
 
   # If desired, do a torsion drive.
   if torsionlib:
      omega.SetTorsionLibrary(torsionlib)

   # Create copy of molecule.
   expanded_molecule = OEMol(molecule)   

   # Enumerate conformations.
   omega(expanded_molecule)


   # verbose output
   if verbose: print "%d conformation(s) produced." % expanded_molecule.NumConfs()

   # return conformationally-expanded molecule
   return expanded_molecule

#=============================================================================================
def modifySubstructureName(mol2file, name):
   """Replace the substructure name (subst_name) in a mol2 file.

   ARGUMENTS
     mol2file (string) - name of the mol2 file to modify
     name (string) - new substructure name

   NOTES
     This is useful becuase the OpenEye tools leave this name set to <0>.
     The transformation is only applied to the first molecule in the mol2 file.

   TODO
     This function is still difficult to read.  It should be rewritten to be comprehensible by humans.
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

def writeMolecule(molecule, filename, substructure_name = 'MOL', preserve_atomtypes = False):
   """Write a molecule to a file in any format OpenEye autodetects from filename (such as .mol2).
   WARNING: The molecule will be standardized before writing by the high-level OEWriteMolecule function.
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
     # create a molecule
     molecule = createMoleculeFromIUPAC('phenol')
     # write it as a mol2 file
     writeMolecule(molecule, 'phenol.mol2')
   """

   # Open output stream.
   ostream = oemolostream()
   ostream.open(filename)

   # Define internal function for writing multiple conformers to an output stream.
   def write_all_conformers(ostream, molecule):
      # write all conformers of each molecule
      for conformer in molecule.GetConfs():
         if preserve_atomtypes: OEWriteMol2File(ostream, conformer)
         else: OEWriteConstMolecule(ostream, conformer)
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
   suffix = os.path.splitext(filename)[-1]
   if (suffix == '.mol2' and substructure_name != None):
      modifySubstructureName(filename, substructure_name)

   return

#=============================================================================================

def generate_mol2(iupac_name, filename, molecule_name=None, formal_charge=None, verbose=False):
    """
    Create a mol2 file from a given IUPAC name.

    ARGUMENTS
      iupac_name (string) - the IUPAC or common name of the molecule to be created
      filename (string) - the mol2 filename to be created.

    OPTIONAL ARGUMENTS
      molecule_name (string) - if provided, will set the molecule name; if None, will use IUPAC name (default: None)
      formal_charge (int) - if set, will select form of the molecule with the given formal charge (default: None)
      verbose (boolean) - if True, extra debug information will be printed (default: False)

    """

    # Create molecule from IUPAC name
    molecule = createMoleculeFromIUPAC(iupac_name, charge=formal_charge)

    # Replace the title with the common name
    if molecule_name:
        molecule.SetTitle(molecule_name)
    else:
        molecule.SetTitle(iupac_name)

    # Expand set of conformations so we have multiple conformations to start from.
    expandConformations(molecule)
    
    # Center the solute molecule.
    OECenter(molecule)

    # Write mol2 file for the molecule.
    writeMolecule(molecule, filename)

    return

#=============================================================================================
# MAIN
#=============================================================================================

# Create a some test molecules.
molecule_names = ['benzene', 'cyclohexane', '1,3-diethylbenzene']
for molecule_name in molecule_names:
    filename = molecule_name + '.mol2'
    generate_mol2(molecule_name, filename)

