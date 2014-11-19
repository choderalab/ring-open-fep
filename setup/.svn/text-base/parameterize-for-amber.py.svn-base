#=============================================================================================
# Parameterize small molecule for vacuum simulation with GAFF and AM1-BCC, generating prmtop/inpcrd.
#
# PROTOCOL
#
# * Run AmberTools 'antechamber' to generate GAFF mol2 file with AM1-BCC charges.
# * Use AmberTools 'parmchk' to guess missing GAFF parameters.
# * Create vacuum system in AmberTools 'leap' and write prmtop/inpcrd for AMBER.
# 
# Written by John D. Chodera <jchodera@gmail.com> 2008-01-31
#=============================================================================================

#=============================================================================================
# Imports
#=============================================================================================

import commands
import os    

#=============================================================================================
# CHANGELOG
#============================================================================================

#=============================================================================================
# KNOWN BUGS
#============================================================================================

#=============================================================================================
# PARAMETERS
#=============================================================================================

clearance = 5.0 # clearance around solute for box construction, in Angstroms
        
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

def createAmberSystem(mol2_filename, prefix, charge_model='bcc', net_charge=0, verbose=True):
    """
    Create AMBER prmtop/inpcrd files from a small molecule mol2 file, parameterizing with GAFF and AM1-BCC.

    ARGUMENTS
      mol2_filename (string) - input Tripos mol2 filename
      prefix (string) - prefix for .prmtop and .inpcrd files

    OPTIONAL ARGUMENTS
      verbose (boolean) - if True, extra debug information will be printed (default: True)
      charge_model (string) - the charge model requested from AmberTools 'antechamber' (default: 'bcc' for AM1-BCC)
      net_charge (int) - total charge (default: 0)

    NOTES
         
    """

    # Run antechamber to assign GAFF atom types.
    if verbose: print "Running antechamber..."
    gaff_mol2_filename = prefix + '.gaff.mol2'
    command = 'antechamber -i %(mol2_filename)s -fi mol2 -o %(gaff_mol2_filename)s -fo mol2 -c %(charge_model)s -nc %(net_charge)d > %(prefix)s.antechamber.out' % vars()
    if verbose: print command
    output = commands.getoutput(command)
    if verbose: print output

    # Generate frcmod file for additional GAFF parameters.
    solute_frcmod_filename = prefix + '.frcmod.solute'
    command = 'parmchk -i %(gaff_mol2_filename)s -f mol2 -o %(solute_frcmod_filename)s' % vars()
    if verbose: print command
    output = commands.getoutput(command)
    if verbose: print output

    # Run LEaP to generate topology / coordinates.
    solute_prmtop_filename = prefix + '.solute.prmtop'
    solute_crd_filename = prefix + '.solute.crd'
    solute_off_filename = prefix + '.solute.off'
    
    tleap_input_filename = prefix + '.leap.in'
    tleap_output_filename = prefix + '.leap.out'
    contents = """
# Load GAFF parameters.
source leaprc.gaff

# load antechamber-generated additional parameters
mods = loadAmberParams "%(solute_frcmod_filename)s"

# load solute
solute = loadMol2 "%(gaff_mol2_filename)s"

# check the solute
check solute

# report net charge
charge solute

# save AMBER parameters
saveAmberParm solute "%(solute_prmtop_filename)s" "%(solute_crd_filename)s"

# write .off file
saveOff solute "%(solute_off_filename)s"

# exit
quit
""" % vars()
    write_file(tleap_input_filename, contents)
    command = 'tleap -f %(tleap_input_filename)s > %(tleap_output_filename)s' % vars()
    output = commands.getoutput(command)

    return

#=============================================================================================
# MAIN
#=============================================================================================

# mol2 files to be converted
mol2_filenames = ['benzene.mol2', 'cyclohexane.mol2', '1,3-diethylbenzene.mol2']

for mol2_filename in mol2_filenames:    
    # Parameterize system for AMBER.
    import os.path
    [prefix, extension] = os.path.splitext(mol2_filename)
    createAmberSystem(mol2_filename, prefix)

