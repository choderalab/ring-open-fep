#!/bin/tcsh

setenv GROMACS_HOME $HOME/code/gromacs/install
setenv PATH ${GROMACS_HOME}/bin:${PATH}

# Clean up.
rm -f confout.gro morse.tpr ener.edr dhdl.xvg traj.xtc 

# Set up calculation.
grompp -f morse.mdp -p morse.top -c morse.gro -o morse.tpr

# Run.
mdrun -v -s morse.tpr


