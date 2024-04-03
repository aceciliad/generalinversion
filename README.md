# InversionNonAdiabatic
# generalinversion

General inversion set up in INV1/inv_run.py, divided in different classes that:
1. Read input parameters
2. Model interior structure of Mars
3. Runs forward for seismic data, i.e., receiver functions, travel times, normal modes, etc.
4. Compute misfits and invert for interior structure and event locations

Runs the prediction of travel times in parallel to account for the "two" models employed to produced phases when we consider a liquid layer at the bottom of the mantle. 

It can be run in independent chains using the run script with slurm in a cluster. However, if the old reflectivity code is used (crfl_sac_mars), it requires an independent directory for each run, which makes things ugly and do not allow for parallel running. 

Many things could be enhanced, but they have evolved through the InSight mission to fit the weekly demand :)
