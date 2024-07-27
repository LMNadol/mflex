## Magnetohydrostatic Field Line Extrapolation Package (MHSFLEX)

#### Lilli Nadol
#### lmn6@st-andrews.ac.uk

#### Introduction 

The following manual describes the set-up of, and how to use, the Magnetohydrostatic Field Line Extrapolation (MFLEX) Python package to extrapolate 3D field line structures from photospheric line-of-sight observations.

The MFLEX package consist of a data reading and preparing, a modelling and a visualisation module. Each of these modules will be described individually later in more detail. All these codes are run in cartesian coordinate system on even, non-staggered grids. 

The data reading and preparation module converts data files into the rquired data class. The main part of the package, the B field modelling module, calculates the 3D magnetic field vector components from data. Generally it follows the model of Neukirch and Wiegelmann (2019) with the change of using an asymptotic approximation of the defining ordinary differential equation instead of the originally presented solution. The main differences and more details on the derivation of the used set of equations can be found in Nadol and Neukirch (2024). The MFLEX package also includes codes for extrapolation of magnetic field lines using fieldline3D.py from the Magnetic Skeleton Analysis Tools (MSAT) package written by Ben Williams [see Williams PhD Thesis 2018, https://github.com/benmatwil/msat]. The plotting module finds and displays field lines into a 3D space. These three modules in turn can be used to fully identify the field line structure above a given photospheric region.

These codes have been written in Python 3 2023 standard. The following standard Python modules are used:
numpy
scipy
matplotlib
numba
astropy
sunpy

#### Quick Start for Solar Orbiter Archive Data

1.  First you need to set-up data on your computer. Solar Orbiter Archive data files can be downloaded at      
https://soar.esac.esa.int/soar/#search. It is necessary to use _blos.fits files for magnetic field line-of-sight observations.
2.  Import the package by using: import mflex
3.  Convert observational data file into required data class by running: mflex.load.read_fits_soar(path_to_blos_fits_file)
4.  Specify MHS model parameter. (Hopefully soon, use internal parameter optimisation.)
5.  Run the 3D magnetic field modelling: mflex.model.field.routine magnetic_field(parameters)
6.  Visualise the magnetic field line skeleton: mflex.plot.plot_fieldlines_grid(parameters)
7.  Visualise the magneticfield plasma parameter variations: mflex.plot.plot_plasma_parameters(parameters)

If any problems arise or for customisation please refer to the main part of the manual. 

#### Preparing the Data

Describe file reading routines and interactive input that is needed from user. 

All the codes require a 2D vector field, from now on called the background or photospheric magnetic field. The data must be stored in a .fits file with a header structured in the SOAR standard. An example of a suitable data file can be found in /data as solo_L2_phi-hrt-blos_20220307T000609_V01.fits. 

Describe header reading and writing routine here. 

#### Main Codes

The 3D magnetic field modelling and visualisation modules are the core of the MFLEX package. The main codes are the field component computer magnetic_field() and the field line extrapolator plot_fieldlines_grid(). (Wow these are all shit names.) 

They must be run in the above order or they will not work as the next code depends on the output from the previous code. The codes work in grid coordinates (a coordinate is given between 0 and nresol_x in the x direction for example). The output from each is in length scale normed real coordinates (a coordinate is given between 0.0 and 1.0 for the x direction and between 0.0 and nresol_y/nresol_x in the y direction, if the x direction is the shorter side for example). 

Describe how choice of pixelsize_z influences results and run time. 

Make choice of upper boundary (currently set at 10Mm) and transitional region (centre currently set at 2Mm) possible. 

##### Seehafer Mirroring 

##### Fast Fourier Transformation

##### Parameters 

Description of parameters for each routine here. Will be endless.  

### Visualisation Routines

Describe how foot points are chosen and how grid spacing can be influenced. For description of fieldline3D interpolation look at Ben Williams repository. 

### Examples

#### Creating boundary data from an analytical expression

Describe how a data set can be created from an analytical expression as seen in Von Mises distribution dipole in Neukirch and Wiegelmann (2019). Write routine for that. 
