[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.322674.svg)](https://doi.org/10.5281/zenodo.322674)

# Measure Features

(c) 2016 - 2017 Barbara Piskur (Slovenian Forestry Insitute) and 
Janez Presern (Agricultural Institute of Slovenia)

This is simple Python3 tool to measure (electron) micrographs. The 
development was initiated after realisation that simple and simply 
extendable tool for work with rust (Pucciniales) electron micrographies 
is needed. Tool allows measuring of various qualities and quantities, 
like density and surface area on the image. 
 
For example, as is it in the case of rusts (Pucciniales), density of 
thorns on the selected surface can be different from species to species. 
The tools assists with measuring and provides .csv with measurements and
.pdf with picture where the estimations were taken from.

Dependencies
------------
* Python >= 3.4 
* Python packages
    * Matplotlib >= 2.0
    * Numpy >=1.11
    * Pandas >=19.2.

Open image file
---------------
Script opens image file through file dialog or startup
parameters. If there is accompanying .txt file containing 
experimental metadata (at the moment only JEOL micrographs are 
supported), it tries to fetch the pixel size and scale bar 
size. Alternatively, image can be via startup parameters (below).

Startup parameters:
--f ... opens file specified. Example:
./measure_features.py --f ./samples/Vzorec_120_005.tif

Open .csv file
--------------
Alternatively, one can open .csv file with old measurements. Image gets
displayed automatically together with old "points-of-interest" drawn onto
the image. User can either continue with his/her measurements or close 
old results untouched.

Measuring
---------
Script offers measuring of two qualities, accessible via console interaction.

* Area measurement: Click on the edges of the area you would like to 
measure. Accept the measurements by pressing ENTER or middle mouse button.
After area selection, there is possibility to count features (and density)
inside the selected area. Density is computed and stored automatically. 
Units are square micrometers. Density unit is number of features per square
micrometer.

* Length measurement: Click two points between which you would like to 
measure distance. After completition, user will be asked whether he/she wishes
to continue with length measurements or quit length measurements. Units are
square micrometers.

Saving results
--------------
A file dialog opens and prompts for the file name and location. 
Saving produces two files with the same name:
* .csv, which contains results, measurements and all other info
required for restoring the measuring session (not implemented yet)
* .pdf, which contains image with measured regions, elements drawn
in. .pdf is layered and can be further edit in varios vector 
manipulating software.

Warning: script overwrites existing files with the same name **without**
prompt.

Acknowledgments 
---------------
Development was supported by ARRS Z4-5518 
(BP) and ARRS P4-0133 (JP).
