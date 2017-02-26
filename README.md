# measure_features

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

Dependencies: 
-------------
* Python >= 3.4 
* Python packages
    * Matplotlib >= 2.0
    * Numpy >=1.11
    * Pandas >=19.2.

Open file: 
----------
Script opens image file through file dialog or startup
parameters. If there is accompanying .txt file containing 
experimental metadata (at the moment only JEOL micrographs are 
supported), it tries to fetch the pixel size and scale bar 
size. Alternatively, image can be via startup parameters (below).

Startup parameters:
--f ... opens file specified. Example:
./measure_features.py --f ./samples/Vzorec_120_005.tif

Measuring:
----------
script offers measuring of two qualities, accessible via console interaction.

* Area measurement: Click on the edges of the area you would like to 
measure. Accept the measurements. Inside of the "Area measurement" is a
"feature counter", allowing user to count elements inside the area. 
Density is computed.

* Length measurement: Click two points between which you would like to 
measure distance.

Saving results: 
---------------
A file dialog opens and prompts for the file name and location. 
Saving produces two files with the same name:
* .csv, which contains results, measurements and all other info
required for restoring the measuring session (not implemented yet)
* .pdf, which contains image with measured regions, elements drawn
in. .pdf is layered and can be further edit in varios vector 
manipulating software.

Warning: script overwrites existing files with the same name **without**
prompt.

Acknowledgments: 
----------------
Development was supported by ARRS Z4-5518 
(to BP) and ARRS P4-0133 (JP).
