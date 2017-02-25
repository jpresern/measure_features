# measure_features
This is simple Python3 tool to measure (electron) micrographs.

1) Open file: Script opens image file through file dialog or startup
parameters. If there is accompanying .txt file containing 
experimental metadata (at the moment only JEOL micrographs are 
supported), it tries to fetch the pixel size and scale bar 
size. 

Alternatively, image can be via startup parameters (below).

Startup parameters:
--f ... opens file specified. Example:
./measure_features.py --f ./samples/Vzorec_120_005.tif

2) Script offers measuring of two qualities, accessible via console 
interaction:
1) area measurement
2) length measurement

Ad. 1.: Click on the edges of the area you would like to measure. Accept
the measurements. Inside of the "Area measurement" is a "feature 
counter", allowing user to count elements inside the area. Density is 
computed.

Ad. 2.: Click two points between which you would like to measure 
distance.

Saving results: A file dialog opens and prompts for the file name. 
Measurements are saved as:
1) .csv
2) .pdf

Warning: script overwrites existing files with the same name.