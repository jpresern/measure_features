# measure_features
This is simple python3 tool to measure (electron) micrographs.

Script opens image file. If there is accompanying .txt file containing 
experimental settings, it tries to fetch the pixel size and scale bar 
size. 

Startup parameters:
--f ... opens file specified. Example:
./measure_features.py --f ./samples/Vzorec_120_005.tif

Script offers measuring of two qualities, accessible via console 
interaction:
1) area measurement
2) length measurement

Ad. 1.: Click on the edges of the area you would like to measure. Accept
the measurements. Inside of the "Area measurement" is a "feature 
counter", allowing user to count elements inside the area. Density is 
computed.

Ad. 2.: Click two points between which you would like to measure 
distance.

Measurements are saved as:
1) .csv
2) .pdf