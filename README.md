# calcium_imaging_analizer
Pyqt5 GUI to analyze processed calcium imaging experiments combined with EEG recordings. 
![GUI](gui.png)

## Inputs:
The program requires  a sync EDF file, a folder with filters, a folder with traces (or a csv), a folder with mat or txt files describing the duration of every short movie used for concatenation and a folder with the sleep scoring

## Outputs:
The outputs are images of the cells color coded by state preference, figure with traces by state for cells that show preference, number of cells with positive or negative preference for each state and their level of activity by state. 
![traces](ca_traces.png)

![summary outpuy](output.png)

## Installation
Install python >=3.7

type:
pip -r install requirements.txt

To execute:
python caimgr2.py

if you have pyhton2 as default, replace python by python3

