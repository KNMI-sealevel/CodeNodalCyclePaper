# Code and data for "The effect of the 18.6-year lunar nodal cycle on steric sea level changes" (2023)
Use tide gauge data and temperature and salinity data to study the effect of the nodal cycle on steric changes in sea level. 

This is the code supporting Bult et al. 2023, [to submit to Geophysical Research Letters]
This repository can be cited using ...

The analysis is organized in three steps:
1. *prepare_atmospheric_data.ipynb* explains how to download the input data, preprocess the data and export to .csv files. 
2. *tide&wind.ipynb* performs the statistical modelling and exports the results to .csv files, the script needs to be re-run for each individual tide gauge station. The script also calculates the error in the determination of the nodal cycle in the tide gauge data per station. 
3. *Nodalcycle_steric.ipynb* makes the final figures as shown in the manuscript. Some code lines are repeated to 

The analysis of the steric changes in sea level datasets as described in the supplementary material is also in *Nodalcycle_steric.ipynb*. 

### Reproducability
Using a conda environment with the most recent version of numpy, pandas, cartopy and xarray should suffice to run the code in all three notebooks. 
In the notebooks, only the path needs to be adjusted if the original structure of the folders is preserved. 

### Data provenance
- Tide gauge data (PSMSL)
- Reanalysis for wind (20CR, ERA5)
- Temperature and salinity data (EN4 & IAP)