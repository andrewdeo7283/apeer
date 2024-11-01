# aPEER

By: Andrew Deonarine

Date: Oct 28, 2024

# 1. Introduction

- This is a basic implementation of the aPEER algorithm and how the figures were created
- The basic steps are:

1. Pollution and chronic disease prevalence rates are loaded - this information is calculated at the county level (assigned to county FIPS codes)
2. Pairs of county-level pollution measures are then clustered using normalization, PCA, and K-means clustering, resulting in clusters of counties
3. Groups of counties with high rates of chronic disease prevalence are identified
4. The groups of pollution counties in #2 are then compared to the groups of counties with high disease prevalence in #3 using the Jaccard Correlation Coefficient (J)
5. The highest pollution-disease associations are identified by the pairs with the highest J values

# 2. Code Overview

- apeer_main.py - contains the code for producing the figures
- apeer_lib.py - contains the code for functions and calculations

# 3. Installation

- you will need to download AirToxScreen, EJSCREEN, and CDC PLACES Data
- Required libraries: matplotlib, pandas, sklearn, seaborn, numpy, descartes

- ## STEP 1: you will need to download the following libraries:
```
seaborn
pandas
sklearn
matplotlib
numpy
descartes
openpyxl
```

## STEP 2: download the data files to perform the analysis:

```
EJSCREEN_2021_USPR_Tracts.csv
https://drive.google.com/file/d/1siIosFHP9JK8VsjY6DHcoQ8Kd_llHF6t/view?usp=sharing

PLACES__Census_Tract_Data__GIS_Friendly_Format___2021_release.csv
https://drive.google.com/file/d/1ftVtQEAFJ3MqLWD-eh3_ipCOy5H2GAoA/view?usp=sharing

PLACES__County_Data__GIS_Friendly_Format___2021_release.tsv
https://drive.google.com/file/d/1ftVtQEAFJ3MqLWD-eh3_ipCOy5H2GAoA/view?usp=sharing

2018_Toxics_Ambient_Concentrations.updated.tract.tsv
https://drive.google.com/file/d/1wcBGlTlS2ZJSgk2_BQlV4SdBofHW_yLF/view?usp=sharing

2018_Toxics_Ambient_Concentrations.updated.county.tsv
https://drive.google.com/file/d/1KnHRpmv2Z_ee6nDfN-agO95qdmit20Mx/view?usp=sharing

GeoJSON File - Tracts:
https://drive.google.com/file/d/1v0uOkMNQJDr1F2UxN_pUbvcafLw2GqAw/view?usp=sharing

GeoJSON File - Counties:
https://drive.google.com/file/d/1T1djNPuDqTkSlD2kRyhHw3coxuBjgLpx/view?usp=sharing

```

## STEP 3: now run the script using the command:

```
python3 apeer_main.py
```

- you will need to set the initial true/false flags to run different pieces of code
- NOTE: running pairwise pollution analysis in the calculations may take a long time depending on your computer speed

# 4. Previous Code

- the previous code and simplified implementation can be found in the aug2024 directory

# 5. Contact

Contact Andrew Deonarine (andrewdeomd@gmail.com) for more information.
