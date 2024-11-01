# aPEER Code

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

# 4. Contact

Contact Andrew Deonarine (andrewdeomd@gmail.com) for more information.
