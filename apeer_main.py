######################
### aPEER Analysis ###
######################

# Updated By: Andrew D
# Date: Oct 28, 2024

#The basic steps to the aPEER algorithm are:
#1. Pollution and chronic disease prevalence rates are loaded - this information is calculated at the county level (assigned to county FIPS codes)
#2. Pairs of county-level pollution measures are then clustered using normalization, PCA, and K-means clustering, resulting in clusters of counties
#3. Groups of counties with high rates of chronic disease prevalence are identified
#4. The groups of pollution counties in #2 are then compared to the groups of counties with high disease prevalence in #3 using the Jaccard Correlation Coefficient (J)
#5. The highest pollution-disease associations are identified by the pairs with the highest J values

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
from collections import defaultdict
import pandas as pd
from descartes import PolygonPatch
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook 
import matplotlib.image as image 
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import csv
import shapefile
from json import dumps
import json
import os, sys
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize
from sklearn.cluster import KMeans
from sklearn.cluster import FeatureAgglomeration
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
from matplotlib.pyplot import gcf
import networkx as nx

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from numpy import loadtxt
from xgboost import XGBClassifier, plot_tree, cv
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#import statsmodels.api as sm

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

import apeer_lib

light_gray = "#f5f5f5"
fore_color = "#b53737"

# Disease List
disease_list = ["COPD", "Hypertension", "Asthma", "Diabetes", "Depression", "Arthritis", "Cancer", "Coronary Heart Disease", "Renal Disease", "Obesity", "Stroke"]

# Part 0 - Load and Process Data
make_geojson = False
make_airtox = False
load_geojson = True
quick_load = True

# Sensitivity Analysis
GetCISensAnalysis = False
TimeseriesMaps = False

calc_stats = False
FindPercentileCutoff = False
makeRefMaps = False
makePollMaps = False
calcAirToxMaps = False
makeDiseaseMaps = False

# Calculate Jaccard Indices
calcJaccardAirTox = False

# Get top 10 Jaccard Indices
GetTopMaps = False

# Plot Networks
MakeAirToxNetwork = False

# Get Regressions for merged data
MakeMergeRegressions = False

# Assemble Merged Maps
MakeAssembledMergeMaps = False

# Generate the Maps
GenerateTopPollutionMaps = False
GenerateTopAirToxMaps = False

# Put on PDF Figures
MakeAirToxPDF = False
MakePollutionPDF = False

MakeFigure2 = False
MakeFigure3 = False
MakeFigure4 = False
MakeFigure5 = False # Rerun on Oct 27, 2024
MakeFigure4b = False # Rerun on Oct 27, 2024
MakeSuppFigs = False

RunDensityPlots = False

# Find Top Pollutants
FindTopPollutants = False

# Do XGBoost Predictions
BaselineRegression = False
RunXGBoost = False
makeEJSCREENMaps = False

# Create Maps
MakeFinalMaps = False

# Create Figure
MakeMapGraph = False
MakeFigure2CorrMap = False
MakeROCFigure = False

DisplayBestMaps = False
ClusterNetworkData = False
makeNetworkPollutionMaps = False

DisplayReferenceMaps = False
GraphFig4BarChart = False
GetMaxJaccardEJScreen = False

# Supplemental Figures
PopScatter = False
RunMoranLisa = False
CountStrokeBelt = False
GetSupplementalFigs = False
GetSuppAUCTable = False
CalibrationTable = True # Rerun on Oct 27, 2024
BaselineRegressionTable = False
MakeViolins = False
GetSuppEJNetworks = False

rpath = "d:\\apeer"
#mappath = rpath + "\\maps"

if make_geojson == True:
	apeer_lib.GetTractGeoJSON()

if make_airtox == True:
	apeer_lib.CalculateAirToxData()
	apeer_lib.GetCountiesByCI()

if load_geojson == True:
	json_data = apeer_lib.LoadMapJSON("tract")
	json_data_county = apeer_lib.LoadMapJSON("county")

#####################################
### Part 1 - Statistical Analysis ###
#####################################

if calc_stats == True:

	merge_data_counties = apeer_lib.MergeStrokeData("county")
	merge_data_tract = apeer_lib.MergeStrokeData("tract")
	
	# merge in life expectancy data
	merge_data_counties = apeer_lib.LoadLifeExpectancy(merge_data_counties, "none", 0, "merge")
	merge_data_tract = apeer_lib.LoadLifeExpectancy(merge_data_tract, "none", 0, "merge")
	
	# merge in stroke mortality data
	merge_data_counties = apeer_lib.LoadStrokeMort(merge_data_counties, "none", 0, "merge")
	merge_data_tract = apeer_lib.LoadStrokeMort(merge_data_tract, "none", 0, "merge")
	
	# merge in median age by county
	merge_data_counties = apeer_lib.Load2019LifeExpect(merge_data_counties)
	print(merge_data_counties)

	finalcounties, finaltracts = apeer_lib.GetMainLists()
	merge_data_tract = merge_data_tract[merge_data_tract.index.isin(finaltracts)]
	merge_data_counties = merge_data_counties[merge_data_counties.index.isin(finalcounties)]
	print("Number of tracts: " + str(len(merge_data_tract.index)))
	print("Number of counties: " + str(len(merge_data_counties.index)))
	
	# Part 1 - Calculate Statistics for CDC Data
	pcutoffs = [0.6, 0.7, 0.8, 0.9]
	odata1 = merge_data_counties.describe(percentiles=pcutoffs)
	odata1 = odata1.T

	# Now calculate the number of counties at or above the 60th, 70th, 80th percentile
	for tcol in merge_data_counties.columns:
		for ncutoff in pcutoffs:
			tcutoff = merge_data_counties[tcol].quantile(ncutoff)
			tcnt = 0
			for titem in merge_data_counties.index:
				tval = float(merge_data_counties.loc[titem, tcol])
				if tval >= tcutoff:
					tcnt += 1
			tlabel = 'N' + str(ncutoff)
			odata1.loc[tcol, tlabel] = tcnt

	odata1.to_csv("table1_county.tsv", sep="\t")
	
	odata2 = merge_data_tract.describe(percentiles=pcutoffs)
	odata2 = odata2.T
	odata2.to_csv("table1_tract.tsv", sep="\t")
	
	# save table for quick loading
	print("Saving Quick Load")
	print(merge_data_counties)
	merge_data_counties.to_csv("quickload_counties.tsv", sep="\t")

	# make and save AirToxScreen Data 
	chemical_data = apeer_lib.LoadChemicalData("county")
	finalcounties, finaltracts = apeer_lib.GetMainLists()
	#merge_data_tract = merge_data_tract[merge_data_tract.index.isin(finaltracts)]
	chemical_data = chemical_data[chemical_data.index.isin(finalcounties)]

	# merge data - remove Air Toxic Cancer, Air Resp Cancer, Diesel PM - these are in AirToxScreen
	pollution_columns = ["Lead Paint", "Traffic Proximity", "Wastewater discharge", "Superfund Proximity", "RMP Facility Proximity", "Hazardous Waste Proximity", "Ozone", "Particulate Matter 2.5", "Underground Storage Tanks"]	
	demographic_columns = ["Percent Minority", "Percent Low Income", "Percent Less than HS Education", "Percent Linguistic Isolation", "Percent Over 64 yrs", "Percent Unemployed"]
	democols = ["minorpct", "lowincpct", "lesshspct", "lingisopct", "over64pct", "unemppct"]

	merge_pollution = list(chemical_data.columns)
	merge_pollution = merge_pollution + pollution_columns.copy()

	merged_pollution_data = chemical_data
	for tcol in pollution_columns:
		merged_pollution_data[tcol] = merge_data_counties[tcol]
			
	print("Merged Pollution list: " + str(pollution_columns))

	print("Saving counties")
	print(chemical_data)
	chemical_data.to_csv("quickload_airtox_merge_counties.tsv", sep="\t", index_label = 'fips')

	countycnt = 0
	for tfips in finalcounties:
		countycnt += 1
		
if quick_load == True:
	merge_data_counties = pd.read_csv("quickload_counties.tsv", sep="\t", converters = {'fips': str})
	merge_data_counties.set_index('fips', inplace=True)
	#chemical_data = pd.read_csv("quickload_airtox_counties.tsv", sep="\t", converters = {'fips': str})
	chemical_data = pd.read_csv("quickload_airtox_merge_counties.tsv", sep="\t", converters = {'fips': str})
	chemical_data.set_index('fips', inplace=True)

############################################################
### Supplemental - Create Chronic Disease Reference Maps ###
############################################################

# Identify counties with high rates of chronic disease prevalence (>= 60% percentile, >= 70% percentile, >= 80% percentile, >= 90% percentile)
# for each of the 12 chronic diseases: stroke, asthma, arthritis, etc.

if makeRefMaps == True:

	quant_cutoffs = [0.6, 0.7, 0.8, 0.9]
	for tquant in quant_cutoffs:
		for tdisease in apeer_lib.disease_columns:
			bfile = "lancet.canonical." + tdisease + "." + str(tquant) + ".binarymap.png"
			ofile = "lancet.canonical." + tdisease + "." + str(tquant) + ".binarymap.tsv"
			tdata = merge_data_counties[[tdisease]]
			maplist = apeer_lib.GetReferenceMap(tdata, ofile, tquant)
			apeer_lib.ShowBinaryMap(maplist, bfile, json_data_county, light_gray, "blue")
		
	# stroke map
	quant_cutoffs = [0.6, 0.7, 0.8, 0.9]
	for tquant in quant_cutoffs:
		bfile = "lancet.canonical.stroke_data." + "." + str(tquant) + ".binarymap.png"
		ofile = "lancet.canonical.stroke_data." + "." + str(tquant) + ".binarymap.tsv"
		maplist = apeer_lib.LoadStrokeMort(merge_data_counties, ofile, tquant, "threshold")
		apeer_lib.ShowBinaryMap(maplist, bfile, json_data_county, light_gray, "blue")
		

#######################################################################################
### Prep Calculations Part 1 - Create Clusters of Counties from Pairs of Pollutants ###
#######################################################################################

# Full Pairwise AirToxScreen Analysis
# This will calculate the clustered pairwise pollution maps (resulting in groups of counties) for comparison
# to the sets of high-prevalence chronic disease rate counties for each of the 12 chronic diseases
if calcAirToxMaps == True:

	# completely merge all data, normalize together
	#mergeframe = pd.concat([merge_data_counties, chemical_data], axis=1)
	chemical_data = chemical_data.replace(np.nan, 0)
	norm_chemical_data = normalize(chemical_data, axis=0)
	chemical_data = pd.DataFrame(norm_chemical_data, columns=chemical_data.columns, index=chemical_data.index)

	tcnt = 1
	for pairx in range(0, len(chemical_data.columns)):
		print("Running pairwise analysis for " + chemical_data.columns[pairx])
		for pairy in range(pairx + 1, len(chemical_data.columns)):
			for nclust in range(2, 6):
				tfile = apeer_lib.mappath + "\\" + "lancet_airtox_pair_merge." + chemical_data.columns[pairx] + "." + chemical_data.columns[pairy] + ".county_cluster." + str(nclust)
				tfile = tfile.replace(" ", "")
				tfile = tfile.replace(",", "_")
				tfile = tfile.replace("-", "_")
				cfile = tfile + '.cluster.tsv'
				#print("Pairwise Pollution File: " + tfile)
				#if not os.path.exists(cfile):
				#print(str(tcnt) + ". Running analysis for " + cfile)
				tdata = chemical_data[[chemical_data.columns[pairx], chemical_data.columns[pairy]]]
				apeer_lib.ShowTractClustersOnMap(tdata, nclust, tfile, "county", json_data_county, showmap=False, forceBinary=False)
				tcnt += 1


#########################################################################################################
### Prep Calculations Part 2 - Calculate Jaccard Corr Coeff. Between Pollution & Chronic Disease Maps ###
#########################################################################################################

# This code will go through all pairwise combinations of pollutants and calculate the Jaccard coefficient value
# J for that pollution map versus a given chronic disease

if calcJaccardAirTox == True:

	print("Calculating Jaccard Values...")

	countycnt = 3141
	#filelist = apeer_lib.BuildMergeFileList()

	# Main Filetable
	main_filetable = defaultdict(lambda: defaultdict(str))
	main_filetable = apeer_lib.BuildFileList()

	#output_file = "lancet_merge_figure3.tsv"
	print("Loading Reference Maps")
	outdata = defaultdict(str)
	output_files = defaultdict(str)
	refbelt = defaultdict(lambda: defaultdict(int))
	disease_cutoffs = [0.6, 0.7, 0.8, 0.9]
	for disease_percentile in disease_cutoffs:

		#filelist = apeer_lib.BuildFileList(disease_percentile)
		output_file = "lancet_merge_" + str(disease_percentile) + "_figure3.tsv"
		output_files[output_file] = ''

		# clear output file
		f = open(output_file, "w")
		f.write("")
		f.close()
		
		for ofile in main_filetable[disease_percentile]:
			print("Processing: " + ofile)
			rawdata = apeer_lib.LoadClusterData(ofile)
			for tid in rawdata:
				if rawdata[tid] == "1":
					refbelt[ofile][tid] = 1

	#chemical_data = apeer_lib.LoadChemicalData("county")
	for pairx in range(0, len(chemical_data.columns)):

		#print("Processing: " + ofile + ": " + chemical_data.columns[pairx])
		print("Processing: " + chemical_data.columns[pairx])
		myobj = datetime.now()
		print(myobj)

		for pairy in range(pairx + 1, len(chemical_data.columns)):	
			for nclust in range(2, 6):

				# Calculate Jaccard Values
				tfile = apeer_lib.mappath + "\\" + "lancet_airtox_pair_merge." + chemical_data.columns[pairx] + "." + chemical_data.columns[pairy] + ".county_cluster." + str(nclust) + ".cluster.tsv"
				tfile = tfile.replace(" ", "")
				tfile = tfile.replace(",", "_")
				tfile = tfile.replace("-", "_")

				clustdata = apeer_lib.LoadClusterData(tfile)
				#rawdata = apeer_lib.LoadClusterData(ofile)
				for disease_percentile in disease_cutoffs:

					output_file = "lancet_merge_" + str(disease_percentile) + "_figure3.tsv"

					# Compare to reference maps
					for ofile in main_filetable[disease_percentile]:
						
						#refbelt = defaultdict(int)
						#for tid in rawdata:
						#	if rawdata[tid] == "1":
						#		refbelt[tid] = 1
						#clustdata = apeer_lib.LoadClusterData(tfile)
						tjaccard, tpval = apeer_lib.CalcJaccard(clustdata, refbelt[ofile], countycnt)
						#f.write(ofile + "\t" + chemical_data.columns[pairx] + "\t" + chemical_data.columns[pairy] + "\t" + str(nclust) + "\t" + str(tjaccard) + "\t" + str(tpval) + "\n")
						output_files[output_file] += ofile + "\t" + chemical_data.columns[pairx] + "\t" + chemical_data.columns[pairy] + "\t" + str(nclust) + "\t" + str(tjaccard) + "\t" + str(tpval) + "\n"

		# Writing out files...
		for output_file in output_files:
			f = open(output_file, "a")
			f.write(output_files[output_file])
			output_files[output_file] = ''
			f.close()

##########################################################
### Figure 2 - Create Pairwise Correlation Map Figures ###
##########################################################

# Now that we've generated pairwise pollution maps, we calculate the 
# Jaccard correlation coefficients between the sets of counties defined in 
# pairwise pollution maps and sets of counties with high rates of chronic
# disease prevalence

if MakeFigure2 == True:

	apeer_lib.GetCorrelationMap(chemical_data, json_data_county)


################################
### Figure 3 - Plot Networks ###
################################

# Plot a Network from the pairwise Jaccard values - this is done for each chronic disease
# creating 12 networks.

# For the networks we have created a set of pollution maps from pairs of pollutants 
# (ex. methanol/formaldehyde, methanol/acetaldehyde, etc)
# This creates thousands of pollution maps from pairwise pollutant combinations. Then the Jaccard correlation is calculated between
# each pairwise pollution map and chronic disease (ie. Jaccard calculates how many counties are in common between 
# the pairwise pollution map and the high prevalence chronic disease counties. This creates a pairwise matrix of Jaccard
# values for all the pairwise combinations of pollutants - this is done for each chronic disease (12 matrices / networks)

if MakeAirToxNetwork == True:

	apeer_lib.PlotJaccardNetwork(chemical_data, "lancet_merge_figure3.tsv", "airtox_merge")
	stroke_cutoffs = [0.6, 0.7, 0.8, 0.9]
	for tcutoff in stroke_cutoffs:
		output_file = "lancet_merge_" + str(tcutoff) + "_figure3.tsv"
		apeer_lib.PlotJaccardNetwork(chemical_data, output_file, "airtox_merge")

# Calculate Baseline/Regressions

###########################################
### Figure 3 - Plot Regression Networks ###
###########################################

# Calculate the elastic net and random forest regressions for Figure 3
# In these models:
# y = dependent variable: if a county is a high prevalence chronic disease (y=1) or not (y=0)
# x = independent variables: these are pollutants, socioeconomic values, etc.

if MakeMergeRegressions == True:

	# get data
	norm_chemical_data = apeer_lib.NormalizeData(chemical_data)

	stroke_cutoffs = [0.9, 0.8, 0.6]
	#stroke_cutoffs = [0.7]
	xpath = "d:\\apeer\\networks"
	ndisease_list = disease_list.copy() + ["STROKE_MORT", "STROKE_BELT"]
	#ndisease_list = ["STROKE_BELT"]
	for tcutoff in stroke_cutoffs:
		for tdisease in ndisease_list:
			chemlist = apeer_lib.GetHubList(tdisease, tcutoff)
			out_file = xpath + "\\" + tdisease + "_roc." + str(tcutoff) + ".merge.network.jpg"
			param_out = apeer_lib.MakeMergeROCFigure(merge_data_counties, norm_chemical_data, chemlist, tdisease, tcutoff, out_file)
			f = open(tdisease + "_" + str(tcutoff) + "_xgboost_params.tsv", "w")
			f.write(param_out)
			f.close()
		
##############################################
### Figure 4a - Top disease pollution maps ###
##############################################

# This code retrieves the pollution-disease associations with the highest Jaccard correlation coefficient

if GetTopMaps == True:

	# get data
	norm_chemical_data = apeer_lib.NormalizeData(chemical_data)

	# get p-value threshold
	mainfile = "lancet_merge_figure3.tsv"
	pval_cutoff = apeer_lib.GetPValCutoff(mainfile)

	# output_file = "lancet_figure2_top_pollutants.tsv"	
	print("Loading Jaccard data")
	score_table = defaultdict(lambda: defaultdict(float))
	cluster_table = defaultdict(lambda: defaultdict(float))
	with open(mainfile, "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			bdisease = ldata[0]
			tclust = int(ldata[3])
			#tdesc = ldata[1] + '|' + ldata[2] + "|" + ldata[3]
			tdesc = ldata[1] + '|' + ldata[2]
			tvjaccard = ldata[4]
			if tvjaccard == "":
				tvjaccard = "1"
			tjaccard = float(tvjaccard)
			tpval = float(ldata[5])
			if tpval <= pval_cutoff:
				if tdesc not in score_table[bdisease]:
					score_table[bdisease][tdesc] = tjaccard
					cluster_table[bdisease][tdesc] = tclust
				if tdesc in score_table[bdisease]:
					if tjaccard > score_table[bdisease][tdesc]:
						score_table[bdisease][tdesc] = tjaccard
						cluster_table[bdisease][tdesc] = tclust
							
	# now sort Jaccard scores - get top 10
	rank_cutoff = 5
	for bdisease in score_table:
		maxj = 0
		tcnt = 1
		for titem in sorted(score_table[bdisease], key=score_table[bdisease].get, reverse=True):

			# Load Reference map data
			if tcnt < 4:
				pollutants = titem.split('|')
				tclust = cluster_table[bdisease][titem]
				tline = str(tcnt) + "\t" + bdisease + "\t" + titem + "\t" + str(tclust) + "\t" + str(score_table[bdisease][titem])
				pairdata = norm_chemical_data[[pollutants[0], pollutants[1]]]
				tfile = "lancet_final_pairfile_" + bdisease + "_" + pollutants[0] + "_" + pollutants[1]
				tfile = tfile.replace(',', '')
				tfile = tfile.replace('-', '')
				tfile = tfile.replace(' ', '')
				print(tline)
				apeer_lib.ShowMatchingCluster(pairdata, cluster_table[bdisease][titem], tfile, json_data_county, bdisease)

			tcnt += 1

if MakeAssembledMergeMaps == True:

	# get top jaccard index values
	# get data
	chemical_data = chemical_data.replace(np.nan, 0)
	norm_chemical_data = normalize(chemical_data, axis=0)
	norm_chemical_data = pd.DataFrame(norm_chemical_data, columns=chemical_data.columns, index=chemical_data.index)

	#apeer_lib.CheckJaccardValues(chemical_data, json_data_county)
	apeer_lib.OptimizeAssemblyByJaccard(norm_chemical_data, json_data_county)

	# get top pollution pairs
	#top_pairs = apeer_lib.GetTopPollutionPair("lancet_merge_figure3.tsv", "all")	
	max_clust = 7
	apeer_lib.GetBestAssembledMaps(max_clust, norm_chemical_data, json_data_county)


##########################################################################
### Figure 4b - Clustering aPEER / elastic net / random forest results ###
##########################################################################

if MakeFigure4b == True:

	# do clustering for different cutoffs
	cutoff_list = [0.6, 0.7, 0.8, 0.9]
	for tcutoff in cutoff_list:
		apeer_lib.ClusterNetworks("cluster_merged_" + str(tcutoff) + "_network.png", tcutoff, "network")
		apeer_lib.ClusterNetworks("cluster_merged_" + str(tcutoff) + "_logistic.png", tcutoff, "logistic")
		apeer_lib.ClusterNetworks("cluster_merged_" + str(tcutoff) + "_xgboost.png", tcutoff, "xgboost")

	apeer_lib.MakeSuppClusterFigure()

	apeer_lib.MakeFigure5Assembly()	

###########################################
### Supplemental - Sensitivity Analysis ###
###########################################

# This code performs a sensitivity analysis - it will remove counties with largest confidence intervals 
# at the confidence interval width 95th, 97.5th, and 99th percentile cutoffs and recalculate the 
# Jaccard correlation coefficient to determine if the removal of values with high levels of 
# uncertainty (large confidence intervals) affects the Jaccard correlation coefficient value

if GetCISensAnalysis == True:
	
	# get CI data
	cidata = apeer_lib.GetCIData()
	main_filetable = apeer_lib.BuildFileList()
	output_files = defaultdict(str)
	disease_cutoffs = [0.6, 0.7, 0.8, 0.9]
	ci_cutoffs = [0.99, 0.975, 0.95, 0.9]
	
	# get exclusion lists for reference diseases, pollution
	countycnt = 3141
	cicols = {"STROKE_Crude95CI": "Stroke", "BPHIGH_Crude95CI": "Hypertension", "COPD_Crude95CI": "COPD", "DIABETES_Crude95CI": "Diabetes", "STROKE_MORT": "stroke_data."}
	pocols = {'FORMALDEHYDE': 'Formaldehyde', 'ACETALDEHYDE': 'Acetaldehyde', 'METHANOL': 'Methanol', 'BENZOAPYRENE': 'Benzoapyrene', 'GLYCOL ETHERS': 'GLYCOL ETHERS'}
	outdata = ""

	#for tcol in cidata.columns:
	#	print("CIDATA column: " + tcol)	
	#exit()

	for tcol in cicols:
		for ci_cut in ci_cutoffs:
			exclude_list = apeer_lib.GetCIExclusionList(cidata, tcol, ci_cut)
			for tquant in disease_cutoffs:
				refbelt = defaultdict(int)
				ofile = "lancet.canonical." + cicols[tcol] + "." + str(tquant) + ".binarymap.tsv"
				rawdata = apeer_lib.LoadClusterData(ofile)
				for tid in rawdata:
					if (rawdata[tid] == "1") and (tid not in exclude_list):
					#if (rawdata[tid] == "1"):
						refbelt[tid] = 1
				for pairx in pocols:
					for pairy in pocols:	
						for nclust in range(2, 6):
							tfile = apeer_lib.mappath + "\\" + "lancet_airtox_pair_merge." + pairx + "." + pairy + ".county_cluster." + str(nclust) + ".cluster.tsv"
							tfile = tfile.replace(" ", "")
							tfile = tfile.replace(",", "_")
							tfile = tfile.replace("-", "_")
							if os.path.isfile(tfile):
								exclude_list_chem1 = apeer_lib.GetCIExclusionList(cidata, pairx, ci_cut)
								exclude_list_chem2 = apeer_lib.GetCIExclusionList(cidata, pairy, ci_cut)
								
								# merge exclude lists:
								total_exclude_list = defaultdict(int)
								for tid in exclude_list:
									total_exclude_list[tid] = 1
								for tid in exclude_list_chem1:
									total_exclude_list[tid] = 1
								for tid in exclude_list_chem2:
									total_exclude_list[tid] = 1
								
								# remove counties from clustering
								clustdata_raw = apeer_lib.LoadClusterData(tfile)
								clustdata = defaultdict(int)
								for tfips in clustdata_raw:
									clustdata[tfips] = clustdata_raw[tfips]
									if (tfips in total_exclude_list):
										clustdata[tfips] = '-1'
								
								# remove counties from refbelt
								for tid in refbelt:
									total_exclude_list.pop(tid, None)
								
								# count up 
								county_count = 0
								for tid in total_exclude_list:
									county_count += 1
								#print("Number of counties excluded: " + str(county_count))
								
								tjaccard, tpval = apeer_lib.CalcJaccard(clustdata, refbelt, countycnt)
								outdata += ofile + "\t" + pairx + "\t" + pairy + "\t" + str(tcol) + "\t" + str(ci_cut) + "\t" + str(tquant) + "\t" + str(nclust) + "\t" + str(tjaccard) + "\t" + str(tpval) + "\t" + str(county_count) + "\n"

	f = open("ci_jaccard_sens.tsv", "w")
	f.write(outdata)
	f.close()
	
	# draw out in HTML table - you will need to manually build this in Excel
	cijaccard = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
	cicnt = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
	cmax_jaccard = defaultdict(float)
	ccnt_jaccard = defaultdict(float)
	with open("ci_jaccard_sens.tsv", "r") as infile:
		
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			chem1 = ldata[1]
			chem2 = ldata[2]
			tdisease = ldata[3]
			cicut = ldata[4]
			tpct = ldata[5]
			tjaccard = float(ldata[7])
			tpval = ldata[8]
			tn = float(ldata[9])
			
			poll_id = tdisease + "_" + chem1 + "_" + chem2 + "_" + tpct + "_" + cicut
			if poll_id in cmax_jaccard:
				if cmax_jaccard[poll_id] < tjaccard:
					cmax_jaccard[poll_id] = tjaccard
					ccnt_jaccard[poll_id] = tn
			if poll_id not in cmax_jaccard:
				cmax_jaccard[poll_id] = tjaccard
				ccnt_jaccard[poll_id] = tn
				
	infile.close()
	
	tmptable = defaultdict(int)
	for tid in sorted(cmax_jaccard):
		ldata = tid.split('_')
		tdisease = ldata[0]
		tpoll1 = ldata[2]
		tpoll2 = ldata[3]
		cival = str(ldata[len(ldata) - 1])
		tpct = str(ldata[len(ldata) - 2])
		nid = tdisease + '_' + tpoll1 + '_' + tpoll2
		
		tinclude = False
		if (tdisease == "DIABETES") and (tpoll1 == "ACETALDEHYDE") and (tpoll2 == "BENZOAPYRENE"):
			tinclude = True
		if (tdisease == "STROKE") and (ldata[1] != "MORT") and (tpoll1 == "ACETALDEHYDE") and (tpoll2 == "BENZOAPYRENE"):
			tinclude = True
		if (tdisease == "BPHIGH") and (tpoll1 == "ACETALDEHYDE") and (tpoll2 == "BENZOAPYRENE"):
			tinclude = True
		if (tdisease == "COPD") and (tpoll1 == "FORMALDEHYDE") and (tpoll2 == "GLYCOL ETHERS"):
			tinclude = True
		if (tdisease == "STROKE") and (ldata[1] == "MORT") and (tpoll1 == "ACETALDEHYDE") and (tpoll2 == "FORMALDEHYDE"):
			tinclude = True
			nid = tdisease + '_MORT_' + tpoll1 + '_' + tpoll2

		if (tinclude == True):
			print(nid + "\t" + str(cival) + "\t" + str(tpct) + "\t" + str(cmax_jaccard[tid]))
			tmptable[nid + "\t" + str(cival) + "\t" + str(tpct) + "\t" + str(cmax_jaccard[tid])] = 1
			cijaccard[nid][str(tpct)][str(cival)] = float(cmax_jaccard[tid])
			cicnt[nid][str(tpct)][str(cival)] = int(ccnt_jaccard[tid])

	#for zid in tmptable:
	#	print("ZID: " + zid)

	for xid in cijaccard:
		print(xid)
		tline = "\t"
		for tdis in disease_cutoffs:
			tline += "n" + "\t" + str(tdis) + "\t"
		tline = tline[:-1]
		print(tline)
		for tcut in ci_cutoffs:
			tline = str(tcut) + "\t"
			for tdis in disease_cutoffs:
				tline += str(int(cicnt[xid][str(tdis)][str(tcut)])) + "\t" + str(round(cijaccard[xid][str(tdis)][str(tcut)], 4)) + "\t"
			tline = tline.strip()
			print(tline)

#####################################
### Supplemental Time Series Maps ###
#####################################
if TimeseriesMaps == True:

	nclust = 5
	bdisease = "lancet.canonical.Stroke.0.7.binarymap.tsv"
	timeseries_yearlist = ['2017', '2018', '2019']
	#timeseries_yearlist = ['2014', '2017', '2018', '2019']
	for tyear in timeseries_yearlist:
		print("Processing " + tyear)
		tfile = "timeseries_" + tyear
		timedata = apeer_lib.LoadTimeSeriesData(tyear)
		#apeer_lib.ShowTractClustersOnMap(timedata, nclust, tfile, "county", json_data_county, showmap=True, forceBinary=False)
		apeer_lib.ShowMatchingCluster(timedata, nclust, tfile, json_data_county, bdisease)
		
	# now process chronic disease maps
	apeer_lib.ChronicDiseaseTimeseries()
	for tyear in range(2020, 2024):

		bfile = "d:\\apeer\\timeseries" + "\\" + "STROKE_CrudePrev_" + str(tyear) + ".binarymap.png"
		ofile = "d:\\apeer\\timeseries" + "\\" + "STROKE_CrudePrev_" + str(tyear) + ".binarymap.tsv"
		ifile = "d:\\apeer\\timeseries" + "\\" + "STROKE_CrudePrev_" + str(tyear) + ".map.tsv"
		
		# load binary data
		maplist = defaultdict(float)
		with open(ifile, "r") as infile:
			for line in infile:
				line = line.strip()
				ldata = line.split("\t")
				tfips = ldata[0]
				tval = float(ldata[1])
				if tval == 1:
					maplist[tfips] = tval
		
		apeer_lib.ShowBinaryMap(maplist, bfile, json_data_county, light_gray, "blue")

	# put figures into a webpage
	tHTML = '<html><body style="font-family: arial; font-size: 32px;">'
	
	tHTML += '<center><br>Stroke Prevalence\n'
	tHTML += '<table><tr>\n'
	for tyear in range(2020, 2023):
		bfile = "d:\\apeer\\timeseries" + "\\" + "STROKE_CrudePrev_" + str(tyear) + ".binarymap.png"
		tHTML += '		<td style="text-align: center;"><img style="width: 500px;" src="' + bfile + '"><br>' + str(tyear) + '</td>\n' 
	tHTML += '	</tr></table>'	

	tHTML += '<br>Acetaldehyde and Benzo(a)pyrene\n'
	tHTML += '<table><tr>\n'
	for tyear in range(2017, 2020):
		bfile = "d:\\apeer\\timeseries_" + str(tyear) + ".map.png"
		tHTML += '		<td style="text-align: center;"><img style="width: 500px;" src="' + bfile + '"><br>' + str(tyear) + '</td>\n' 
	tHTML += '	</tr></table>\n'	

	tHTML += "</center></html>"

	f = open("timeseries.html", "w")
	f.write(tHTML)
	f.close()

####################################
### Other Supplemental Functions ###
####################################

if MakeFigure4 == True:

	# figure 4 with sample networks
	apeer_lib.Figure4Networks()

if MakeFigure5 == True:
	
	# Figure 5 - top XGBoost and ElasticNet AUCs
	# ReferenceMapFigure - Figure 6 (lancet_figure2_ref.html, lancet_figure6_compare.html)
	apeer_lib.ReferenceMapFigure()
	# d:\bphc\sdi\networks\lancet_fig5_regressions.html
	apeer_lib.Figure5Regression()
	# d:\bphc\sdi\networks\supp_compare_regressions.html
	apeer_lib.SupRegressionResults()
	# d:\bphc\sdi\networks\supp_logistic_regressions.html
	apeer_lib.SupLogRegressionResults()
	apeer_lib.MakeNetworkFig()
	# CompareRegressionResults - supp_compare_regressions.html
	apeer_lib.CompareRegressionResults()

	
if CalibrationTable == True:

	# Produces files (supplemental_calibration_curves.html, supplemental_logistic_calibration_curves.html)
	apeer_lib.MakeCalibrationFigure("xgboost")
	apeer_lib.MakeCalibrationFigure("logistic")

if BaselineRegressionTable == True:

	# normalize data
	chemical_data = chemical_data.replace(np.nan, 0)
	norm_chemical_data = normalize(chemical_data, axis=0)

	#ndisease_list = disease_list.copy() + ["Life Expectancy", "Stroke Mortality"]
	ndisease_list = disease_list.copy() + ["STROKE_MORT"]
	
	for tdisease in ndisease_list:

		tdata = pd.DataFrame(norm_chemical_data, columns=chemical_data.columns, index=chemical_data.index)
	
		# we don't need to normalize y and x, just x 
		# combine with disease data
		for tfips in tdata.index:
			tdata.loc[tfips, tdisease] = merge_data_counties.loc[tfips, tdisease]
				
		ofile = tdisease + ".lancet_merge.regression.tsv"
		print("Processing Regression for " + tdisease)
		
		# merge health data with pollution data
		apeer_lib.RunUnivariateRegression(tdata, tdisease, ofile)
	
	apeer_lib.MakeBaselineRegTable()

if RunMoranLisa == True:

	tstr = "You will need to run this manually using Anaconda (typing anaconda in cmd)\n"
	tstr += "MAKE SURE YOU RUN IN ADMINISTRATOR MODE\n"
	tstr += "python spatial_reg.py\n"
	
	print(tstr)

if RunDensityPlots == True:

	# merge into single dataframe
	tcolumns = ["FORMALDEHYDE", "ACETALDEHYDE", "METHANOL", "Ozone", "Particulate Matter 2.5"]
	tdiseases = ["Stroke", "STROKE_MORT", "Hypertension", "Diabetes", "Renal Disease"]

	merge_data = chemical_data[tcolumns]
	for tfips in merge_data.index:
		for tdis in tdiseases:
			if tfips in merge_data_counties.index:
				merge_data.loc[tfips, tdis] = merge_data_counties.loc[tfips, tdis]

	# get data
	norm_merge_data = apeer_lib.NormalizeData(merge_data)
	norm_merge_data = norm_merge_data.rename(columns={'FORMALDEHYDE': 'Formaldehyde', 'ACETALDEHYDE': 'Acetaldehyde', "METHANOL": "Methanol", "STROKE_MORT": "Stroke Mortality"})
	
	# get pollution density
	#apeer_lib.GetDensityPlot(norm_chemical_data, tcolumns, tcolumns, "lancet_merge_density_pollution.png", 0.1)
	
	# get disease density plots
	apeer_lib.GetDensityPlot(norm_merge_data, norm_merge_data.columns, norm_merge_data.columns, "lancet_merge_density_disease.png", 0.05)
	
if PopScatter == True:

	popdata = apeer_lib.GetPopulationData()
	norm_chemical_data = apeer_lib.NormalizeData(chemical_data)

	tcolumns = ["FORMALDEHYDE", "ACETALDEHYDE", "METHANOL", "Ozone", "Particulate Matter 2.5"]
	tdiseases = ["Stroke", "STROKE_MORT", "Hypertension", "Diabetes", "Renal Disease"]
	
	filelist = []
	for tcol in tcolumns:
		tfile = tcol + "_population_scatter.png"
		apeer_lib.PlotPopCorrelation(popdata, norm_chemical_data, tcol, tfile)
		filelist.append(tfile)

	for tcol in tdiseases:
		tfile = tcol + "_population_scatter.png"
		apeer_lib.PlotPopCorrelation(popdata, merge_data_counties, tcol, tfile)
		filelist.append(tfile)
	
	tHTML = "<html><table>\n"
	x = 0
	while x < len(filelist):
		tHTML += "<tr><td><img src=\"" + filelist[x] + "\" style=\"width: 300px;\"></td>\n"
		tHTML += "<td><img src=\"" + filelist[x + 1] + "\" style=\"width: 300px;\"></td>\n"
		tHTML += "<td><img src=\"" + filelist[x + 2] + "\" style=\"width: 300px;\"></td>\n"
		tHTML += "<td><img src=\"" + filelist[x + 3] + "\" style=\"width: 300px;\"></td>\n"
		tHTML += "<td><img src=\"" + filelist[x + 4] + "\" style=\"width: 300px;\"></td></tr>\n"
		x += 5
	tHTML += "</table></html>\n"

	f = open("lancet_merge_pop_corr.html", "w")
	f.write(tHTML)
	f.close()

