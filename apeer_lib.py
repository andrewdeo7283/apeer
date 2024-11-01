import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter("ignore", category=UserWarning)

import matplotlib
import math
matplotlib.use("Agg")

from matplotlib.colors import LinearSegmentedColormap

from ckmeans_1d_dp import ckmeans
import glob
from sklearn import mixture
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegressionCV 
from sklearn.linear_model import Ridge
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
import os
import statistics

import statsmodels.api as sm
from decimal import Decimal

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize
from sklearn.cluster import KMeans
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score, make_scorer, r2_score, mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler
from numpy import arange

from matplotlib.pyplot import gcf
import networkx as nx

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

from numpy import loadtxt
from xgboost import XGBClassifier, plot_tree, cv
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import imgkit

#import statsmodels.api as sm

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

import delong

rpath = "d:\\apeer"
mappath = rpath + "\\maps"

disease_columns = ["Arthritis", "Hypertension", "Cancer", "Asthma", "Coronary Heart Disease", "COPD", "Depression", "Diabetes", "Renal Disease", "Obesity", "Stroke"]
pollution_columns = ["Lead Paint", "Diesel Particulate Matter", "Air Toxic Cancer Risk", "Air Toxic Resp Risk", "Traffic Proximity", "Wastewater discharge", "Superfund Proximity", "RMP Facility Proximity", "Hazardous Waste Proximity", "Ozone", "Particulate Matter 2.5", "Underground Storage Tanks"]
#demographics_model = ['Percent Minority', 'Percent Low Income', 'Percent Less than HS Education', 'Percent Linguistic Isolation', 'Percent Unemployed']

#demographics_model = ['Percent Minority', 'Percent Low Income', 'Percent Less than HS Education']
#pollution_model = ["Air Toxic Cancer Risk", "Air Toxic Resp Risk", "Diesel Particulate Matter", "Particulate Matter 2.5", "Ozone"]
#all_pollution_model = ["Diesel Particulate Matter", "Air Toxic Cancer Risk", "Air Toxic Resp Risk", "Traffic Proximity", "Wastewater discharge", "Superfund Proximity", "RMP Facility Proximity", "Hazardous Waste Proximity", "Ozone", "Particulate Matter 2.5", "Underground Storage Tanks", "Lead Paint"]
#disease_model = ["Mammogram", "Core Female Prevention", "Core Male Prevention", "Dental Services", "Annual Checkup"]

# Updated on Oct 20, 2024 - adjustments with demographics
demographics_model = ['Percent Minority', 'Percent Linguistic Isolation', 'Percent Unemployed']
pollution_model = ["Air Toxic Cancer Risk", "Air Toxic Resp Risk", "Diesel Particulate Matter", "Particulate Matter 2.5", "Ozone"]
all_pollution_model = ["Diesel Particulate Matter", "Air Toxic Cancer Risk", "Air Toxic Resp Risk", "Traffic Proximity", "Wastewater discharge", "Superfund Proximity", "RMP Facility Proximity", "Hazardous Waste Proximity", "Ozone", "Particulate Matter 2.5", "Underground Storage Tanks", "Lead Paint"]
disease_model = ["Mammogram", "Core Female Prevention", "Core Male Prevention", "Dental Services", "Annual Checkup"]

strokebelt_states = ["01", "05", "12", "13", "18", "21", "22", "28", "37", "45", "47", "48", "51"]

light_gray = "#f5f5f5"
blue_color = "blue"

def formatPVal(tval):

	nval = str(tval)
	ldata = nval.split('e')
	pval = nval[0:4] + 'x10<sup>' + ldata[1] + "</sup>"
	
	return pval

def median(data):

	data.sort()
	mid = len(data) // 2
	return (data[mid] + data[~mid]) / 2.0

def FormatFilename(tfile):

	tfile = tfile.replace(" ", "")
	tfile = tfile.replace(",", "_")
	tfile = tfile.replace("-", "_")

	return tfile

def FormatPercentile(tstr):

	retval = ""
	if (tstr == "0.2"):
		retval = "20th %ile"
	if (tstr == "0.3"):
		retval = "30th %ile"
	if (tstr == "0.4"):
		retval = "40th %ile"
	if (tstr == "0.5"):
		retval = "50th %ile"

	if (tstr == "0.6"):
		retval = "60th %ile"
	if (tstr == "0.7"):
		retval = "70th %ile"
	if (tstr == "0.8"):
		retval = "80th %ile"
	if (tstr == "0.9"):
		retval = "90th %ile"

	return retval

def Cluster1d(points):

	clusters = []
	eps = 0.2
	points_sorted = sorted(points)
	curr_point = points_sorted[0]
	curr_cluster = [curr_point]
	for point in points_sorted[1:]:
		if point <= curr_point + eps:
			curr_cluster.append(point)
		else:
			clusters.append(curr_cluster)
			curr_cluster = [point]
		curr_point = point
	clusters.append(curr_cluster)

	return clusters

def GetJaccardColorScale(tval):

	retval = "#0000FF"
	
	'''
	if (tval > 0) and (tval < 0.1):
		retval = "#06065D"
	if (tval >= 0.1) and (tval < 0.15):
		retval = "#0A2889"
	if (tval >= 0.15) and (tval < 0.2):
		retval = "#0E49B5"
	if (tval >= 0.2) and (tval < 0.25):
		retval = "#5892CB"
	if (tval >= 0.25) and (tval < 0.3):
		retval = "#A2DAE0"
	if (tval >= 0.3) and (tval < 0.35):
		retval = "#FF4949"
	if (tval >= 0.35) and (tval < 0.4):
		retval = "#F62525"
	if (tval >= 0.4) and (tval < 0.45):
		retval = "#ED0101"
	if (tval >= 0.45) and (tval < 0.5):
		retval = "#CA0104"
	if (tval >= 0.5):
		retval = "#A70107"
	'''

	tcolors = ["FF0000", "D3212D", "A2264B", "722B6A", "412F88", "1034A6"]
	if tval < 0.25:
		retval = tcolors[5]
	if (tval >= 0.25) and (tval < 0.3):
		retval = tcolors[4]
	if (tval >= 0.3) and (tval < 0.35):
		retval = tcolors[3]
	if (tval >= 0.35) and (tval < 0.4):
		retval = tcolors[2]
	if (tval >= 0.4) and (tval < 0.45):
		retval = tcolors[1]
	if (tval >= 0.45):
		retval = tcolors[0]

	return retval

def NormalizeData(chemical_data):

	# get data
	chemical_data = chemical_data.replace(np.nan, 0)
	norm_chemical_data = normalize(chemical_data, axis=0)
	norm_chemical_data = pd.DataFrame(norm_chemical_data, columns=chemical_data.columns, index=chemical_data.index)

	return norm_chemical_data

def LoadMedianAge(merge_data_counties):

	tfile = "d:\\apeer_data\\ACSDT5Y2021.B01002-Data.csv"
	
	trow = 0
	age_table = defaultdict(float)
	with open(tfile, "r") as infile:
		csv_reader = csv.reader(infile, delimiter=",")
		for row in csv_reader:
			if trow > 1:
				tfips = row[0].split('US')
				tfips = tfips[1]
				tage = row[2]
				age_table[tfips] = tage
				#print(tfips + "\t" + str(tage))
			trow += 1
	infile.close()
	
	for tfips in merge_data_counties.index:
		merge_data_counties.loc[tfips, "MEDIAN_AGE"] = age_table[tfips]
	
	return merge_data_counties

def Load2019LifeExpect(merge_data_counties):

	tfile = "d:\\apeer_data\\lancet_life_expect_2019.csv"
	
	trow = 0
	age_table = defaultdict(float)
	with open(tfile, "r") as infile:
		csv_reader = csv.reader(infile, delimiter=",")
		for row in csv_reader:
			if trow > 0:
				tfips = row[4]
				if len(tfips) == 4:
					tfips = '0' + tfips
				tcat = row[6]
				tbin = row[10]
				if (tcat == "Total") and (tbin == "<1 year"):
					print(tfips + "\t" + tcat + "\t" + tbin + "\t" + row[14])
					tval = row[14]
					if tval == "":
						tval = "0"
					tval = float(tval)
					age_table[tfips] = tval
			trow += 1
	infile.close()
	
	for tfips in merge_data_counties.index:
		merge_data_counties.loc[tfips, "LIFE_EXPECT_2019"] = age_table[tfips]
	
	return merge_data_counties
	
def LoadTimeSeriesData(tyear):
	
	bpath = "d:\\apeer\\timeseries"
	# 2017_Acetaldehyde, 2017_BENZO-A-PYRENE

	odata = defaultdict(lambda: defaultdict(float))	
	substance_list = ['Acetaldehyde', 'BENZO-A-PYRENE']
	#substance_list = ['Acetaldehyde', 'Formaldehyde']
	for tsubstance in substance_list:
		popcount = defaultdict(int)
		countypop = defaultdict(int)
		blockdata = defaultdict(float)
		with open(bpath + "\\" + str(tyear) + "_" + tsubstance + "\\Ambient Concentration (ug_m3).txt") as infile:
			csv_reader = csv.reader(infile, delimiter=',')
			for row in csv_reader:
				tcounty = row[2]
				if (tcounty != "Entire US") and (tcounty != "Entire State"):
					tcounty = row[3]
					tfips = row[4]
					tpop = row[5]
					tfield = row[6]
					nval = row[7]
					countypop[tcounty] += int(tpop)
					popcount[tfips] = int(tpop)
					if nval == '':
						nval = '0'
					tval = float(nval)
					blockdata[tfips] = tval
					#odata[tfips] = tval
		infile.close()
		
		# now calculate population-weighted concentrations
		for tfips in blockdata:
			tcounty = tfips[0:5]
			tweight = popcount[tfips] / countypop[tcounty]
			tval = blockdata[tfips] * tweight
			odata[tcounty][tsubstance] += tval

	# convert to dataframe
	rowdata = []
	maintable = []
	for tfips in odata:
		rowdata = [tfips]
		for tfield in sorted(odata[tfips]):
			rowdata.append(odata[tfips][tfield])
		maintable.append(rowdata)

	ncols = ['fips', 'Acetaldehyde', 'BENZO-A-PYRENE']
	acols = ['Acetaldehyde', 'BENZO-A-PYRENE']
	otable = pd.DataFrame(maintable, columns=ncols)
	otable.set_index("fips", inplace=True)
	otable[acols] = otable[acols].apply(pd.to_numeric)
	otable = otable.replace(np.nan, 0)

	# normalize data
	ctable = normalize(otable, axis=0)
	otable = pd.DataFrame(ctable, columns=otable.columns, index=otable.index)
	
	return otable

def ChronicDiseaseTimeseries():
	
	def get_disease_map(tlabel, outframe, tcol1):

		intersect_table = pd.DataFrame()
		datalist = defaultdict(lambda: defaultdict(int))

		tmatch = 0
		tmismatch = 0
		ttotal = 0
		tcutoff1 = outframe[tcol1].quantile(0.7)
		for tfips in outframe.index:
			tval = float(outframe[tcol1][tfips])
			bval = 0
			if tval >= tcutoff1:
				bval = 1
			datalist[tfips][tcol1] = bval
		
		f = open("d:\\apeer\\timeseries" + "\\" + tlabel + "_" + tcol1 + ".map.tsv", "w")
		for tfips in datalist:
			f.write(tfips + "\t" + str(datalist[tfips][tcol1]) + "\n")
		f.close()
	
	# plot out reference maps for stroke for a few years
	datatable = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
	cols ={'BPHIGH_CrudePrev': 0, 'STROKE_CrudePrev': 0, 'COPD_CrudePrev': 0, 'DIABETES_CrudePrev': 0}
	for x in range(2020, 2025):

		tfile = "d:\\apeer\\timeseries" + "\\" + 'PLACES__County_Data__GIS_Friendly_Format___' + str(x) + '_release_20241013.csv'
		rowcnt = 0
		with open(tfile, "r") as infile:
			csv_reader = csv.reader(infile, delimiter=',')
			for row in csv_reader:

				if rowcnt == 0:
					for y in range(0, len(row)):
						trow = row[y].lower().strip()
						for ncol in cols:
							tcol = ncol.lower().strip()
							if tcol == trow:
								cols[ncol] = y

				if rowcnt > 0:
					for ncol in cols:
						tfips = row[3]
						tindex = cols[ncol]
						datatable[str(x)][ncol][tfips] = row[tindex]					
						#print(tfile + "\t" + ncol + "\t" + tfips + "\t" + row[tindex])

				rowcnt += 1
				
		infile.close()

	# build a map timeline showing little change from year to year...
	for ncol in cols:

		labels = []
		# show pairwise matches between [2020, 2021], [2021, 2022], ...
		
		clusterframe = pd.DataFrame()
		for x in range(2020, 2025):
			labels.append(str(x))
			for tfips in datatable[str(x)][ncol]:
				tval = datatable[str(x)][ncol][tfips]
				if tval == '':
					tval = '0'
				clusterframe.loc[tfips, str(x)] = float(tval)

		tyear = 2020
		while tyear < 2024:
			#odata = calculate_county_overlap(clusterframe, str(tyear), str(tyear + 1))
			#print(ncol + "\t" + str(tyear) + "\t" + str(tyear + 1) + "\t" + str(odata))
			get_disease_map(ncol, clusterframe, str(tyear))
			tyear += 1

def LoadPLACES(tmode):

	# load counties
	includeCols = ["ACCESS2_CrudePrev","ARTHRITIS_CrudePrev","BINGE_CrudePrev","BPHIGH_CrudePrev","BPMED_CrudePrev","CANCER_CrudePrev","CASTHMA_CrudePrev","CERVICAL_CrudePrev","CHD_CrudePrev","CHECKUP_CrudePrev","CHOLSCREEN_CrudePrev","COLON_SCREEN_CrudePrev","COPD_CrudePrev","COREM_CrudePrev","COREW_CrudePrev","CSMOKING_CrudePrev","DENTAL_CrudePrev","DEPRESSION_CrudePrev","DIABETES_CrudePrev","GHLTH_CrudePrev","HIGHCHOL_CrudePrev","KIDNEY_CrudePrev","LPA_CrudePrev","MAMMOUSE_CrudePrev","MHLTH_CrudePrev","OBESITY_CrudePrev","PHLTH_CrudePrev","SLEEP_CrudePrev","STROKE_CrudePrev","TEETHLOST_CrudePrev"]

	cdcdata = defaultdict(lambda: defaultdict(str))
	#bpath = "c:\\Apache24\\htdocs\\bphc\\data"
	#bpath = "d:\\bphc\\data"
	bpath = "d:\\apeer_data"
	popdata = defaultdict(int)
	rowcnt = 0
	
	#tfile = bpath + "\\" + "PLACES__County_Data__GIS_Friendly_Format___2021_release.tsv"
	#tfile = bpath + "\\" + "PLACES__County_Data__GIS_Friendly_Format___2023_release_20240109.csv"	
	tfile = bpath + "\\" + "PLACES__County_Data__GIS_Friendly_Format___2022_release_20240504.csv"

	#tsep = "\t"
	tsep = ','
	tcol = "CountyFIPS"
	if tmode == "tract":
		#tfile = bpath + "\\" + "PLACES__Census_Tract_Data__GIS_Friendly_Format___2021_release.csv"
		tfile = bpath + "\\" + "PLACES__Census_Tract_Data__GIS_Friendly_Format___2023_release_20240109.csv"
		tsep = ","
		tcol = "TractFIPS"
	
	#print("Loading " + tfile)
	tplaces = pd.read_table(tfile, delimiter=tsep, dtype=str)
	tplaces = tplaces.set_index(tcol)
	#tplaces.index = tplaces.index.map(str)
	tindex = tplaces.index
	tplaces[includeCols] = tplaces[includeCols].apply(pd.to_numeric)
	# check to see if fips is 5 digits
	fipscodes = tplaces.index
	for x in range(0, len(fipscodes)):
	
		tindex = tplaces.index[x]
		
		if tmode == "county":
			if len(str(tindex)) == 4:
				nindex = "0" + tindex
				#print("Renaming " + tindex + " to " + nindex)
				tplaces = tplaces.rename(index={tindex: nindex})

		if tmode == "tract":
			if len(str(tindex)) == 10:
				#print("Padding zero: " + tindex)
				nindex = "0" + tindex
				#print("Renaming " + tindex + " to " + nindex)
				tplaces = tplaces.rename(index={tindex: nindex})
				
		#if ((x % 100) == 0):
		#	print(str(x) + ". Index: " + tindex)

	tfinal = tplaces[includeCols]
	#print("LoadPLACES:")
	#print(tfinal)

	return tfinal

def LoadEPAData(tmode):

	# get file
	tfile = "d:\\apeer_data\\EJSCREEN_2021_USPR.csv\\EJSCREEN_2021_USPR.csv"
	
	# load into dataframe
	# https://www.epa.gov/ejscreen/overview-environmental-indicators-ejscreen
	
	#6	REGION	US EPA Region number
	#7	ACSTOTPOP	Total population
	#8	MINORPCT	% people of color
	#9	LOWINCPCT	% low income
	#10	LESSHSPCT	% less than high school education
	#11	LINGISOPCT	% linguistically isolated
	#12	UNDER5PCT	% under age 5
	#13	OVER64PCT	% over age 64
	#14	PRE1960PCT	Lead paint
	#15	UNEMPPCT	Unemployment rate
	#16	VULEOPCT	Demographic Index
	#17	DSLPM	2017 Diesel particulate matter
	#18	CANCER	2017 Air toxics cancer risk
	#19	RESP	2017 Air toxics respiratory HI
	#20	PTRAF	Traffic proximity
	#21	PWDIS	Wastewater discharge
	#22	PNPL	Superfund proximity
	#23	PRMP	RMP facility proximity
	#24	PTSDF	Hazardous waste proximity
	#25	OZONE	Ozone
	#26	PM25	Particulate Matter 2.5
	#27	UST	Underground storage tanks
	
	#collist = ['ACSTOTPOP', "CANCER", "DSLPM", "RESP", "PTRAF", "PWDIS", "PNPL", "PRMP", "PTSDF", "OZONE", "PM25", "UST", "PRE1960PCT", "MINORPCT", "LOWINCPCT", "LESSHSPCT", "LINGISOPCT", "UNDER5PCT", "OVER64PCT", "UNEMPPCT"]
	collist = ['ACSTOTPOP', "MINORPCT", "LOWINCPCT", "LESSHSPCT", "LINGISOPCT", "UNDER5PCT", "OVER64PCT", "UNEMPPCT", \
	"PRE1960PCT", "DSLPM", "CANCER", "RESP", "PTRAF", "PWDIS", "PNPL", "PRMP", "PTSDF", "OZONE", "PM25", "UST"]
	
	collist_final = []
	for tid in collist:
		if tid != "CANCER":
			collist_final.append(tid)
		if tid == "CANCER":
			collist_final.append("CANCER_AIR")
	
	#including PTRAF or PWDIS removes the clusters
	raw_data = pd.read_csv(tfile, header=0, low_memory=False)
	raw_data.set_index("ID", inplace=True)
	raw_data = raw_data[collist]
	
	# average county data
	ttable = []
	tempCountyData = defaultdict(lambda: defaultdict(float))
	tempTractData = defaultdict(lambda: defaultdict(float))
	county_data = pd.DataFrame()	
	countytotal = defaultdict(float)
	tracttotal = defaultdict(float)
	tractdata = defaultdict(float)

	# get county pop totals for weights
	for x in range(0, len(raw_data.index)):
		tfips = str(raw_data.index[x])
		nfips = str(int(tfips))
		nfips = nfips.replace('.0', '')
		if len(nfips) == 11:
			nfips = "0" + nfips
		ttract = str(nfips)[0:11]
		tcounty = str(nfips)[0:5]
		countytotal[tcounty] += raw_data.loc[raw_data.index[x], "ACSTOTPOP"]
		tracttotal[ttract] += raw_data.loc[raw_data.index[x], "ACSTOTPOP"]
	
	for x in range(0, len(raw_data.index)):
		tfips = raw_data.index[x]
		nfips = str(tfips)
		nfips = nfips.replace('.0', '')
		if len(nfips) == 11:
			nfips = "0" + nfips
		tstate = str(nfips)[0:2]
		ttract = str(nfips)[0:11]
		tcounty = str(nfips)[0:5]
		# ignore protectorates
		if int(tstate) < 60:
			weight_county = raw_data.loc[raw_data.index[x], "ACSTOTPOP"] / countytotal[tcounty]
			weight_tract = raw_data.loc[raw_data.index[x], "ACSTOTPOP"] / tracttotal[ttract]
			for tcol in collist:
				ncol = tcol
				if ncol == "CANCER":
					ncol = "CANCER_AIR"
				if tmode == "county":
					tempCountyData[tcounty][ncol] += raw_data.loc[tfips, tcol] * weight_county
				if tmode == "tract":
					tempTractData[ttract][ncol] += raw_data.loc[tfips, tcol] * weight_tract

	if tmode == "county":
		for tfips in tempCountyData:
			trow = [tfips]
			for tcol in collist:
				ncol = tcol
				if ncol == "CANCER":
					ncol = "CANCER_AIR"
				trow.append(tempCountyData[tfips][ncol])
			ttable.append(trow)

	if tmode == "tract":
		for tfips in tempTractData:
			trow = [tfips]
			for tcol in collist:
				ncol = tcol
				if ncol == "CANCER":
					ncol = "CANCER_AIR"
				trow.append(tempTractData[tfips][ncol])
			ttable.append(trow)

	clist = ["fips"] + collist_final
	epa_data = pd.DataFrame(ttable, columns=clist)
	epa_data.set_index("fips", inplace=True)
	epa_data[collist_final] = epa_data[collist_final].apply(pd.to_numeric)
	epa_data = epa_data.replace(np.nan, 0)
	epa_data = epa_data.drop("ACSTOTPOP", axis=1)
	
	return epa_data

def LoadMapJSON(tmode):

	# Map the data
	print("LoadMapJSON: " + tmode)
	tfile = "us_counties.geojson"
	if tmode == "tract":
		tfile = "d:\\apeer_data\\tracts.geojson"
	with open(tfile, "r") as json_file:
		json_data = json.load(json_file)

	return json_data

def ShowClustersOnMap(datatable, numclust, covidcases, coviddeaths, tfile):

	# make filenames
	pcafile = tfile + ".pca.pdf"
	kfile = tfile + ".kmeans.pca.pdf"
	cluster_file = tfile + ".kmeans.pca.clusters.pdf"
	mapfile = tfile + ".map.pdf"
	
	# PCA
	datatable = datatable.replace(np.nan, 0)
	Xt_epa = RunPCA(datatable, pcafile, tfile)
	epa_clust = PCA_Kmeans(Xt_epa, numclust, datatable, covidcases, coviddeaths, kfile, cluster_file)

	# set colors
	tcolors = ["#ff0000", "#00ff00", "#0000ff", "#6a0dad", "#ffff00"]
	
	json_data = LoadMapJSON()

	plt.close()
	plt.axis("off")
	fig = plt.figure(figsize=(5.5, 4))
	ax = fig.gca()
	for titem in json_data["features"]:
		tfips = titem["properties"]["GEOID"]
		if tfips in epa_clust.index:
			color_index = epa_clust.loc[tfips, "cluster"]
			get_color = tcolors[color_index]
			poly = titem["geometry"]
			ax.add_patch(PolygonPatch(poly, fc=get_color, ec=get_color, alpha=0.5, zorder=2))
	
	#ax.axis('scaled')
	ax.set_xlim(-180, -60)
	ax.set_ylim(20, 80)
	plt.xlabel("Longitude")
	plt.ylabel("Latitude")

	fig.tight_layout()
	plt.savefig(mapfile)

def GetPopulationData():

	tractpop = defaultdict(float)
	countypop = defaultdict(float)
	
	tfile = "d:\\apeer_data\\EJSCREEN_2021_USPR.csv\\EJSCREEN_2021_USPR.csv"
	with open(tfile, "r") as infile:
		csv_reader = csv.reader(infile, delimiter=",")
		tcnt = 0
		for trow in csv_reader:
			if tcnt > 0:
				tid = trow[1]
				tpop = trow[2]
				tcounty = tid[0:5]
				ttract = tid[0:11]
				#print(tid + "\t" + tcounty + "\t" + tpop)
				tractpop[ttract] += float(tpop)
				countypop[tcounty] += float(tpop)
				#print(tid + "\t" + tpop)
			tcnt += 1
	
	infile.close()

	return countypop

def PlotPopCorrelation(popdata, compare_data, tcol, ofile):
	
	# get x and y data
	np.seterr(divide = 'ignore') 
	plt.close()
	x = []
	y = []
	for tfips in compare_data.index:
		if tfips in popdata:
			
			tpop = float(popdata[tfips])
			tval = float(compare_data.loc[tfips, tcol])
			
			tpop = np.log(tpop + 1)
			tval = np.log(tval + 1)
			
			x.append(tpop)
			y.append(tval)

	xvals = np.array(x)
	yvals = np.array(y)
	idx = np.isfinite(xvals) & np.isfinite(yvals)
	#plt.rcParams["figure.figsize"] = (6, 5)
	plt.figure(figsize=(5, 5))
	plt.scatter(xvals[idx], yvals[idx])
	m, b = np.polyfit(xvals[idx], yvals[idx], deg=1)
	plt.axline(xy1=(0, b), slope=m, color='r', label=f'$y = {m:.2f}x {b:+.2f}$')
	plt.xlabel("Ln(Population)")
	
	nlabel = tcol.title()
	if nlabel.lower() == "stroke_mort":
		nlabel = "Stroke Mortality"
	
	plt.ylabel("Ln(" + nlabel + ")")
	plt.legend()
	plt.savefig(ofile, dpi=600)
		

def CalculateAirToxData():

	tractpop = defaultdict(float)
	countypop = defaultdict(float)
	
	'''
	tfile = "d:\\apeer_data\\ACSST5Y2021.S0101_2023-08-20T232554\\ACSST5Y2021.S0101-Data.csv"
	with open(tfile, "r") as infile:
		csv_reader = csv.reader(infile, delimiter=",")
		tcnt = 0
		for trow in csv_reader:
			if tcnt > 1:
				tid = trow[0]
				iddata = tid.split('US')
				tid = iddata[1]
				tpop = trow[2]
				tcounty = tid[0:5]
				#print(tid + "\t" + tcounty + "\t" + tpop)
				tractpop[tid] = float(tpop)
				countypop[tcounty] += float(tpop)
				#print(tid + "\t" + tpop)
			tcnt += 1
	
	infile.close()
	'''
	
	tfile = "d:\\apeer_data\\EJSCREEN_2021_USPR.csv\\EJSCREEN_2021_USPR.csv"
	with open(tfile, "r") as infile:
		csv_reader = csv.reader(infile, delimiter=",")
		tcnt = 0
		for trow in csv_reader:
			if tcnt > 0:
				tid = trow[1]
				tpop = trow[2]
				tcounty = tid[0:5]
				ttract = tid[0:11]
				#print(tid + "\t" + tcounty + "\t" + tpop)
				tractpop[ttract] += float(tpop)
				countypop[tcounty] += float(tpop)
				#print(tid + "\t" + tpop)
			tcnt += 1
	
	infile.close()
	
	
	for tcounty in countypop:
		print(tcounty + "\t" + str(countypop[tcounty]))
		
	# load chemical data - 
	missing_counties = defaultdict(str)
	chemtable = defaultdict(lambda: defaultdict(float))
	countytable = defaultdict(lambda: defaultdict(float))
	valtable = defaultdict(lambda: defaultdict(list))
	chemlist = defaultdict(str)
	
	with open("d:\\apeer_data\\2018_Toxics_Ambient_Concentrations.txt", "r") as infile:
		#line = infile.readline()
		line = infile.readline().strip()
		chemcols = line.split("\t")
		#print(chemcols)
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			tlabel = ldata[2]
			tcounty = ldata[3]
			ttract = ldata[4]
			#print(tcounty + "\t" + ttract)
			if tlabel.find('Entire') == -1:
				for x in range(5, len(chemcols)):
					chemtable[ttract][chemcols[x]] = float(ldata[x])
					if tcounty not in countypop:
						#print("Missing county: " + tcounty)
						missing_counties[tcounty] = ""
					if tcounty in countypop:
						tweight = tractpop[ttract] / countypop[tcounty]
						tval = float(ldata[x]) * tweight
						countytable[tcounty][chemcols[x]] += tval
						valtable[tcounty][chemcols[x]].append(tval)
					chemlist[chemcols[x]] = ''
					#print(ttract + "\t" + chemcols[x] + "\t" + ldata[x])
				
	infile.close()
	
	# write out tracts
	print("Writing out tracts...")
	f = open("d:\\apeer_data\\2018_Toxics_Ambient_Concentrations.updated.tract.tsv", "w")
	theader = "FIPS\t"
	for tchem in sorted(chemlist):
		nchem = tchem.replace('"', '')
		theader = theader + nchem + "\t"
	theader = theader.strip()
	theader = theader + "\n"
	f.write(theader)
	
	tline = ""
	for tfips in sorted(chemtable):
		tline = tfips + "\t"
		for tchem in sorted(chemlist):
			tline = tline + str(chemtable[tfips][tchem]) + "\t"
		tline = tline.strip() + "\n"
		f.write(tline)
	f.close()
	
	# write out county data
	print("Writing out counties...")
	f = open("d:\\apeer_data\\2018_Toxics_Ambient_Concentrations.updated.county.tsv", "w")
	f.write(theader)
	tline = ""
	for tfips in sorted(countytable):
		tline = tfips + "\t"
		for tchem in sorted(chemlist):
			tline = tline + str(countytable[tfips][tchem]) + "\t"
		tline = tline.strip() + "\n"
		f.write(tline)
	f.close()

	# calculate confidence intervals
	print("Outputting confidence intervals...")
	f = open("airtoxscreen_county_confidence_intervals.tsv", "w")
	for tcounty in valtable:	
		for tchem in valtable[tcounty]:
			cintervals = stats.t.interval(0.95, len(valtable[tcounty][tchem])-1, loc=np.mean(valtable[tcounty][tchem]), scale=stats.sem(valtable[tcounty][tchem]))
			tdiff = cintervals[1] - cintervals[0]
			f.write(tcounty + "\t" + tchem + "\t" + str(cintervals[0]) + "\t" + str(cintervals[1]) + "\t" + str(tdiff) + "\n")
	f.close()
	
	return tractpop

def GetCountiesByCI():
	
	# Get CDC PLACES CIs
	# load counties
	# get confidence intervals - ACCESS2_Crude95CI
	includeCols = ["ACCESS2_CrudePrev","ARTHRITIS_CrudePrev","BINGE_CrudePrev","BPHIGH_CrudePrev","BPMED_CrudePrev","CANCER_CrudePrev","CASTHMA_CrudePrev","CERVICAL_CrudePrev","CHD_CrudePrev","CHECKUP_CrudePrev","CHOLSCREEN_CrudePrev","COLON_SCREEN_CrudePrev","COPD_CrudePrev","COREM_CrudePrev","COREW_CrudePrev","CSMOKING_CrudePrev","DENTAL_CrudePrev","DEPRESSION_CrudePrev","DIABETES_CrudePrev","GHLTH_CrudePrev","HIGHCHOL_CrudePrev","KIDNEY_CrudePrev","LPA_CrudePrev","MAMMOUSE_CrudePrev","MHLTH_CrudePrev","OBESITY_CrudePrev","PHLTH_CrudePrev","SLEEP_CrudePrev","STROKE_CrudePrev","TEETHLOST_CrudePrev"]
	
	#for x in range(0, len(includeCols)):
	#	includeCols[x] = includeCols[x].replace('CrudePrev', 'Crude95CI')

	cdcdata = defaultdict(lambda: defaultdict(str))
	#bpath = "c:\\Apache24\\htdocs\\bphc\\data"
	#bpath = "d:\\bphc\\data"
	bpath = "d:\\apeer_data"
	popdata = defaultdict(int)
	rowcnt = 0

	tfile = bpath + "\\" + "PLACES__County_Data__GIS_Friendly_Format___2022_release_20240504.csv"
	tsep = ','
	tcol = "CountyFIPS"
	
	tcnt = 0
	cicols = ["STROKE_Crude95CI", "BPHIGH_Crude95CI", "COPD_Crude95CI", "DIABETES_Crude95CI"]
	cidata = defaultdict(lambda: defaultdict(float))
	citable = pd.DataFrame()
	with open(tfile, "r") as infile:
		csv_reader = csv.reader(infile, delimiter = ',')
		for row in csv_reader:
			if tcnt == 0:
				theader = row
			if tcnt > 1:
				tfips = row[3]
				for x in range(0, len(row)):
					if (theader[x] in cicols):
						tval = row[x]
						tval = tval.replace('(', '')
						tval = tval.replace(')', '')
						vals = tval.split(',')

						tdiff = 0
						if len(vals) > 1:						
							v1 = vals[0]
							v2 = vals[1]
							if v1 == '':
								v1 = '0'
							if v2 == '':
								v2 = '0'
							tlow = float(v1)
							thigh = float(v2)
							tdiff = thigh - tlow

						citable.loc[tfips, theader[x]] = tdiff

			tcnt += 1

	infile.close()

	#for tcol in cicols:
	#	for tfips in cidata[tcol]:
	#		citable.loc[tfips, tcol] = cidata[tcol][tfips]
			
	# Get AirToxScreen CIs
	with open("airtoxscreen_county_confidence_intervals.tsv", "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			tfips = ldata[0]
			tdiff = float(ldata[4])
			tchem = ldata[1]
			citable.loc[tfips, tchem] = tdiff
	infile.close()

	# Identify counties with 5%, 10%, 15% thresholds identified
	f = open("lancet_ci_cutoffs.tsv", "w")
	tcutoffs = [0.99, 0.975, 0.95, 0.9]
	for tcol in citable.columns:
		for ncut in tcutoffs:
			tcut = citable[tcol].quantile(ncut)
			tcnt = 0
			for tfips in citable.index:
				tval = citable.loc[tfips, tcol]
				if tval > tcut:
					tcnt += 1
			f.write(tcol + "\t" + str(ncut) + "\t" + str(tcut) + "\t" + str(tcnt) + "\n")
	f.close()
	
def GetCIData():
	
	# Draw Maps - ShowMatchingCluster
	tcnt = 0
	bpath = "d:\\apeer_data"
	tfile = bpath + "\\" + "PLACES__County_Data__GIS_Friendly_Format___2022_release_20240504.csv"
	cidata = defaultdict(lambda: defaultdict(float))
	citable = pd.DataFrame()
	with open(tfile, "r") as infile:
		csv_reader = csv.reader(infile, delimiter = ',')
		for row in csv_reader:
			if tcnt == 0:
				theader = row
			if tcnt > 1:
				tfips = row[3]
				for x in range(0, len(row)):
					if (theader[x].find('Crude95CI') > -1):
						tval = row[x]
						tval = tval.replace('(', '')
						tval = tval.replace(')', '')
						vals = tval.split(',')

						tdiff = 0
						if len(vals) > 1:						
							v1 = vals[0]
							v2 = vals[1]
							if v1 == '':
								v1 = '0'
							if v2 == '':
								v2 = '0'
							tlow = float(v1)
							thigh = float(v2)
							tdiff = thigh - tlow

						citable.loc[tfips, theader[x]] = tdiff

			tcnt += 1

	infile.close()

	# Get AirToxScreen CIs
	with open("airtoxscreen_county_confidence_intervals.tsv", "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			tfips = ldata[0]
			tdiff = float(ldata[4])
			tchem = ldata[1]
			citable.loc[tfips, tchem] = tdiff
	infile.close()

	# Stroke Mortality - cdc_stroke_report.csv
	tcnt = 0
	with open("cdc_stroke_report.csv", "r") as infile:
		csv_reader = csv.reader(infile, delimiter=',')
		for row in csv_reader:
			if tcnt > 0:
				tfips = row[0]
				traw_range = row[3]
				tdiff = 100
				if (traw_range.find('(') > -1):
					traw_range = traw_range.replace('"', '')
					tdata = traw_range.split(' (')
					tdata[0] = tdata[0].strip()
					tdata2 = tdata[0].split(' - ')
					lval = float(tdata2[0].strip())
					hval = float(tdata2[1].strip())
					tdiff = hval - lval
					print("STROKE_MORT" + "\t" + tfips + "\t" + str(tdiff))
				citable.loc[tfips, 'STROKE_MORT'] = tdiff
			tcnt += 1
	infile.close()
	
	# get density plots
	#maxval = citable.max(numeric_only=True).max()
	maxval = 10
	GetDensityPlot(citable, ["STROKE_Crude95CI", "BPHIGH_Crude95CI", "COPD_Crude95CI", "DIABETES_Crude95CI"], ['Stroke', 'Hypertension', 'COPD', 'Diabetes'], "ci_density_disease.png", 4)
	GetDensityPlot(citable, ["ACETALDEHYDE", "BENZOAPYRENE", "FORMALDEHYDE", "GLYCOL ETHERS"], ['Acetaldehyde', 'Benzoapyrene', 'Formaldehyde', 'Glycol Ethers'], "ci_density_pollutant.png", 10)

	return citable

def GetCIExclusionList(citable, ccol, cpct):
	
	exclude_list = defaultdict(int)
	tcutoff = citable[ccol].quantile(cpct)
	for tfips in citable.index:
		tval = float(citable.loc[tfips, ccol])
		if tval >= tcutoff:
			exclude_list[tfips] = 1
		
	return exclude_list

def LoadChemicalData(tmode):

	if tmode == "county":
		chemdata_county = pd.read_csv("d:\\apeer_data\\2018_Toxics_Ambient_Concentrations.updated.county.tsv", header=0, index_col=0, sep="\t", low_memory=False)
		tindex = []
		for tid in chemdata_county.index:
			nid = str(tid)
			if len(nid) == 4:
				nid = "0" + nid
			tindex.append(nid)
		chemdata_county.index = tindex
		
		return chemdata_county

	if tmode == "tract":
		chemdata_tract = pd.read_csv("d:\\apeer_data\\2018_Toxics_Ambient_Concentrations.updated.tract.tsv", header=0, index_col=0, sep="\t")
		tindex = []
		for tid in chemdata_tract.index:
			nid = str(tid)
			if len(nid) == 10:
				nid = "0" + nid
			tindex.append(nid)
		chemdata_tract.index = tindex
		
		return chemdata_tract

def MergeStrokeData(tmode):

	epacols = ["MINORPCT","LOWINCPCT","LESSHSPCT","LINGISOPCT","UNDER5PCT", \
	"OVER64PCT","UNEMPPCT","PRE1960PCT","DSLPM","CANCER_AIR","RESP","PTRAF", \
	"PWDIS","PNPL","PRMP","PTSDF","OZONE","PM25","UST"]
	
	cdccols = ["ACCESS2_CrudePrev","ARTHRITIS_CrudePrev","BINGE_CrudePrev","BPHIGH_CrudePrev", \
	"BPMED_CrudePrev","CANCER_CrudePrev","CASTHMA_CrudePrev","CERVICAL_CrudePrev","CHD_CrudePrev", \
	"CHECKUP_CrudePrev","CHOLSCREEN_CrudePrev","COLON_SCREEN_CrudePrev","COPD_CrudePrev", \
	"COREM_CrudePrev","COREW_CrudePrev","CSMOKING_CrudePrev","DENTAL_CrudePrev","DEPRESSION_CrudePrev", \
	"DIABETES_CrudePrev","GHLTH_CrudePrev","HIGHCHOL_CrudePrev","KIDNEY_CrudePrev","LPA_CrudePrev", \
	"MAMMOUSE_CrudePrev","MHLTH_CrudePrev","OBESITY_CrudePrev","PHLTH_CrudePrev","SLEEP_CrudePrev", \
	"STROKE_CrudePrev","TEETHLOST_CrudePrev"]

	epadata = defaultdict(lambda: defaultdict(str))
	cdcdata = defaultdict(lambda: defaultdict(str))

	# load CDC data
	epadata_pandas = LoadEPAData(tmode)
	cdcdata_pandas = LoadPLACES(tmode)

	# convert EPA pandas to dictionary
	for tcol in epadata_pandas.columns:
		for tfips in epadata_pandas.index:
			epadata[tfips][tcol] = epadata_pandas.loc[tfips, tcol]

	# convert pandas to dictionary
	for tcol in cdcdata_pandas.columns:
		for tfips in cdcdata_pandas.index:
			cdcdata[tfips][tcol] = cdcdata_pandas.loc[tfips, tcol]
		   
	# merge data tables
	mergedata = defaultdict(lambda: defaultdict(str))
	for tfips in cdcdata:
		if tfips in epadata:
			for x in range(0, len(epacols)):
				mergedata[tfips][epacols[x]] = epadata[tfips][epacols[x]]
			for x in range(0, len(cdccols)):
				mergedata[tfips][cdccols[x]] = cdcdata[tfips][cdccols[x]]
	 
	# convert to pandas dataframe
	ttable = []
	for tid in mergedata:
		trow = [tid]
		for x in range(0, len(epacols)):
			trow.append(mergedata[tid][epacols[x]])
		for x in range(0, len(cdccols)):
			trow.append(mergedata[tid][cdccols[x]])
		ttable.append(trow)
		
	clist = ["fips"] + epacols + cdccols
	olist = epacols + cdccols
	finaldata = pd.DataFrame(ttable, columns=clist)
	finaldata.set_index("fips", inplace=True)
	finaldata[olist] = finaldata[olist].apply(pd.to_numeric)
	
	# fill missing values
	finaldata = finaldata.fillna(0)
		
	finaldata = RenameColumns(finaldata)
	#print(finaldata)

	return finaldata

def GetTractGeoJSON():

	# https://stackoverflow.com/questions/43119040/shapefile-into-geojson-conversion-python-3
	tdir = "d:\\apeer_data"

	# read the shapefile
	tstates = ["01", "02", "04", "05", "06", "08", "09", "10", "12", "13", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "44", "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56"]

	for tid in tstates:
	
		# generate geojson
		buffer = []
		#print("Generating GeoJSON - Processing " + tid)
		reader = shapefile.Reader(tdir + "\\" + "tl_2021_" + tid + "_tract.shp")
		fields = reader.fields[1:]
		field_names = [field[0] for field in fields]
		for sr in reader.shapeRecords():
			atr = dict(zip(field_names, sr.record))
			geom = sr.shape.__geo_interface__
			buffer.append(dict(type="Feature", geometry=geom, properties=atr)) 
	
		# write the GeoJSON file
		ifile = tdir + "\\" + tid + ".geojson"
		ofile = tdir + "\\" + tid + ".compress.geojson"
		geojson = open(ifile, "w")
		geojson.write(dumps({"type": "FeatureCollection", "features": buffer}, indent=2) + "\n")
		geojson.close()
		
		# compress data
		tcmd = "mapshaper-xl " + ifile + " -simplify 5% -o " + ofile
		os.system(tcmd)

	merge_data = {"type": "FeatureCollection", "features": []}
	for tid in tstates:
	
		ifile = tdir + "\\" + tid + ".compress.geojson"
		with open(ifile, "r") as f:
			data1 = json.load(f)

		for titem in data1["features"]:
			merge_data["features"].append(titem)

	f = open(tdir + "\\" + "tracts.geojson", "w")
	f.write(json.dumps(merge_data))
	f.close()

def GetMainLists():

	# print out CDC PLACES counties / tracts
	countylist = LoadPLACES("county")
	tractlist = LoadPLACES("tract")
	epacounties = LoadEPAData("county")
	epatracts = LoadEPAData("tract")
	chemdata_county = LoadChemicalData("county")
	chemdata_tract = LoadChemicalData("tract")
	master_counties = defaultdict(str)
	master_tracts = defaultdict(str)

	tcnt1 = 1
	hcountylist = defaultdict(str)
	for tcounty in countylist.index:
		#print(str(tcnt) + "\t" + tcounty)
		hcountylist[tcounty] = ""
	for tcounty in hcountylist:
		master_counties[tcounty] = ""
		tcnt1 += 1
	#print("Number of CDC counties: " + str(tcnt1))
	#print("CDC Counties: " + str(hcountylist)[0:500])

	tcnt2 = 1
	htractlist = defaultdict(str)
	for ttract in tractlist.index:
		#print(str(tcnt) + "\t" + ttract)
		htractlist[ttract] = ""
	for ttract in htractlist:
		master_tracts[ttract] = ""
		tcnt2 += 1
	#print("Number of CDC tracts: " + str(tcnt2))
	#print("CDC Tracts: " + str(htractlist)[0:500])

	# EPA EJScreen
	ecnt1 = 1
	hcountylist2 = defaultdict(str)
	for tcounty in epacounties.index:
		#print(str(tcnt) + "\t" + tcounty)
		hcountylist2[tcounty] = ""
	for tcounty in hcountylist2:
		master_counties[tcounty] = ""
		ecnt1 += 1
	#print("Number of EPA counties: " + str(ecnt1))
	#print("EPA Counties: " + str(hcountylist2)[0:500])

	ecnt2 = 1
	htractlist2 = defaultdict(str)
	for ttract in epatracts.index:
		#print(str(tcnt) + "\t" + tcounty)
		htractlist2[ttract] = ""
	for ttract in htractlist2:
		master_tracts[ttract] = ""
		ecnt2 += 1
	#print("Number of EPA tracts: " + str(ecnt2))
	#print("EPA Tracts: " + str(htractlist2)[0:500])
	
	# EPA EJScreen
	ccnt1 = 1
	hcountylist3 = defaultdict(str)
	for tcounty in chemdata_county.index:
		#print(str(tcnt) + "\t" + tcounty)
		hcountylist3[tcounty] = ""
	for tcounty in hcountylist3:
		master_counties[tcounty] = ""
		ccnt1 += 1
	#print("Number of Chem counties: " + str(ccnt1))
	#print("Chem Counties: " + str(hcountylist3)[0:500])

	ccnt2 = 1
	htractlist3 = defaultdict(str)
	for ttract in chemdata_tract.index:
		#print(str(tcnt) + "\t" + tcounty)
		htractlist3[ttract] = ""
	for ttract in htractlist3:
		master_tracts[ttract] = ""
		ccnt2 += 1
	#print("Number of Chem tracts: " + str(ccnt2))
	#print("Chem Tracts: " + str(htractlist3)[0:500])
	
	#cdc_cnt = 0
	#epa_cnt = 0
	#air_cnt = 0
	#for ttract in master_tracts:
	#	if ttract in htractlist:
	#		cdc_cnt += 1
	#	if ttract in htractlist2:
	#		epa_cnt += 1
	#	if ttract in htractlist3:
	#		air_cnt += 1
	
	#print("***** Tract Count *****")
	#print("CDC PLACES: " + str(cdc_cnt))
	#print("EPA EJSCREEN: " + str(epa_cnt))
	#print("AirToxScreen: " + str(air_cnt))
		
	# get intersections
	mcounty = 0
	mtract = 0
	finalcounties = []
	finaltracts = []
	for tcounty in master_counties:
		if (tcounty in hcountylist) and (tcounty in hcountylist2) and (tcounty in hcountylist3):
			finalcounties.append(tcounty)
			if len(tcounty.strip()) != 5:
				print("County Potential Error: " + tcounty)
			mcounty += 1
	for ttract in master_tracts:
		if (ttract in htractlist) and (ttract in htractlist2) and (ttract in htractlist3):
			finaltracts.append(ttract)
			if len(ttract.strip()) != 11:
				print("Tract Potential Error: " + ttract)
			mtract += 1
	print("Number of common counties: " + str(mcounty))
	print("Number of common tracts: " + str(mtract))
	
	return finalcounties, finaltracts

def GetNewLabels():

	changelabels = defaultdict(str)
	
	changelabels["access2"] = "Healthcare Access"
	changelabels["arthritis"] = "Arthritis"
	changelabels["binge"] = "Binge Drinking"
	changelabels["bphigh"] = "Hypertension"
	changelabels["cancer"] = "Cancer"
	changelabels["casthma"] = "Asthma"
	changelabels["cervical"] = "Cervical Cancer Screen"
	changelabels["bpmed"] = "BP Medication"
	changelabels["colon_screen"] = "Colon Cancer Screen"
	changelabels["stroke_mort"] = "Stroke Mortality"
	
	changelabels["chd"] = "Coronary Heart Disease"
	changelabels["checkup"] = "Annual Checkup"
	changelabels["cholscreen"] = "Cholesterol Screen"
	changelabels["copd"] = "COPD"
	changelabels["corem"] = "Core Male Prevention"
	changelabels["corew"] = "Core Female Prevention"
	changelabels["csmoking"] = "Smoking"
	changelabels["dental"] = "Dental Services"

	changelabels["depression"] = "Depression"
	changelabels["diabetes"] = "Diabetes"
	changelabels["ghlth"] = "Poor General Health"
	changelabels["highchol"] = "High Cholesterol"
	changelabels["kidney"] = "Renal Disease"
	changelabels["lpa"] = "Limited Phys. Activity"
	changelabels["mammouse"] = "Mammogram"
	changelabels["mhlth"] = "Poor Mental Health"

	changelabels["obesity"] = "Obesity"
	changelabels["phlth"] = "Poor Physical Health"
	changelabels["sleep"] = "Sleep"
	changelabels["stroke"] = "Stroke"
	changelabels["teethlost"] = "Tooth Loss"
	#changelabels["ltfpl100"] = "Below Poverty"
	#changelabels["singlparntfly"] = "Single Parent Family"
	#changelabels["black"] = "Black"

	#changelabels["dropout"] = "School Dropout"
	#changelabels["hhnocar"] = "No Car"
	#changelabels["rentoccup"] = "Home Rental"
	#changelabels["crowding"] = "Crowding"
	#changelabels["nonemp"] = "Non-Employment"
	#changelabels["unemp"] = "Unemployed"
	#changelabels["obesity"] = "Obesity"
	#changelabels["highneeds"] = "High Needs"
	#changelabels["hispanic"] = "Hispanic"
	#changelabels["frgnborn"] = "Foreign Born"
	#changelabels["lingisol"] = "Limited English"
	#changelabels["cases"] = "COVID-19 Cases"
	#changelabels["deaths"] = "COVID-19 Deaths"
	
	changelabels["dslpm"] = "Diesel Particulate Matter"
	changelabels["cancer_air"] = "Air Toxic Cancer Risk"
	changelabels["resp"] = "Air Toxic Resp Risk"
	changelabels["ptraf"] = "Traffic Proximity"
	changelabels["pwdis"] = "Wastewater discharge"
	changelabels["pnpl"] = "Superfund Proximity"
	changelabels["prmp"] = "RMP Facility Proximity"
	changelabels["ptsdf"] = "Hazardous Waste Proximity"
	changelabels["ozone"] = "Ozone"
	changelabels["pm25"] = "Particulate Matter 2.5"
	changelabels["ust"] = "Underground Storage Tanks"
	changelabels["pre1960pct"] = "Lead Paint"
	
	changelabels["minorpct"] = "Percent Minority"
	changelabels["lowincpct"] = "Percent Low Income"
	changelabels["lesshspct"] = "Percent Less than HS Education"
	changelabels["lingisopct"] = "Percent Linguistic Isolation"
	changelabels["under5pct"] = "Percent Under 5 yrs"
	changelabels["over64pct"] = "Percent Over 64 yrs"
	changelabels["unemppct"] = "Percent Unemployed"
		
	# add median age, density, etc. - agedata = pd.DataFrame(ttable, columns=["fips", "median_age", "age_male", "age_female", "over65"])
	#changelabels["median_age"] = "Median Age"
	#changelabels["age_male"] = "Median Male Age"
	#changelabels["age_female"] = "Median Female Age"
	#changelabels["over65"] = "Over 65"
	
	return changelabels

def RenameColumns(tdata):

	newlabels = GetNewLabels()
	newcols = []
	for tcol in tdata.columns:
		ncol = tcol.lower().strip()
		ncol = ncol.replace("_crudeprev", "")
		olabel = tcol
		
		# find exact match first...
		tfound = False
		for nlabel in newlabels:
			if nlabel == ncol:
				olabel = newlabels[nlabel]
				tfound = True
		# if no exact, do substring
		if tfound == False:
			for nlabel in newlabels:
				if ncol.find(nlabel) > -1:
					olabel = newlabels[nlabel]
					
		newcols.append(olabel)
		#if olabel == "none":
		#	print(ncol + "\t" + olabel)
	tdata.columns = newcols

	return tdata

def Calc1DCluster(tdata, json_data):

	print("50th quantile: " + str(tdata["STROKE_MORT"].quantile(0.5)))
	print("60th quantile: " + str(tdata["STROKE_MORT"].quantile(0.6)))
	print("70th quantile: " + str(tdata["STROKE_MORT"].quantile(0.7)))
	print("80th quantile: " + str(tdata["STROKE_MORT"].quantile(0.8)))
	
	#tcolors = ["#ff0000", "#00ff00", "#0000ff", "#6a0dad", "#ffff00", "#4e32a8", "#ff0000", "#00ff00", "#0000ff", "#6a0dad", "#ffff00", "#4e32a8", "#ff0000", "#00ff00", "#0000ff", "#6a0dad", "#ffff00", "#4e32a8"]
	
	tcolors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

	x = []
	fips_list = []
	for tindex in tdata.index:
		tval = tdata.loc[tindex, "STROKE_MORT"]
		x.append(tval)
		fips_list.append(tindex)

	input_data = list(zip(x, np.zeros(len(x))))
	X = np.array(input_data, dtype=np.float)
	bandwidth = estimate_bandwidth(X, quantile=0.1)
	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
	ms.fit(X)
	labels = ms.labels_
	cluster_centers = ms.cluster_centers_

	ttable = []
	sum_table = defaultdict(float)
	count_table = defaultdict(int)
	mean_table = defaultdict(float)
	for i in range(0, len(X)):
		ttable.append([fips_list[i], labels[i]])
		mean_table[labels[i]] += x[i]
		count_table[labels[i]] += 1

	for tlabel in sorted(mean_table):
		tmean = sum_table[tlabel] / count_table[tlabel]
		print("Cluster: " + str(tlabel) + "\tMean: " + str(tmean))

	clusterframe = pd.DataFrame(ttable, columns=["fips", "cluster"])
	clusterframe = clusterframe.set_index("fips")

	labels_unique = np.unique(labels)
	n_clusters_ = len(labels_unique)
	for k in range(n_clusters_):
		my_members = labels == k
		print("cluster {0}: {1}".format(k, X[my_members, 0]))

	#json_data = LoadMapJSON(tmode)
	plt.close()
	fig = plt.figure(figsize=(25, 16))
	ax = fig.gca()
	for titem in json_data["features"]:
		tfips = titem["properties"]["GEOID"]
		if tfips in clusterframe.index:
			#print("FIPS Code: " + tfips)
			if int(tfips[0:1]) < 6:
				#print("FIPS found")
				color_index = clusterframe.loc[tfips, "cluster"]
				#if forceBinary == True:
				#	if color_index > 0:
				#		color_index = 1
				get_color = tcolors[color_index]
				poly = titem["geometry"]
				if str(poly) != "None":
					#print(tfips + "\t" + str(get_color))
					ax.add_patch(PolygonPatch(poly, fc=get_color, ec=get_color, alpha=1, zorder=2))
	
	#ax.axis('scaled')
	#ax.set_xlim(-180, -60)
	#ax.set_ylim(20, 80)
	ax.set_facecolor('xkcd:white')
	ax.set_xlim(-130, -60)
	ax.set_ylim(23, 50)

	#plt.xlabel("Longitude")
	#plt.ylabel("Latitude")
	plt.axis("off")

	fig.tight_layout()
	plt.savefig("stroke_meanshift.png")


def ShowMatchingCluster(datatable, numclust, tfile, json_data, reference_map):

	def GetClusterJaccard(clustdata, countybelt, totalcnt):
	
		# Jaccard = intersection / union
		tintersect = 0
		tunion = 0
		jaccard_clust = defaultdict(str)
		unionlist = defaultdict(str)
		
		# get list of clusters
		clustlist = defaultdict(str)
		for tfips in clustdata:
			clustlist[clustdata[tfips]] = ""
			
		# get number of counties in stroke belt
		list2cnt = 0
		for tid in countybelt:
			list2cnt += 1

		# iterate through each cluster
		max_jaccard = 0
		pval_max = 1
		max_clust = 0
		for clustnum in clustlist:
		
			curr_clustdata = defaultdict(str)
			
			# get FIPS for current cluster
			list1cnt = 0
			tintersect = 0
			tunion = 0
			for tfips in clustdata:
				#print(tfips + "\t" + clustdata[tfips] + "\t" + clustnum)
				if str(clustdata[tfips]) == str(clustnum):
					curr_clustdata[tfips] = "1"
					list1cnt += 1

			# debug
			#print("County belt: " + str(countybelt)[0:500])
			#print("Cluster belt: " + str(curr_clustdata)[0:500])
					
			# calculate Jaccard for cluster
			for tfips in curr_clustdata:
				unionlist[tfips] = ''
				if tfips in countybelt:
					tintersect += 1
			for tfips in countybelt:
				unionlist[tfips] = ''
			for tfips in unionlist:
				tunion += 1
						
			# calculate p-value with FET
			#print("Number of counties in cluster: " + str(list1cnt))
			#print("Number of counties in stroke belt: " + str(list2cnt))
			#print("Number intersecting: " + str(tintersect))
			#print("Number total counties: " + str(totalcnt))
			pval = CalcFisherPVal(list1cnt, list2cnt, tintersect, totalcnt)
						
			tjaccard = tintersect / tunion
			jaccard_clust[clustnum] = str(tjaccard)
			
			print("Cluster: " + str(clustnum) + "\t" + str(tjaccard) + "\t" + str(pval))
			
			if tjaccard > max_jaccard:
				max_jaccard = tjaccard
				pval_max = pval
				max_clust = clustnum
		
		return max_jaccard, pval_max, max_clust


	# make filenames
	pcafile = tfile + ".pca.png"
	kfile = tfile + ".kmeans.pca.png"
	cluster_file = tfile + ".kmeans.pca.clusters.png"
	mapfile = tfile + ".map.png"
	cluster_data_file = tfile + ".cluster.tsv"
	
	# PCA
	pca = PCA()
	Xt = pca.fit_transform(datatable)
	PCA_components = pd.DataFrame(Xt)
	model = KMeans(n_clusters = numclust)
	model.fit(PCA_components.iloc[:,:2])
	labels = model.predict(PCA_components.iloc[:,:2])

	# store clusters in dataframe
	labellist = defaultdict(str)
	clustertable = []
	for i, name in enumerate(datatable.index):
		tfips = str(name)
		if len(tfips) == 10:
			tfips = '0' + tfips
		if len(tfips) == 4:
			tfips = '0' + tfips
		clustertable.append([name, labels[i]])
		if labels[i] not in labellist:
			labellist[labels[i]] = ""

	clusterframe = pd.DataFrame(clustertable, columns=["county", "cluster"])
	clusterframe = clusterframe.set_index("county")
	#clusterframe.to_csv(cluster_data_file, sep="\t")

	clustdata = defaultdict(int)
	for tfips in clusterframe.index:
		clustdata[tfips] = clusterframe.loc[tfips, "cluster"]

	rawdata2 = LoadClusterData(reference_map)
	refbelt = defaultdict(int)
	for tid in rawdata2:
		if str(rawdata2[tid]) == "1":
			refbelt[tid] = 1

	# Check each cluster to find best match to reference based on Jaccard
	countycnt = 3141
	tjaccard, tpval, maxclust = GetClusterJaccard(clustdata, refbelt, countycnt)
	print("Top cluster: " + str(maxclust) + "\t" + str(tjaccard) + "\t" + str(tpval))
	
	# now set colors based on maxclust
	cluster_mapping = defaultdict(str)
	for tfips in clustdata:
		cluster_mapping[tfips] = 0
		if clustdata[tfips] == maxclust:
			cluster_mapping[tfips] = 1
		
	# set colors
	tcolors = ["#E0E0E0", "#0000FF"]
	
	#json_data = LoadMapJSON(tmode)
	plt.close()
	fig = plt.figure(figsize=(25, 16))
	ax = fig.gca()
	for titem in json_data["features"]:
		tfips = titem["properties"]["GEOID"]
		if tfips in clusterframe.index:
			#print("FIPS Code: " + tfips)
			if int(tfips[0:1]) < 6:
				#print("FIPS found")
				color_index = cluster_mapping[tfips]
				get_color = tcolors[color_index]
				poly = titem["geometry"]
				if str(poly) != "None":
					#print(tfips + "\t" + str(get_color))
					ax.add_patch(PolygonPatch(poly, fc=get_color, ec=get_color, alpha=1, zorder=2))
	
	#ax.axis('scaled')
	#ax.set_xlim(-180, -60)
	#ax.set_ylim(20, 80)
	ax.set_facecolor('xkcd:white')
	ax.set_xlim(-130, -60)
	ax.set_ylim(23, 50)

	#plt.xlabel("Longitude")
	#plt.ylabel("Latitude")
	plt.axis("off")

	fig.tight_layout()
	plt.savefig(mapfile)


def ShowTractClustersOnMap(datatable, numclust, tfile, tmode, json_data, showmap=True, forceBinary=False, colorCluster=0):

	#print("Making File: " + tfile)

	# make filenames
	pcafile = tfile + ".pca.png"
	kfile = tfile + ".kmeans.pca.png"
	cluster_file = tfile + ".kmeans.pca.clusters.png"
	mapfile = tfile + ".map.png"
	cluster_data_file = tfile + ".cluster.tsv"
	
	#if (showmap==False):
	#	print("ShowTractClustersOnMap Running (nomap) calculation for " + tfile)
	#if (showmap==True):
	#	print("ShowTractClustersOnMap Running (MAP) calculation for " + tfile)
	
	#print(str(datatable.columns))
	
	num_cols = len(datatable.columns)
	
	# PCA
	#datatable = datatable.replace(np.nan, 0)
	#norm_datatable = normalize(datatable, axis=0)
	#datatable = pd.DataFrame(norm_datatable, columns=datatable.columns, index=datatable.index)
	pca = PCA()
	Xt = pca.fit_transform(datatable)
	PCA_components = pd.DataFrame(Xt)
	model = KMeans(n_clusters = numclust, random_state = 0)
	model.fit(PCA_components.iloc[:,:2])
	labels = model.predict(PCA_components.iloc[:,:2])

	# print loadings
	ttable = []
	loadings = pca.components_ * np.sqrt(pca.explained_variance_)
	tcols = list(datatable.columns.values)
	#f = open(tfile + ".loadings.csv", "w")
	for x in range(0, len(tcols)):
		nlabel = tcols[x].lower().strip()
		olabel = nlabel
		#tline = olabel + "\t" + str(loadings[0][x]) + "\t" + str(loadings[1][x])
		if (num_cols == 1):
			trow = [olabel, loadings[0][x]]
		if (num_cols > 1):
			trow = [olabel, loadings[0][x], loadings[1][x]]
		if len(datatable.columns) > 2:
			#tline = olabel + "\t" + str(loadings[0][x]) + "\t" + str(loadings[1][x]) + "\t" + str(loadings[2][x])
			if (num_cols == 1):
				trow = [olabel, loadings[0][x]]
			if (num_cols > 1):
				trow = [olabel, loadings[0][x], loadings[1][x], loadings[2][x]]

		#f.write(tline)
		#print(tline)
		ttable.append(trow)
	#f.close()

	#tcols = ["feature", "PC1", "PC2"]
	#if len(datatable.columns) > 2:
	#	tcols = ["feature", "PC1", "PC2", "PC3"]
	#clusterframe = pd.DataFrame(ttable, columns=tcols)
	#clusterframe = clusterframe.set_index("feature")
	#clusterframe.sort_values(by=['feature'], ascending=False)
	#print(clusterframe)
	
	# Now Cluster PCA Data
	#epa_clust = PCA_Kmeans_Tracts(Xt_epa, numclust, datatable, kfile, cluster_file)

	# store clusters in dataframe
	clustertable = []
	for i, name in enumerate(datatable.index):
		tfips = str(name)
		if len(tfips) == 10:
			tfips = '0' + tfips
		if len(tfips) == 4:
			tfips = '0' + tfips
		clustertable.append([name, labels[i]])

	clusterframe = pd.DataFrame(clustertable, columns=["county", "cluster"])
	clusterframe = clusterframe.set_index("county")
	clusterframe.to_csv(cluster_data_file, sep="\t")

	if showmap == True:
	
		# set colors
		tcolors = ["#ff0000", "#00ff00", "#0000ff", "#6a0dad", "#ffff00", "#4e32a8", "#00ffff", "#ff00ff"]

		# get which cluster has more FIPS codes in it (larger in size) - set this as the gray background
		# and set the smaller cluster as blue
		clust0 = 0
		clust1 = 0
		clust2 = 0
		clust3 = 0
		if numclust == 2:
			for tfips in clusterframe.index:
				tstate = tfips[0:2]
				tclust = clusterframe.loc[tfips, "cluster"]
				if (tclust == 0):
					if (tstate in strokebelt_states):
						clust0 += 1
				if (tclust == 1):
					if (tstate in strokebelt_states):
						clust1 += 1
				#print(tfips + "\t" + tstate + "\t" + "Cluster 0: " + str(clust0) + " and Cluster 1: " + str(clust1))
			if (tmode == "county"):
				if (clust0 < clust1):
					tcolors = ["#f5f5f5", "blue"]
				if (clust0 >= clust1):
					tcolors = ["blue", "#f5f5f5"]
			if (tmode == "tract"):
				if (clust0 >= clust1):
					tcolors = ["blue", "#f5f5f5"]
				if (clust0 < clust1):
					tcolors = ["#f5f5f5", "blue"]

		if (numclust > 2) and (forceBinary == True):
				
			for tfips in clusterframe.index:
				tclust = clusterframe.loc[tfips, "cluster"]
				if (tclust == 0):
					clust0 += 1
				if (tclust == 1):
					clust1 += 1
				if (tclust == 2):
					clust2 += 1
				#if (tclust == 3):
				#	clust3 += 1

				cmax = 0
				max_index = 0
				#clustvals = [clust0, clust1, clust2, clust3]
				clustvals = [clust0, clust1, clust2]
				for x in range(0, numclust):
					if clustvals[x] > cmax:
						cmax = clustvals[x]
						max_index = x
			
			# now re-index into 2 clusters
			# note - reset max_index to zero:
			max_index = 0
			
			clust0 = 0
			clust1 = 0
			for tfips in clusterframe.index:
				tclust = clusterframe.loc[tfips, "cluster"]
				tstate = tfips[0:2]
				if (tclust == max_index):
					clusterframe.loc[tfips, "cluster"] = 1
					if tstate in strokebelt_states:
						clust1 += 1

				if (tclust != max_index):
					clusterframe.loc[tfips, "cluster"] = 0
					if tstate in strokebelt_states:
						clust0 += 1
			
			if (tmode == "county"):
				if (clust0 >= clust1):
					tcolors = ["blue", "#f5f5f5"]
				if (clust0 < clust1):
					tcolors = ["#f5f5f5", "blue"]
				if colorCluster == 1:
					tcolors = ["#f5f5f5", "blue"]
			if (tmode == "tract"):
				if (clust0 >= clust1):
					tcolors = ["blue", "#f5f5f5"]
				if (clust0 < clust1):
					tcolors = ["#f5f5f5", "blue"]
					
			# re-save after re-indexing the clustering
			clusterframe.to_csv(cluster_data_file, sep="\t")
		
		#json_data = LoadMapJSON(tmode)
		plt.close()
		fig = plt.figure(figsize=(25, 16))
		ax = fig.gca()
		for titem in json_data["features"]:
			tfips = titem["properties"]["GEOID"]
			if tfips in clusterframe.index:
				#print("FIPS Code: " + tfips)
				if int(tfips[0:1]) < 6:
					#print("FIPS found")
					color_index = clusterframe.loc[tfips, "cluster"]
					if forceBinary == True:
						if color_index > 0:
							color_index = 1
					get_color = tcolors[color_index]
					poly = titem["geometry"]
					if str(poly) != "None":
						#print(tfips + "\t" + str(get_color))
						ax.add_patch(PolygonPatch(poly, fc=get_color, ec=get_color, alpha=1, zorder=2))
		
		#ax.axis('scaled')
		#ax.set_xlim(-180, -60)
		#ax.set_ylim(20, 80)
		ax.set_facecolor('xkcd:white')
		ax.set_xlim(-130, -60)
		ax.set_ylim(23, 50)

		#plt.xlabel("Longitude")
		#plt.ylabel("Latitude")
		plt.axis("off")

		fig.tight_layout()
		print("Saving map file: " + mapfile)
		plt.savefig(mapfile)

def GetCanonicalStrokeBelt(tmode):

	# Download from here for 2017-2019 using Stroke mortality, and use the export function
	# Texas: 48
	# Florida: 12
	# strokebelt_states = ["01", "05", "13", "18", "21", "22", "28", "37", "45", "47", "51"]
	
	tcnt = 0
	#tcutoff = 70.6
	found_count = 0
	tcutoff = 78
	tfile = "d:\\apeer_data\\cdc_stroke_mort_2017_2019.csv"
	beltlist = defaultdict(float)
	with open(tfile, "r") as infile:
		csv_reader = csv.reader(infile, delimiter=",")
		for row in csv_reader:
			if tcnt == 0:
				theader = row
			if tcnt > 0:
				for row in csv_reader:
					tfips = row[0]
					tstate = tfips[0:2]
					
					inState = 1
					if tmode == "canonical":
						if tstate not in strokebelt_states:
							inState = 0
					
					if inState == 1:
					#if tstate in strokebelt_states:
						tvalue = 0
						if (row[2] == "") or (row[2] == "-1"):
							tvalue = 0
						else:
							tvalue = float(row[2])
						if tvalue > tcutoff:
							beltlist[tfips] = tvalue
			tcnt += 1
	infile.close()
		
	return beltlist

def GetReferenceMap(tdata, ofile, tcutoff):

	# get 75th percentile cutoff
	x = 0
	curr_col = ""
	cutoff = 0
	for tcol in tdata.columns:
		if x == 0:
			tcutoff = tdata[tcol].quantile(tcutoff)
			curr_col = tcol
		x += 1
	
	print("Cutoff: " + str(tcutoff))
	
	# find all values over the cutoff
	beltlist = defaultdict(float)
	for tindex in tdata.index:
		tval = tdata.loc[tindex, curr_col]
		if tval >= tcutoff:
			beltlist[tindex] = tval

	# store clusters in dataframe
	clustertable = []
	for tfips in tdata.index:
		tclust = 0
		if tfips in beltlist:
			tclust = 1
		#print(tfips + "\t" + str(tclust))
		tfips = str(tfips)
		if len(tfips) == 10:
			tfips = '0' + tfips
		if len(tfips) == 4:
			tfips = '0' + tfips
		clustertable.append([tfips, tclust])

	clusterframe = pd.DataFrame(clustertable, columns=["county", "cluster"])
	clusterframe = clusterframe.set_index("county")
	clusterframe.to_csv(ofile, sep="\t")

	return beltlist
	
def ShowBinaryMap(beltlist, mapfile, json_data, backcolor, forecolor):

	# set colors
	tcolors = ["#ff0000", "#00ff00", "#0000ff", "#6a0dad", "#ffff00"]

	print("ShowBinaryMap - Generating " + mapfile)
	plt.close()
	fig = plt.figure(figsize=(25, 16))
	ax = fig.gca()
	for titem in json_data["features"]:
		tfips = titem["properties"]["GEOID"]
		#print("ShowBinaryMap Debug 2: " + tfips)
		if int(tfips[0:1]) < 6:
			get_color = backcolor
			if tfips in beltlist:
				get_color = forecolor
			#get_color = tcolors[color_index]
			poly = titem["geometry"]
			if str(poly) != "None":
				#print(tfips + "\t" + str(get_color))
				ax.add_patch(PolygonPatch(poly, fc=get_color, ec=get_color, alpha=1, zorder=2))
	
	#ax.axis('scaled')
	#ax.set_xlim(-180, -60)
	#ax.set_ylim(20, 80)
	ax.set_xlim(-130, -60)
	ax.set_ylim(23, 50)
	ax.set_facecolor('xkcd:white')
	#plt.xlabel("Longitude")
	#plt.ylabel("Latitude")

	fig.tight_layout()
	plt.axis("off")
	plt.savefig(mapfile)

def LoadClusterData(tfile):

	clusterdata = defaultdict(str)
	with open(tfile, "r", encoding="latin-1") as infile:
		theader = infile.readline()
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			tfips = ldata[0]
			tclust = ldata[1]
			clusterdata[tfips] = tclust
	infile.close()
	
	return clusterdata

def CalcJaccard(clustdata, countybelt, totalcnt):

	# Jaccard = intersection / union
	tintersect = 0
	tunion = 0
	jaccard_clust = defaultdict(str)
	unionlist = defaultdict(str)
	
	# get list of clusters
	clustlist = defaultdict(str)
	for tfips in clustdata:
		#if int(clustdata[tfips]) >= 0:
		clustlist[clustdata[tfips]] = ""

	# get number of counties in stroke belt
	list2cnt = 0
	for tid in countybelt:
		list2cnt += 1

	# iterate through each cluster
	max_jaccard = 0
	pval_max = 1
	for clustnum in clustlist:
	
		curr_clustdata = defaultdict(str)
		
		# get FIPS for current cluster
		list1cnt = 0
		tintersect = 0
		tunion = 0
		for tfips in clustdata:
			if clustdata[tfips] == clustnum:
				curr_clustdata[tfips] = "1"
				list1cnt += 1
	
		# calculate Jaccard for cluster
		for tfips in curr_clustdata:
			unionlist[tfips] = ''
			if tfips in countybelt:
				tintersect += 1
		for tfips in countybelt:
			unionlist[tfips] = ''
		for tfips in unionlist:
			tunion += 1
					
		# calculate p-value with FET
		#print("Number of counties in cluster: " + str(list1cnt))
		#print("Number of counties in stroke belt: " + str(list2cnt))
		#print("Number intersecting: " + str(tintersect))
		#print("Number total counties: " + str(totalcnt))
		# https://rdrr.io/bioc/GeneOverlap/man/GeneOverlap.html
		pval = CalcFisherPVal(list1cnt, list2cnt, tintersect, totalcnt)
		
		tjaccard = tintersect / tunion
		jaccard_clust[clustnum] = str(tjaccard)
		#print("Cluster: " + clustnum + "\t" + str(tjaccard))
		if tjaccard > max_jaccard:
			max_jaccard = tjaccard
			pval_max = pval
	
	return max_jaccard, pval_max

def CalcFisherPVal(list1, list2, overlap, popsize):

	# http://crazyhottommy.blogspot.com/2014/10/test-significance-of-overlapping-genes.html
	#list1 = 3000
	#list2 = 400
	#overlap = 100
	#popsize = 15000
	# set up table:
	
	# additional explanation here: https://rdrr.io/bioc/GeneOverlap/man/GeneOverlap.html
	# a = # that are the same between lists
	a = popsize - list1 - list2 + overlap
	# b = # unique to list 2
	b = list2 - overlap
	# c = # unique to list 1
	c = list1 - overlap
	# d = # in common
	d = overlap
	#fisher.test(matrix(c(a,b,c,d),nrow=2,byrow=T),alternative="greater")

	odds_ratio = 1
	p_value = 99
	if (a > 0) and (b > 0) and (c > 0) and (d > 0):
		data = [[a, b], [c, d]]
	  
		# performing fishers exact test on the data
		odd_ratio, p_value = stats.fisher_exact(data)
		#print('odd ratio is : ' + str(odd_ratio))
		#print('p_value is : ' + str(p_value))
	
	return p_value	

def LoadLifeExpectancy(merge_data, ofile, tquant, tmode):

	# get file
	epa_data = "d:\\apeer_data\\EJSCREEN_2021_USPR.csv\\EJSCREEN_2021_USPR.csv"
	raw_data = defaultdict(int)
	#raw_data = pd.read_csv(epa_data, header=0, low_memory=False)
	#raw_data.set_index("ID", inplace=True)

	# https://www.cdc.gov/nchs/nvss/usaleep/usaleep.html
	# https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NVSS/USALEEP/CSV/US_A.CSV
	tcnt = 0
	tfile = "US_A.CSV"
	lifedata = defaultdict(float)
	with open(tfile, "r") as infile:
		csv_reader = csv.reader(infile, delimiter=",")
		for row in csv_reader:
			if tcnt > 0:
				tfips = row[0]
				if len(tfips) == 4:
					tfips = '0' + tfips
				if len(tfips) == 10:
					tfips = '0' + tfips
				tlife = row[4]
				lifedata[tfips] = float(tlife)
				#print(tfips + "\t" + str(lifedata[tfips]))
			tcnt += 1
	infile.close()
	
	# get county pop totals for weights
	countytotal = defaultdict(float)
	tractpop = defaultdict(float)
	county_lifeexpect = defaultdict(float)
	for tfips in lifedata:
		tcounty = str(tfips)[0:5]
		ttract = str(tfips)[0:11]
		countytotal[tcounty] += lifedata[tfips]
		tractpop[ttract] += lifedata[tfips]
	
	for tfips in lifedata:
		cfips = tfips[0:5]
		tweight = tractpop[tfips] / countytotal[cfips]
		county_lifeexpect[cfips] += lifedata[tfips] * tweight
		
	if tmode == "merge":
	
		for tfips in county_lifeexpect:
			merge_data.loc[tfips, "LIFE_EXPECT"] = county_lifeexpect[tfips]
			
		merge_data = merge_data.fillna(0)
			
		return merge_data

	if tmode != "merge":
		
		# convert dictionary to dataframe, get quantile cutoff
		#life_mapdata = pd.DataFrame.from_dict(county_lifeexpect, orient='index')
		life_mapdata = pd.DataFrame()
		for tfips in county_lifeexpect:
			life_mapdata.loc[tfips, "LIFE_EXPECT"] = county_lifeexpect[tfips]
		
		#life_mapdata.columns = ["LIFE_EXPECT"]
		tcutoff = life_mapdata["LIFE_EXPECT"].quantile(tquant)
		clustertable = []
		beltlist = defaultdict(float)
		#print("Life expectancy cutoff: " + str(tcutoff))
		
		for tfips in life_mapdata.index:
			tclust = 0
			tval = life_mapdata.loc[tfips, "LIFE_EXPECT"]
			if tval <= tcutoff:
				tclust = 1
				beltlist[tfips] = tval
			clustertable.append([tfips, tclust])
			
		clusterframe = pd.DataFrame(clustertable, columns=["county", "cluster"])
		clusterframe = clusterframe.set_index("county")
		if ofile != "none":
			clusterframe.to_csv(ofile, sep="\t")
			
		if tmode == "threshold":

			return beltlist
		
		if tmode == "merge_threshold_data":
		
			for x in range(0, len(clusterframe.index)):
				tfips = clusterframe.index[x]
				merge_data.loc[tfips, "LIFE_EXPECT_THRESHOLD"] = clusterframe.loc[tfips, "cluster"]
			
			merge_data = merge_data.fillna(0)
				
			return merge_data

def LoadStrokeBelt(merge_data_counties, ofile, tcutoff, tmode):

	stroke_states = ["01", "05", "13", "18", "21", "22", "28", "37", "45", "47", "51"]

	# merge data
	for tfips in merge_data_counties.index:
		tval = 0
		tstate = tfips[0:2]
		if tstate in stroke_states:
			tval = 1
		merge_data_counties.loc[tfips, "STROKE_BELT"] = tval
		
	if tmode == "merge":
		return merge_data_counties
	
	if tmode != "merge":
		clustertable = []
		beltlist = defaultdict(float)
		for tfips in merge_data_counties.index:
			tstate = tfips[0:2]
			tval = 0
			if tstate in stroke_states:
				tval = 1
				beltlist[tfips] = tval
			clustertable.append([tfips, tval])
			
		clusterframe = pd.DataFrame(clustertable, columns=["county", "cluster"])
		clusterframe = clusterframe.set_index("county")
		if ofile != "none":
			clusterframe.to_csv(ofile, sep="\t")

		if tmode == "threshold":
			return beltlist
	
		if tmode == "merge_threshold_data":
			for x in range(0, len(clusterframe.index)):
				tfips = clusterframe.index[x]
				merge_data_counties.loc[tfips, "STROKE_BELT_THRESHOLD"] = clusterframe.loc[tfips, "cluster"]
				
			return merge_data_counties
	
	
def LoadStrokeMort(merge_data_counties, ofile, tcutoff, tmode):

	# https://nccd.cdc.gov/DHDSPAtlas/?state=County&class=1&subclass=6&theme=3&filters=[[9,1],[2,1],[3,1],[4,1],[7,1]]

	# https://www.cdc.gov/nchs/nvss/usaleep/usaleep.html
	# https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NVSS/USALEEP/CSV/US_A.CSV
	tcnt = 0
	tfile = "cdc_stroke_report.csv"
	strokedata = defaultdict(float)
	with open(tfile, "r") as infile:
		csv_reader = csv.reader(infile, delimiter=",")
		for row in csv_reader:
			if tcnt > 0:
				tfips = row[0]
				if len(tfips) == 4:
					tfips = '0' + tfips
				if len(tfips) == 10:
					tfips = '0' + tfips
				tlife = row[2]
				strokedata[tfips] = float(tlife)	
			tcnt += 1
	infile.close()

	# convert into Pandas dataframe
	#stroke_table = pd.DataFrame.from_dict(strokedata, orient='index')
	#stroke_table.columns = ["STROKE"]

	# merge data
	for tfips in merge_data_counties.index:
		tval = 0
		if tfips in strokedata:
			tval = strokedata[tfips]
		merge_data_counties.loc[tfips, "STROKE_MORT"] = tval
		
	if tmode == "merge":

		return merge_data_counties
	
	if tmode != "merge":
		
		clustertable = []
		beltlist = defaultdict(float)
		tcutoff = merge_data_counties["STROKE_MORT"].quantile(tcutoff)
		print("Stroke cutoff: " + str(tcutoff))
		for tfips in merge_data_counties.index:
			tclust = 0
			tval = merge_data_counties.loc[tfips, "STROKE_MORT"]
			if tval >= tcutoff:
				tclust = 1
				beltlist[tfips] = tval
			clustertable.append([tfips, tclust])
			
		clusterframe = pd.DataFrame(clustertable, columns=["county", "cluster"])
		clusterframe = clusterframe.set_index("county")
		clusterframe.to_csv(ofile, sep="\t")

		if tmode == "threshold":

			return beltlist
	
		if tmode == "merge_threshold_data":
		
			for x in range(0, len(clusterframe.index)):
				tfips = clusterframe.index[x]
				merge_data_counties.loc[tfips, "STROKE_MORT_THRESHOLD"] = clusterframe.loc[tfips, "cluster"]
				
			return merge_data_counties

def MergeDiscreteVals(cutoff_data, merge_data_counties, tcol, tcutoff):

	# https://nccd.cdc.gov/DHDSPAtlas/?state=County&class=1&subclass=6&theme=3&filters=[[9,1],[2,1],[3,1],[4,1],[7,1]]

	# https://www.cdc.gov/nchs/nvss/usaleep/usaleep.html
	# https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NVSS/USALEEP/CSV/US_A.CSV

	# convert into Pandas dataframe
	#stroke_table = pd.DataFrame.from_dict(strokedata, orient='index')
	#stroke_table.columns = ["STROKE"]

	clustertable = []
	beltlist = defaultdict(float)	
	tcutoff = cutoff_data[tcol].quantile(tcutoff)

	for tfips in cutoff_data.index:
		tclust = 0
		#tval = cutoff_data.loc[tfips, "STROKE_MORT"]
		tval = cutoff_data.loc[tfips, tcol]
		if tval >= tcutoff:
			tclust = 1
			beltlist[tfips] = tval
		clustertable.append([tfips, tclust])
		
	clusterframe = pd.DataFrame(clustertable, columns=["county", "cluster"])
	clusterframe = clusterframe.set_index("county")
	#clusterframe.to_csv(ofile, sep="\t")

	for x in range(0, len(clusterframe.index)):
		tfips = clusterframe.index[x]
		merge_data_counties.loc[tfips, tcol + "_THRESHOLD"] = clusterframe.loc[tfips, "cluster"]
	
	return merge_data_counties
	
def BuildMergeFileList():

	filelist = []

	nclust = 3	
	total_list = ["Arthritis", "Hypertension", "Coronary Heart Disease", "COPD", "Depression", "Diabetes", "Renal Disease", "Obesity", "Stroke", "STROKE_MORT", "Cancer", "Asthma"]
	for pairx in range(0, len(total_list)):
		tfile = "d:\\apeer\\maps\\lancet_disease_pair." + total_list[pairx] + ".LIFE_EXPECT_2019.county_cluster." + str(nclust) + ".cluster.tsv"
		#tfile2 = "lancet.canonical." + total_list[pairx] + ".0.6.binarymap.png"
		tfile = tfile.replace(" ", "")
		#if total_list[pairx] == "STROKE_MORT":
		#	tfile2 += "lancet.canonical.stroke_data..0.7.binarymap.png"
		filelist.append(tfile)
		
	return filelist

def BuildFileListOld(tquant):

	filelist = []

	#quant_cutoffs = [0.6, 0.7, 0.8, 0.9]
	#quant_cutoffs = [0.7]
	#for tquant in quant_cutoffs:
	for tdisease in disease_columns:
		ofile = "lancet.canonical." + tdisease + "." + str(tquant) + ".binarymap.tsv"
		filelist.append(ofile)
	
	# get life expectancy map
	#quant_cutoffs = [0.2, 0.3, 0.4, 0.5]
	#for tquant in quant_cutoffs:
	#	ofile = "lancet.canonical.life_expect." + str(tquant) + ".binarymap.tsv"
	#	filelist.append(ofile)
	
	# stroke map
	#quant_cutoffs = [0.4, 0.5, 0.6, 0.7]
	#quant_cutoffs = [0.7]
	#for tquant in quant_cutoffs:
	ofile = "lancet.canonical.stroke_data." + "." + str(tquant) + ".binarymap.tsv"
	filelist.append(ofile)

	# stroke belt
	ofile = "lancet.canonical.stroke_belt.binarymap.tsv"
	filelist.append(ofile)

	return filelist

def BuildFileList():

	filelist = defaultdict(lambda: defaultdict(int))

	quant_cutoffs = [0.6, 0.7, 0.8, 0.9]
	#quant_cutoffs = [0.7]
	for tquant in quant_cutoffs:
		for tdisease in disease_columns:
			ofile = "lancet.canonical." + tdisease + "." + str(tquant) + ".binarymap.tsv"
			filelist[tquant][ofile] = 1
				
	# stroke map
	for tquant in quant_cutoffs:
		ofile = "lancet.canonical.stroke_data." + "." + str(tquant) + ".binarymap.tsv"
		filelist[tquant][ofile] = 1
		ofile = "lancet.canonical.stroke_belt.binarymap.tsv"
		filelist[tquant][ofile] = 1

	# stroke belt
	#ofile = "lancet.canonical.stroke_belt.binarymap.tsv"
	#filelist.append(ofile)

	return filelist

def GetPValCutoff(input_file):

	# get p-value threshold
	n = 0
	unique_table = defaultdict(float)
	
	# input_file = "lancet_figure2.tsv"
	with open(input_file, "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			colcnt = len(ldata)
			tid = ldata[0] + "\t" + ldata[1] + "\t" + ldata[2] + "\t" + ldata[3]
			unique_table[tid] = float(ldata[4])
	infile.close()

	for tid in unique_table:
		n += 1

	pval_cutoff = 0.01 / n
	#print("Bonferroni-adjusted p-value threshold of significance for n=" + str(n) + " p=" + str(pval_cutoff))

	return pval_cutoff

def GetBestPercentile(tdata, json_data_county):

	# get a list of counties from the stroke belt
	stroke_belt_ref = defaultdict(int)
	for tfips in tdata.index:
		tstate = str(tfips)[0:2]
		if tstate in strokebelt_states:
			stroke_belt_ref[tfips] = 1
	
	# now go through a list of percentiles for stroke mortality, find highest Jaccard
	x = 0
	curr_col = ""
	cutoff = 0

	prevJ = 0
	cutoff_list = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
	for tcutoff in cutoff_list:
		cutoff_val = tdata["STROKE_MORT"].quantile(tcutoff)		
		county_list = defaultdict(int)
		for tfips in tdata.index:
			tval = tdata.loc[tfips, "STROKE_MORT"]
			if tval >= cutoff_val:
				county_list[tfips] = 1

		# calculate jaccard
		jval = 0
		tunion = 0
		tintersect = 0
		tunion_list = defaultdict(int)
		tintersect_list = defaultdict(int)

		for tfips in stroke_belt_ref:
			tunion_list[tfips] = 1
		for tfips in county_list:
			tunion_list[tfips] = 1
		for tfips in tunion_list:
			tunion += 1
			if (tfips in stroke_belt_ref) and (tfips in county_list):
				tintersect += 1

		jval = tintersect / tunion
		
		# calculate forbes coefficient
		# F = a N/[(a + b) (a + c)]
		# https://bio.mq.edu.au/~jalroy/Forbes.html
		# N = a + b + c + d
		# a = the number found in both lists
		# b = the number found only in the first list
		# c = the number found only in the second
		# d = the number found in neither one.
		N = 0
		a = 0
		b = 0
		c = 0
		d = 0
		totalN = 0
		for tfips in tdata.index:
			totalN += 1
			if (tfips in stroke_belt_ref) and (tfips in county_list):
				a += 1
			if (tfips in stroke_belt_ref) and (tfips not in county_list):
				b += 1
			if (tfips not in stroke_belt_ref) and (tfips in county_list):
				c += 1
			if (tfips not in stroke_belt_ref) and (tfips not in county_list):
				d += 1
		N = a + b + c + d
		n = a + b + c
		tForbes = a * N / ((a + b) * (a + c))
		cForbes = (a * (n + math.sqrt(n))) / (a * (n + math.sqrt(n)) + ((3/2) * b * c))
		
		# Jaccard change for elbow method
		deltaJ = jval - prevJ
		
		print(str(tcutoff) + "\t" + str(cutoff_val) + "\t" + str(tunion) + "\t" + str(tintersect) + "\t" + str(jval) + "\t" + str(deltaJ))
		prevJ = jval

		# show map
		ShowBinaryMap(county_list, "new_stroke_belt." + str(tcutoff) + ".png", json_data_county, light_gray, "blue")
		
		county_list.clear()
		

def GetTopRankJaccard(datatable, input_file, json_data_county, output_file, norm_chemical_data):

	# get p-value threshold
	pval_cutoff = GetPValCutoff(input_file)

	# output_file = "lancet_figure2_top_pollutants.tsv"
	f = open(output_file, "w")
	
	print("Loading Jaccard data")
	score_table = defaultdict(lambda: defaultdict(float))
	cluster_table = defaultdict(lambda: defaultdict(float))
	with open(input_file, "r") as infile:
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
	rank_cutoff = 3 + 1
	for bdisease in score_table:
		maxj = 0
		tcnt = 1
		for titem in sorted(score_table[bdisease], key=score_table[bdisease].get, reverse=True):
			if tcnt == 1:
				maxj = score_table[bdisease][titem]
			tcnt += 1

		tthreshold = maxj - (maxj * 0.01)
		tcnt = 1
		# get a list of pollutants with max score
		pollution_list = []
		for titem in sorted(score_table[bdisease], key=score_table[bdisease].get, reverse=True):
			tscore = score_table[bdisease][titem]
			#if (tscore >= tthreshold):
			#	pollutant_pairs = titem.split('|')
			#	if pollutant_pairs[0] not in pollution_list:
			#		pollution_list.append(pollutant_pairs[0])
			#	if pollutant_pairs[1] not in pollution_list:
			#		pollution_list.append(pollutant_pairs[1])		
			#print(str(tcnt) + "\t" + bdisease + "\t" + titem + "\t" + str(score_table[bdisease][titem]))

			'''
			chem_data = titem.split('|')
			chem1 = chem_data[0]
			chem2 = chem_data[1]
			tclust = int(chem_data[2])
			tdata = norm_chemical_data[[chem1, chem2]]
			tfile = bdisease + "_best_match_" + chem1 + "_" + chem2
			tfile = tfile.replace(' ', '')
			tfile = tfile.replace(',', '')
			tfile = tfile.replace('-', '')
			print("Generating assembled map with J = " + str(maxj) + ": " + tfile)
			#ShowTractClustersOnMap(tdata, tclust, tfile, "county", norm_chemical_data, showmap=True)
			ShowTractClustersOnMap(tdata, tclust, tfile, "county", json_data_county, showmap=True, forceBinary=False)
			'''
			
			if tcnt == 1:
				# Load Reference map data
				tclust = cluster_table[bdisease][titem]
				tline = str(tcnt) + "\t" + bdisease + "\t" + titem + "\t" + str(tclust) + "\t" + str(score_table[bdisease][titem])
				print(tline)
				f.write(tline + "\n")
				tfile = "lancet_final_pairfile_" + bdisease + "_" + str(tcnt)
				ShowMatchingCluster(datatable, cluster_table[bdisease][titem], tfile, json_data_county, bdisease)

			tcnt += 1

	f.close()

def PlotPairwiseMaps(input_file, output_file, json_data_county):

	base_id = ""
	curr_id = base_id

	if output_file == "figure2.html":
		chemical_data = MergeStrokeData("county")
	if output_file == "figure3.html":
		chemical_data = LoadChemicalData("county")
	
	#print(chemical_data.columns)
	#exit()
	
	# output_file = "figure3.html"
	# intput_file = "lancet_figure3_top_pollutants.tsv"
	f = open(output_file, "w")
	with open(input_file, "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			base_id = ldata[0]
			tdata = ldata[2].split('|')
			chem1 = tdata[0]
			chem2 = tdata[1]
			nclust = int(tdata[2])
			nfile = "lancet_airtox_pair." + chem1 + "." + chem2 + ".county_cluster." + str(nclust)
			tfile = mappath + "\\" + nfile
			tfile = tfile.replace(" ", "")
			tfile = tfile.replace(",", "_")
			tfile = tfile.replace("-", "_")				
			print("Pairwise Pollution File: " + tfile)
			
			if (base_id != curr_id) or (base_id == ""):
				f.write('<br><br>' + base_id + "\n")
				
			f.write('<br>' + chem1 + " and " + chem2 + " Clusters=" + str(nclust) + "\n")
			f.write('<br><img src="maps/' + nfile + '">' + "\n")

			cfile = tfile + ".map.png"
			if (os.path.isfile(cfile) == False):
				tdata = chemical_data[[chem1, chem2]]
				ShowTractClustersOnMap(tdata, nclust, tfile, "county", json_data_county, showmap=True)
				
			curr_id = base_id
			
	infile.close()
	f.close()

def GetImage(tfile):

	with cbook.get_sample_data(tfile) as image_file: 
		image = plt.imread(image_file) 
	image_file.close()
	
	return image

def MakeMapPDF(input_file, tlabel, tdisease):

	#filelist = ['d:\\apeer\\maps\\lancet_airtox_pair.1_2_4_TRICHLOROBENZENE.ACETALDEHYDE.county_cluster.2.map.png', 'd:\\apeer\\maps\\lancet_airtox_pair.ACETALDEHYDE.HEXANE.county_cluster.4.map.png', 'd:\\apeer\\maps\\lancet_airtox_pair.1_2_4_TRICHLOROBENZENE.ACETALDEHYDE.county_cluster.2.map.png', 'd:\\apeer\\maps\\lancet_airtox_pair.ACETALDEHYDE.HEXANE.county_cluster.4.map.png']

	filelist = []
	trow = []
	jrow = []
	jaccard_vals = []
	topmaps = []
	baseid = ""
	previd = ""
	with open(input_file, "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			curr_file = ldata[1]
			curr_base_file = curr_file
			curr_base_file = rpath + "\\" + curr_base_file.replace('.tsv', '.png')
			curr_file = curr_file.replace('..', '.')
			fdata = curr_file.split('.')
			pollution_id = fdata[2]
			clust_id = fdata[3] + '.' + fdata[4]
			if curr_file.find("stroke_data") > -1:
				clust_id = fdata[4] + '.' + fdata[5]
			baseid = pollution_id + '.' + clust_id
			#baseid = pollution_id + " (" + fdata[3] + ")"
			fdata = ldata[2].split('|')
			id1 = fdata[0]
			id2 = fdata[1]
			tclust = fdata[2]
			tjaccard = ldata[3]
			nfile = "lancet_airtox_pair." + id1 + "." + id2 + ".county_cluster." + tclust + ".map.png"
			tfile = mappath + "\\" + nfile
			tfile = tfile.replace(" ", "")
			tfile = tfile.replace(",", "_")
			tfile = tfile.replace("-", "_")				
			if pollution_id == tdisease:
				trow.append(tfile)
				jrow.append(tjaccard)
				if curr_base_file not in topmaps:
					topmaps.append(curr_base_file)
				#print(curr_file)
				#print(baseid + "\t" + tfile)
			if (previd != baseid) and (previd != ""):
				if len(trow) > 1:
					filelist.append(trow)
					jaccard_vals.append(jrow)
					trow = []
					jrow = []
			previd = baseid
	infile.close()

	if len(trow) > 1:
		filelist.append(trow)
		jaccard_vals.append(jrow)
		topmaps.append(curr_base_file)

	#for x in range(0, len(filelist)):
	#	print(str(x) + "\t" + str(filelist[x]))
	#exit()
	
	fig, ax = plt.subplots(nrows=6, ncols=4)

	# x = column
	# y = row
	y = 0
	for row in ax:
		x = 0
		for col in row:
			if y == 0:
				timg = GetImage(topmaps[x])
				col.axis("off")
				col.imshow(timg)
				plt.rcParams.update({'font.size': 4})
				fdata = topmaps[x].split('.')
				col.text(1, 1650, fdata[2] + " (" + fdata[3] + '.' + fdata[4] + ")")
				#print(topmaps[x])
			if y > 0:
				y2 = y - 1
				timg = GetImage(filelist[x][y2])
				col.axis("off")
				col.imshow(timg)
				fdata = filelist[x][y2].split('.')
				jval = jaccard_vals[x][y2]
				jval = jval[0:6]
				plt.rcParams.update({'font.size': 6})
				col.text(1, 1500, " J=" + jval)
				plt.rcParams.update({'font.size': 3})
				col.text(1, 1650, fdata[1] + ' vs ')
				col.text(1, 1750, fdata[2])
				#print(str(y2) + "\t" + str(x) + "\t" + filelist[x][y2])
			x += 1
		y += 1

	#plt.title('matplotlib.pyplot.imread() function Example', fontweight ="bold") 
	plt.show() 
	plt.savefig(tdisease + "." + tlabel + ".pdf")

def PlotJaccardNetwork(chemical_data, mainfile, tlabel):

	ifile = mainfile
	#ifile = "lancet_figure3.tsv"
	pval_cutoff = GetPValCutoff(ifile)
	print("P-value cutoff: " + str(pval_cutoff))

	# get p-value threshold
	n = 0
	jvals = defaultdict(float)
	pvals = defaultdict(float)

	# set jaccard using mean for each graph below - not here!
	jaccard_cutoff = 0.25
	tfont = 14
	if ifile == "lancet_figure2.tsv":
		jaccard_cutoff = 0.3
		tfont = 24

	base_disease = defaultdict(int)
	pollution_list = defaultdict(float)
	disease_network = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
	#tfile = "lancet_figure2.tsv"
	
	with open(ifile, "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			colcnt = len(ldata)
			if (ldata[1] in chemical_data.columns) and (ldata[2] in chemical_data.columns):
				tid = ldata[0] + "\t" + ldata[1] + "\t" + ldata[2] + "\t" + ldata[3]
				jvals[tid] = float(ldata[4])
				pvals[tid] = float(ldata[5])
	infile.close()
	
	for mainid in jvals:
		ldata = mainid.split("\t")	
		baseid = ldata[0]
		poll1 = ldata[1]
		poll2 = ldata[2]
		clust = ldata[3]
		jval = float(jvals[mainid])
		pval = float(pvals[mainid])
		
		# build matrix and indices
		base_disease[baseid] = 1
		pollution_list[poll1] = 1
		pollution_list[poll2] = 1
		if (pval <= pval_cutoff):
			disease_network[baseid][poll1][poll2] = jval
			#print("P value: " + str(pval))

	# convert to pandas dataframe
	for baseid in base_disease:

		print("Processing " + baseid)

		median_vals = []	
		collist = []
		ttable = []
		num_jaccard = 0
		jaccard_total = 0
		for poll1 in sorted(pollution_list):
			trow = []
			collist.append(poll1)
			for poll2 in sorted(pollution_list):
				tval = 0
				if (poll1 in disease_network[baseid]):
					if (poll2 in disease_network[baseid][poll1]):
						tval = float(disease_network[baseid][poll1][poll2])
						jaccard_total += tval
						num_jaccard += 1
						median_vals.append(tval)

				#tval = disease_network[baseid][poll1][poll2]
				nval = 0
				if tval != "":
					nval = float(tval)
				trow.append(nval)
			ttable.append(trow)
		finaldata = pd.DataFrame(ttable, columns=collist, index=collist)
	
		# set jaccard cutoff:
		#jaccard_cutoff = 0
		#jaccard_cutoff = jaccard_total / num_jaccard
		#print("Median vals: " + str(median_vals))
		#jaccard_cutoff = statistics.median(median_vals)

		# 69th percentile - mean + 0.5SD
		data_list = np.array(median_vals)
		jaccard_cutoff = np.percentile(data_list, 84)
		print("95th Percentile Jaccard cutoff: " + str(jaccard_cutoff))
					
		# set files
		opath = "d:\\apeer\\networks\\"
		density_file = opath + tlabel + "_" + "density_" + baseid + ".png"
		network_file = opath + tlabel + "_" + "network_" + baseid + ".jpg"
		cluster_file = opath + tlabel + "_" + "cluster_" + baseid + ".pdf"
		stats_file = opath + tlabel + "_" + "stats_" + baseid + ".txt"

		# draw out cluster map
		cmap = LinearSegmentedColormap.from_list('RedBlackGreen', ['lime', 'black', 'red'])
		plt.subplots_adjust(bottom=0.2, top=0.8, left=0.2, right=0.9)
		sns.set(font_scale=0.2)
		finalimg = sns.clustermap(finaldata, cmap=cmap, yticklabels=True, xticklabels=True)
		plt.savefig(cluster_file, dpi=1200)
		plt.close()

		# build links
		links = finaldata.stack().reset_index()
		links.columns = ['var1', 'var2', 'value']
		#print(links)
		 
		# Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)
		links_filtered=links.loc[ (links['value'] >= jaccard_cutoff) & (links['var1'] != links['var2']) ]
		G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
		 
		# Plot the network:
		plt.close()
		fig = plt.figure(1, figsize=(20, 16), dpi=300)
		#pos = nx.draw_spectral(G, k=0.25, iterations=50)
		#nx.draw(G, pos, with_labels=True, font_weight='normal', font_size=10, linewidths=1, edge_color='grey', node_color='blue')

		df = pd.DataFrame(index=G.nodes(), columns=G.nodes())
		for row, data in nx.shortest_path_length(G):
			for col, dist in data.items():
				df.loc[row,col] = dist

		df = df.fillna(df.max().max())
		layout = nx.kamada_kawai_layout(G, dist=df.to_dict())

		nx.draw(G, layout, with_labels=True, font_weight='normal', font_size=tfont, linewidths=1, edge_color='grey', node_color='blue')

		# draw network and summary statistics
		#density = nx.density(G)
		#diameter = nx.diameter(G)
		#triadic_closure = nx.transitivity(G)

		# print out node degrees, centrality, HITS statistics for identifying hubs
		degree_dict = dict(G.degree(G.nodes()))		
		sort_centrality = nx.degree_centrality(G)		
		sort_degree = defaultdict(int)
		node_names = []
		hub_hits = defaultdict(float)
		hub_authorities = defaultdict(float)
		nx.set_node_attributes(G, degree_dict, 'degree')

		hubs, authorities = nx.hits(G, max_iter = 50, normalized = True) 
		for tnode in G.nodes:
			sort_degree[tnode] = G.nodes[tnode]['degree']
			hub_hits[tnode] = hubs[tnode]
			hub_authorities[tnode] = authorities[tnode]
		
		hub_list = []
		for tnode in sorted(sort_centrality, key=sort_centrality.get, reverse=True):
			#hubval = hub_authorities[tnode]
			hubval = sort_centrality[tnode]
			hub_list.append(hubval)
			node_names.append(tnode)
		
		hub_list = np.array(hub_list)
		#print("Hub list: " + str(hub_list))

		#nx.draw_spectral(G, with_labels=True, font_weight='normal', font_size=10, linewidths=1, edge_color='grey', node_color='blue') 
		plt.savefig(network_file, format="JPG")
		
		# plot density of hub_hits
		plt.clf()
		plt.close()

		'''
		hub_list2 = hub_list.reshape(-1,1)
		g = mixture.GaussianMixture(n_components=2, covariance_type='full')
		g.fit(hub_list2)
		weights = g.weights_
		means = g.means_
		covars = g.covariances_
		standard_deviations = g.covariances_**0.5
		samples = []
		samples0 = []
		samples1 = []		

		# create two distributions for plotting
		mu1 = means[0][0]
		mu2 = means[1][0]
		std1 = standard_deviations[0][0][0]
		std2 = standard_deviations[1][0][0]
		mc = weights
		d1 = stats.norm(mu1, std1)
		d2 = stats.norm(mu2, std2)
		x = np.linspace(-0.005, max(hub_list) + 0.01, 501)
		c1 = d1.pdf(x) * mc[0]
		c2 = d2.pdf(x) * mc[1]
		
		print("Means: " + str(means[0][0]) + " STD: " + str(standard_deviations[0][0][0]) + " Weights: " + str(weights))
		
		# We're going to get all the pollutants with hub values above the mean of the mixture with the higher mean
		# and add 2 STD
		linepos = mu2 + std2
		hub_cutoff = mu2 + std2
		if mu1 > mu2:
			linepos = mu1 - std1
			hub_cutoff = mu1 - std1

		sns.set_style('whitegrid')
		density_plot = sns.kdeplot(hub_list)
		fig = density_plot.get_figure()
		plt.axvline(linepos, color='red')
		plt.axvline(means[0][0], color='green', linestyle='--')
		plt.axvline(means[1][0], color='green', linestyle='--')
		plt.plot(x, c1, label='Component 1', color='green', linestyle='--')
		plt.plot(x, c2, label='Component 2', color='green', linestyle='--')
		fig.savefig(density_file, format="JPG", dpi=300) 
		'''
		
		data_list = np.array(hub_list)		
		cluster_data = ckmeans(data_list, k=2).cluster
		cluster_list = defaultdict(int)
		for x in range(0, len(cluster_data)):
			cluster_list[node_names[x]] = cluster_data[x]

		# write out stats
		f = open(stats_file, "w")
		
		#f.write(str(nx.info(G)) + "\n")
		#f.write("Density: " + str(density) + "\n")
		#f.write("Diameter: " + str(diameter) + "\n")
		#f.write("Triadic Closure: " + str(triadic_closure) + "\n")

		final_hub_list = []
		trank = 1
		for tnode in sorted(sort_centrality, key=sort_centrality.get, reverse=True):
			hubval = sort_centrality[tnode]
			ishub = "No"
			if cluster_list[tnode] == 1:
				ishub = "Hub"
			#if hubval >= hub_cutoff:
			#	ishub = "Hub"
			#	final_hub_list.append(tnode)
			tline = str(trank) + "\t" + ishub + "\t" + tnode + "\t" + str(sort_degree[tnode]) + "\t" + str(sort_centrality[tnode]) + "\t" + str(hubval)  + "\t" + str(hub_authorities[tnode])
			f.write(tline + "\n")
			if ishub == "Hub":
				print(tline)
			trank += 1
		
		f.close()

		# Hubs should be connected to each other and have a high Jaccard Correlation Coefficient
		# If they do not have a high Jaccard, drop it
		#final_list = final_hub_list.copy()
		#for tnode1 in range(0, len(final_hub_list)):
		#	for tnode2 in range(tnode1 + 1, len(final_hub_list)):
		#		id1 = final_hub_list[tnode1]
		#		id2 = final_hub_list[tnode2]
		#		val1 = 0
		#		val2 = 0
		#		if id1 in disease_network[baseid]:
		#			if id2 in disease_network[baseid][id1]:
		#				val1 = disease_network[baseid][id1][id2]
		#		if id2 in disease_network[baseid]:
		#			if id1 in disease_network[baseid][id2]:
		#				val2 = disease_network[baseid][id2][id1]
		#		
		#		if (val1 < jaccard_cutoff) and (val2 < jaccard_cutoff):
		#			print("Delete: " + baseid + "\t" + id1 + "\t" + id2 + "\t" + str(val1) + "\t" + str(val2))
		#			if id1 in final_list:
		#				final_list.remove(id1)
		#				print("Deleting: " + id1)
		#			if id2 in final_list:
		#				final_list.remove(id2)
		#				print("Deleting: " + id2)

		#print("Initial hubs: " + str(len(final_hub_list)) + ":" + str(final_hub_list))
		#print("Final hubs: " + str(len(final_list)) + ":" + str(final_list))

def MakeNetworkFig():

	# airtox_merge_density_lancet.canonical.Arthritis.0.7.binarymap.tsv.png
	
	# convert to pandas dataframe
	tHTML = "<html>\n<table>\n"
	base_disease = ["Arthritis", "Asthma", "Cancer", "COPD", "Coronary Heart Disease", "Depression", "Diabetes", "Hypertension", "Obesity", "Renal Disease", "Stroke", "stroke_data"]
	for baseid in base_disease:

		# set files
		tlabel = baseid
		opath = "networks/"
		network_file = opath + "airtox_merge_network_lancet.canonical." + baseid + ".0.7.binarymap.tsv.jpg"
		stats_file = opath + "airtox_merge_stats_lancet.canonical." + baseid + ".0.7.binarymap.tsv.txt"
		
		if (baseid == "stroke_belt"):
			network_file = opath + "airtox_merge_network_lancet.canonical." + baseid + ".binarymap.tsv.jpg"
			stats_file = opath + "airtox_merge_stats_lancet.canonical." + baseid + ".binarymap.tsv.txt"
			tlabel = "Stroke Belt"

		if (baseid == "stroke_data"):
			network_file = opath + "airtox_merge_network_lancet.canonical." + baseid + "..0.7.binarymap.tsv.jpg"
			stats_file = opath + "airtox_merge_stats_lancet.canonical." + baseid + "..0.7.binarymap.tsv.txt"
			tlabel = "Stroke Data"
		
		# airtox_merge_stats_lancet.canonical.Arthritis.0.7.binarymap.tsv.txt
		tcnt = 0.
		topcnt = 20
		# ACETALDEHYDE	176
		ttable = []
		chem_list = []
		print("Loading Network File: " + stats_file)
		with open(stats_file, "r") as infile:
			for line in infile:
				line = line.strip()
				ldata = line.split("\t")
				tid = ldata[2]
				thub = ldata[1]
				if tid.find('(') > -1:
					cdata = tid.split('(')
					tid = cdata[0].strip()
				if thub == "Hub":
					tid = tid + '*'
				if tcnt < topcnt:
					chem_list.append(tid)
					trow = [tid, ldata[3]]
					ttable.append(trow)
				tcnt += 1
		infile.close()
		#print("Network nodes found: " + str(chem_list))

		bar_data = pd.DataFrame(ttable, columns=['columns', 'importances'])
		#epa_data.set_index("fips", inplace=True)
		bar_data[['importances']] = bar_data[['importances']].apply(pd.to_numeric)
		
		# reverse pandas dataframe order
		bar_data = bar_data.iloc[::-1]
		
		# bar plot node degree
		bar_file = stats_file + ".bar.jpg"
		TopValsBarChart(bar_data, bar_file, "Degree (# Connections)")
		
		# html content
		tHTML += "<tr>\n" 
		tHTML += "	<td style=\"text-align: center\"><span style=\"font-family: arial; font-size: 24px; text-align: right;\">" + tlabel + "</span></td>\n"
		tHTML += "	<td><img src=\"" + network_file + "\" style=\"width: 750px;\"></td>\n"
		tHTML += "	<td><img src=\"" + bar_file + "\" style=\"width: 400px;\"></td>\n"
		tHTML += "</tr>\n"
		
	tHTML += "</html>\n"
	html_file = open("lancet_merge_network_supp.html", "w")
	html_file.write(tHTML)
	html_file.close()

def MakeFigure5Assembly():

	# load the optimized file
	tfile = "lancet_merge_assembled_jaccards.tsv"

	# Diabetes_4_assembled.jpg.cluster.tsv    4       2       ACETALDEHYDE!!BENZOAPYRENE      0.4424588992137241      2.345119333952249e-127
	datatable = defaultdict(lambda: defaultdict(str))
	sort_jvals = defaultdict(float)
	with open(tfile, "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			cdata = ldata[0].split('_')
			tdisease = cdata[0]
			tid = ldata[0]
			jval = ldata[4]
			pval = ldata[5]
			if (tid.find("STROKE_BELT") == -1):
				sort_jvals[tid] = float(jval)
				datatable[tid]["jval"] = float(jval)
				datatable[tid]["pollutants"] = ldata[3]
				datatable[tid]["pval"] = ldata[5]
	infile.close()
	
	tHTML = "<html><div style=\"font-family: arial;\"><table>\n"

	tindex = 1
	tHTML += "<tr><td><center>Disease</center></td>\n"
	for tid in sorted(sort_jvals, key=sort_jvals.get, reverse=True):
		ddata = tid.split('_')
		dlabel = ddata[0]
		plabel = dlabel
		if tid.find('STROKE_MORT') > -1:
			dlabel = 'STROKE_MORT'
			plabel = "Stroke Mortality"
		jval = round(float(datatable[tid]["jval"]), 4)
		pval = datatable[tid]["pval"]
		pval_data = pval.split('e-')
		pstr = pval_data[0][0:4] + 'x10<sup>-' + pval_data[1][0:] + '</sup>'
		tHTML += "<td>" + "<center><br><h3>" + str(tindex) + '.' + plabel + "</h3>\n<i>J</i> = " + str(jval) + "<br><i>p</i> = " + pstr + "</center></td>"
		tindex += 1
	tHTML += "</tr>\n"
	
	tHTML += "<tr><td><center>Reference Map</center></td>\n"
	for tid in sorted(sort_jvals, key=sort_jvals.get, reverse=True):
		ddata = tid.split('_')
		dlabel = ddata[0]
		if tid.find('STROKE_MORT') > -1:
			dlabel = 'stroke_data.'
		map_pic = 'lancet.canonical.' + dlabel + '.0.7.binarymap.png'
		tHTML += "<td><img src=\"" + map_pic + "\" style=\"width: 200px;\"></td>"
	tHTML += "</tr>\n"

	tHTML += "<tr><td><center>Assembled Map</center></td>\n"
	for tid in sorted(sort_jvals, key=sort_jvals.get, reverse=True):
		ddata = tid.split('_')
		dlabel = ddata[0]
		if tid.find('STROKE_MORT') > -1:
			dlabel = 'STROKE_MORT'
		map_pic = 'lancet_final_' + dlabel + '_assembled.map.png'
		tHTML += "<td><img src=\"" + map_pic + "\" style=\"width: 200px;\"></td>"
	tHTML += "</tr>\n"

	tHTML += "<tr><td><center>Pollutants (Hubs)</center></td>\n"
	for tid in sorted(sort_jvals, key=sort_jvals.get, reverse=True):
		polldata = datatable[tid]["pollutants"].split('!!')
		poll1 = polldata[0].title()
		poll2 = polldata[1].title()

		if poll1 == "Benzoapyrene":
			poll1 = "Benzo(a)pyrene"
		if poll2 == "Benzoapyrene":
			poll2 = "Benzo(a)pyrene"
		
		tHTML += "<td><center>" + poll1 + ', <br>' + poll2 + "</center></td>"
	tHTML += "</tr>\n"
		
	tHTML += "</table></div></html>\n"
	
	f = open("lancet_merge_fig4_assembled_maps.html", "w")
	f.write(tHTML)
	f.close()
	

def OptimizeAssemblyByJaccard(chemical_data, json_data_county):

	top_jaccard_pollutants = GetTopPollutantsByJaccard()

	# create assembled maps
	countycnt = 3141
	stroke_cutoffs = [0.7]
	xpath = "d:\\apeer\\networks"
	disease_list = ["COPD", "Hypertension", "Asthma", "Diabetes", "Depression", "Arthritis", "Cancer", "Coronary Heart Disease", "Renal Disease", "Obesity", "Stroke"]
	#ndisease_list = disease_list.copy() + ["STROKE_MORT", "STROKE_BELT"]
	ndisease_list = disease_list.copy() + ["STROKE_MORT"]
	param_data = ""
	
	# Now Select hubs 
	tcutoff = 0.7
	#jfile = "lancet_merge_figure3.tsv"
	jfile = "lancet_merge_0.7_figure3.tsv"
	pval_cutoff = GetPValCutoff(jfile)
	f = open("lancet_merge_assembled_jaccards.tsv", "w")
	for tdisease in top_jaccard_pollutants:

		# Load Hub Data - airtox_merge_stats_lancet.canonical.Cancer.0.7.binarymap.tsv.txt
		hub_table = defaultdict(str)
		hub_file = 'd:\\apeer\\networks\\airtox_merge_stats_lancet.canonical.' + tdisease + '.0.7.binarymap.tsv.txt'
		if (tdisease == "STROKE_MORT"):
			hub_file = 'd:\\apeer\\networks\\airtox_merge_stats_lancet.canonical.stroke_data..0.7.binarymap.tsv.txt'
		if (tdisease == "STROKE_BELT"):
			hub_file = 'd:\\apeer\\networks\\airtox_merge_stats_lancet.canonical.stroke_belt.binarymap.tsv.txt'

		top_list = []		
		with open(hub_file, "r") as infile2:
			for line in infile2:
				line = line.strip()
				ldata = line.split("\t")
				tpollutant = ldata[2].lower().strip()
				if ldata[1] == "Hub":
					#hub_table[tpollutant] = ldata[1]
					top_list.append(ldata[2])
		infile2.close()

		# get top pairwise value
		max_jval = 0
		max_jdesc = ""
		with open(jfile, "r") as infile3:
			for line in infile3:
				line = line.strip()
				ldata = line.split("\t")
				nid = ldata[0]
				fdata = nid.split('.')
				nid = fdata[2]
				id1 = ldata[1]
				id2 = ldata[2]
				tclust = float(ldata[3])
				jval = float(ldata[4])
				pval = float(ldata[5])
				if (nid.lower().strip() == tdisease.lower().strip()) or ((tdisease == "STROKE_MORT") and (nid == "stroke_data")):
					if (jval > max_jval):
						if (tdisease != "Asthma") and ((id1 in top_list) and (id2 in top_list)):
							max_jdesc = line
							max_jval = jval
						if (tdisease == "Asthma") and ((id1 in top_list) or (id2 in top_list)):
							max_jdesc = line
							max_jval = jval

		infile3.close()
		
		# get top 2 indicators
		fdata = max_jdesc.split("\t")
		if len(fdata) > 4:
			tjaccard = fdata[4]
			tpval = fdata[5]
			nclust = int(fdata[3])
			chemlist = [fdata[1], fdata[2]]
			final_list = fdata[1] + '!!' + fdata[2]
			tdata = chemical_data[chemlist]
			tfile = tdisease + "_" + str(nclust) + "_assembled.jpg"
			ofile = tfile + ".cluster.tsv"
			#print("Top Match: " + tdisease + ": " + max_jdesc)
			#print("Making cluster file: " + ofile)
			#ShowTractClustersOnMap(tdata, nclust, tfile, "county", json_data_county, showmap=False, forceBinary=False)
			#tjaccard, tpval = CalcJaccard(rawdata, refbelt, countycnt)
			oline = ofile + "\t" + str(nclust) + "\t" + str(len(chemlist)) + "\t" + final_list + "\t" + str(tjaccard) + "\t" + str(tpval)
			#print(tdisease + "\t" + str(nclust) + "\t" + final_list + "\t" + str(tjaccard) + "\t" + str(tpval))
			f.write(oline + "\n")
						
		'''					
		nclust = 4
		tstart = 0
		for x in range(2, len(top_list)):
			tend = x
			chemlist = top_list.copy()
			tend = tstart + x
			chemlist = top_list[tstart:tend].copy()
			if len(chemlist) > 1:
				tdata = chemical_data[chemlist]
				#print(tdisease + ": Top Jaccard-scoring Hubs: " + str(chemlist))
				tfile = tdisease + "_" + str(x) + "_assembled.jpg"

				# Load Reference map data
				rfile = "lancet.canonical." + tdisease + "." + str(tcutoff) + ".binarymap.tsv"
				if tdisease == "STROKE_MORT":
					rfile = "lancet.canonical.stroke_data." + "." + str(tcutoff) + ".binarymap.tsv"
				if tdisease == "STROKE_BELT":
					rfile = "lancet.canonical.stroke_belt.binarymap.tsv"

				# Transform clustered data
				rawdata2 = LoadClusterData(rfile)
				refbelt = defaultdict(int)
				for tid in rawdata2:
					if rawdata2[tid] == "1":
						refbelt[tid] = 1
				
				# cluster 2-4: Hypertension: 0.45, Stroke Mortality: 0.45, Stroke: 0.41, Renal Disease: 0.41
				# cluster 2-5: Hypertension: 0.51, Renal Disease: 0.47, Stroke Mortality: 0.46, CAD: 0.43, Stroke: 0.44
				
				for nclust in range(2, 7):
				
					# create clusters
					ShowTractClustersOnMap(tdata, nclust, tfile, "county", json_data_county, showmap=False, forceBinary=False)

					# Load Clustered Map
					ofile = tfile + ".cluster.tsv"
					rawdata = LoadClusterData(ofile)
					#clustdata = defaultdict(int)
					#for tid in rawdata:
					#	if rawdata[tid] == "1":
					#		clustdata[tid] = 1
								
					# Calculate Jaccard overlap
					#tjaccard, tpval = CalcJaccard(clustdata, refbelt, countycnt)
					final_list = ""
					for titem in chemlist:
						final_list += titem + '!!'
					final_list = final_list[:-2]
					tjaccard, tpval = CalcJaccard(rawdata, refbelt, countycnt)
					oline = ofile + "\t" + str(nclust) + "\t" + str(len(chemlist)) + "\t" + final_list + "\t" + str(tjaccard) + "\t" + str(tpval)
					#print(tdisease + "\t" + str(nclust) + "\t" + final_list + "\t" + str(tjaccard) + "\t" + str(tpval))
					f.write(oline + "\n")
		'''
			
	f.close()	

def GetBestAssembledMaps(max_clust, chemical_data, json_data_county):

	max_jaccard = defaultdict(lambda: defaultdict(str))
	with open("lancet_merge_assembled_jaccards.tsv", "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			ddata = ldata[0].split('_')
			tdisease = ddata[0]
			if tdisease == "STROKE":
				tdisease = ddata[0] + '_' + ddata[1]
			# round this to 2 digits
			tjaccard = round(float(ldata[4]), 4)
			tpval = ldata[5]
			tclust = int(ldata[1])
			if (tclust < max_clust):
				if tdisease not in max_jaccard:
					max_jaccard[tdisease]["jaccard"] = str(tjaccard)
					max_jaccard[tdisease]["cluster"] = ldata[1]
					max_jaccard[tdisease]["pollutants"] = ldata[3]
					max_jaccard[tdisease]["pvals"] = ldata[5]
					
				if round(float(max_jaccard[tdisease]["jaccard"]), 4) < tjaccard:
					max_jaccard[tdisease]["jaccard"] = str(tjaccard)
					max_jaccard[tdisease]["cluster"] = ldata[1]
					max_jaccard[tdisease]["pollutants"] = ldata[3]
					max_jaccard[tdisease]["pvals"] = ldata[5]

	infile.close()
	
	tcutoff = 0.7
	f2 = open("best_assembled_maps.tsv", "w")
	print("Best Matches**************")
	for tdisease in max_jaccard:
	
		# Load Reference map data
		rfile = "lancet.canonical." + tdisease + "." + str(tcutoff) + ".binarymap.tsv"
		if tdisease == "STROKE_MORT":
			rfile = "lancet.canonical.stroke_data." + "." + str(tcutoff) + ".binarymap.tsv"
		if tdisease == "STROKE_BELT":
			rfile = "lancet.canonical.stroke_belt.binarymap.tsv"

		# Transform clustered data
		rawdata2 = LoadClusterData(rfile)
		refbelt = defaultdict(int)
		for tid in rawdata2:
			if rawdata2[tid] == "1":
				refbelt[tid] = 1
	
		# get map
		assemble_map = "lancet_final_" + tdisease + "_assembled"
		numclust = int(max_jaccard[tdisease]["cluster"])
		pollution_list = max_jaccard[tdisease]["pollutants"].split('!!')
		selected_datatable = chemical_data[pollution_list]
		BestMatchClusterOnMap(selected_datatable, numclust, assemble_map, "county", json_data_county, refbelt)
		tline = tdisease + "\t" + assemble_map + ".map.png" + "\t" + max_jaccard[tdisease]["jaccard"] + "\t" + max_jaccard[tdisease]["cluster"] + "\t" + max_jaccard[tdisease]["pollutants"] + "\t" + max_jaccard[tdisease]["pvals"]

		f2.write(tline + "\n")
		
	f2.close()

def GetTopPollutionPair(mainfile, tlabel):

	base_disease = defaultdict(int)
	pollution_list = defaultdict(float)
	disease_network = defaultdict(lambda: defaultdict(str))
	#tfile = "lancet_figure2.tsv"
	
	# get p-value cutoff
	pval_cutoff_cnt = 0
	tfile = mainfile
	with open(tfile, "r", encoding="utf8") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			if len(ldata) > 4:
				pval_cutoff_cnt += 1
	infile.close()
	
	pval_threshold = 0.01 / pval_cutoff_cnt
	
	max_jaccard = defaultdict(str)
	max_jaccard_pair = defaultdict(lambda: defaultdict(float))
	tfile = mainfile
	with open(tfile, "r") as infile:

		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			baseid = ldata[0]
			poll1 = ldata[1]
			poll2 = ldata[2]
			clust = ldata[3]
			jval = float(ldata[4])
			pval = float(ldata[5])
			
			if (pval < pval_threshold):			

				if baseid not in max_jaccard:
					max_jaccard[baseid] = jval

				if max_jaccard[baseid] == jval:
					max_jaccard_pair[baseid][poll1 + "\t" + poll2 + "\t" + str(clust)] = jval

				if max_jaccard[baseid] < jval:
					max_jaccard[baseid] = jval
					del max_jaccard_pair[baseid]
					max_jaccard_pair[baseid][poll1 + "\t" + poll2 + "\t" + str(clust)] = jval
							
	infile.close()
				
	return max_jaccard_pair


def GetTopPollutants(mainfile, tlabel, num_top):

	base_disease = defaultdict(int)
	pollution_list = defaultdict(float)
	disease_network = defaultdict(lambda: defaultdict(str))
	top_pair = defaultdict(float)
	opath = "d:\\apeer\\networks\\"
	#tfile = "lancet_figure2.tsv"
		
	tfile = mainfile
	with open(tfile, "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			baseid = ldata[0]
			poll1 = ldata[1]
			poll2 = ldata[2]
			clust = ldata[3]
			jval = float(ldata[4])
			pval = float(ldata[5])
			
			# build matrix and indices
			base_disease[baseid] = 1
			pollution_list[poll1] = 1
			pollution_list[poll2] = 1
			
	infile.close()
	
	# convert to pandas dataframe
	pollution_score = defaultdict(lambda: defaultdict(float))
	disease_list = defaultdict(int)
	for baseid in base_disease:
		
		# set files
		stats_file = opath + tlabel + "_" + "stats_" + baseid + ".txt"
		file_data = stats_file.split('.')
		diseaseid = file_data[2].lower()
		disease_list[diseaseid] = 1
		#tcnt = 0
		#print("Processing " + stats_file)
		with open(stats_file, "r") as infile:
			for line in infile:
				line = line.strip()
				ldata = line.split("\t")
				tid = ldata[0]
				tscore = ldata[1]
				#if tcnt < 10:
				pollution_score[diseaseid][tid] += float(tscore)
				#tcnt += 1
		infile.close()
	
	for diseaseid in disease_list:
		tcnt = 0
		for tid in sorted(pollution_score[diseaseid], key=pollution_score[diseaseid].get, reverse=True):
			if tcnt < num_top:
				#print(str(tcnt) + "\t" + diseaseid + "\t" + tid + "\t" + str(pollution_score[diseaseid][tid]))
				disease_network[diseaseid][tid] = str(pollution_score[diseaseid][tid])
			tcnt += 1
			
	return disease_network

def GetHubList(tdisease, tcutoff):

	# filename = airtox_merge_stats_lancet.canonical.Arthritis.0.7.binarymap.tsv.txt
	tpath = "d:\\apeer\\networks"
	tfile = tpath + "\\" + "airtox_merge_stats_lancet.canonical." + tdisease + "." + str(tcutoff) + ".binarymap.tsv.txt"
	if tdisease == "STROKE_MORT":
		tfile = tpath + "\\" + "airtox_merge_stats_lancet.canonical.stroke_data.." + str(tcutoff) + ".binarymap.tsv.txt"
	if tdisease == "STROKE_BELT":
		tfile = tpath + "\\" + "airtox_merge_stats_lancet.canonical.stroke_belt.binarymap.tsv.txt"
	pollution_list = []
	with open(tfile, "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			if ldata[1] == "Hub":
				pollution_list.append(ldata[2])
	infile.close()
	
	return pollution_list
	

def SummarizeNetworkHubs():

	# Getting Pollution List
	topcnt = 10
	
	f = open("top_pollutants.tsv", "w")

	ndisease_list = disease_columns.copy()
	ndisease_list = ndisease_list + ["LIFE_EXPECT", "STROKE_MORT"]
	
	for tdisease in ndisease_list:

		chem_list = []
		f.write("Rank\tDisease\tPollutant\tConnectivity\n")
		#rdata = tfile.split('.')
		#mainfield = rdata[0].replace('_roc', '')
		mainfield = tdisease
		mainfield_match = mainfield.lower()
		if mainfield == "STROKE_MORT":
			mainfield_match = "stroke_data"	

		rauc = "0.6"
		if mainfield == "LIFE_EXPECT":
			rauc = "0.3"
		#rauc = rdata[len(rdata) - 3] + '.' + rdata[len(rdata) - 2]
		#if (tfile.find("network") > -1):
		#	rauc = rdata[len(rdata) - 4] + '.' + rdata[len(rdata) - 3]

		# Get top network nodes
		network_file = "d:\\apeer\\networks\\airtox_stats_lancet.canonical." + mainfield_match + "." + str(rauc) + '.binarymap.tsv.txt'
		if (mainfield_match == "stroke_data") or (mainfield_match == "stroke_mort"):
			network_file = "d:\\apeer\\networks\\airtox_stats_lancet.canonical.stroke_data.." + str(rauc) + '.binarymap.tsv.txt'
		if mainfield_match == "stroke_belt":
			network_file = "d:\\apeer\\networks\\airtox_stats_lancet.canonical.stroke_belt.binarymap.tsv.txt"

		tcnt = 0
		# ACETALDEHYDE	176
		ttable = []
		#print("Loading Network File: " + network_file)
		with open(network_file, "r") as infile:
			for line in infile:
				line = line.strip()
				ldata = line.split("\t")
				tid = ldata[0]
				if tcnt < topcnt:
					f.write(str(tcnt + 1) + "\t" + mainfield + "\t" + ldata[0] + "\t" + ldata[1] + "\n")
					#chem_list.append(tid)
					#trow = [ldata[0], ldata[1]]
					#ttable.append(trow)
				tcnt += 1
		infile.close()
	
	f.close()
	

def GetTopNetworkPollutants(tfile, topcnt):

	# Getting Pollution List
	chem_list = []
	rdata = tfile.split('.')
	mainfield = rdata[0].replace('_roc', '')
	mainfield_match = mainfield.lower()
	if mainfield == "STROKE_MORT":
		mainfield_match = "stroke_data"	

	tprefix = "airtox_stats_lancet.canonical"
	bpath = "d:\\apeer\\networks"
	rauc = rdata[len(rdata) - 3] + '.' + rdata[len(rdata) - 2]
	if (tfile.find("network") > -1):
		rauc = rdata[len(rdata) - 4] + '.' + rdata[len(rdata) - 3]

	# Check file for pollution - pollution_stats_lancet.canonical.
	if tfile.find('pollution_stats_lancet.canonical') > -1:

		mainfield = rdata[2]
		mainfield_match = mainfield.lower()
		if mainfield == "STROKE_MORT":
			mainfield_match = "stroke_data"	

		rauc = rdata[len(rdata) - 5] + '.' + rdata[len(rdata) - 4]
		tprefix = "pollution_stats_lancet.canonical"

	# Get top network nodes
	network_file = bpath + "\\" + tprefix + "." + mainfield_match + "." + str(rauc) + '.binarymap.tsv.txt'
	if (mainfield_match == "stroke_data") or (mainfield_match == "stroke_mort"):
		network_file = bpath + "\\" + tprefix + ".stroke_data.." + str(rauc) + '.binarymap.tsv.txt'
	if mainfield_match == "stroke_belt":
		network_file = bpath + "\\" + tprefix + ".stroke_belt.binarymap.tsv.txt"
	if mainfield_match == "life_expect":
		network_file = bpath + "\\" + tprefix + ".life_expect.0.3.binarymap.tsv.txt"
	
	# airtox_merge_stats_lancet.canonical.Arthritis.0.7.binarymap.tsv.txt
	tcnt = 0
	# ACETALDEHYDE	176
	ttable = []
	print("Loading Network File: " + network_file)
	with open(network_file, "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			tid = ldata[0]
			if tcnt < topcnt:
				chem_list.append(tid)
				trow = [ldata[0], ldata[1]]
				ttable.append(trow)
			tcnt += 1
	infile.close()
	#print("Network nodes found: " + str(chem_list))

	bar_data = pd.DataFrame(ttable, columns=['columns', 'importances'])
	#epa_data.set_index("fips", inplace=True)
	bar_data[['importances']] = bar_data[['importances']].apply(pd.to_numeric)
	
	# reverse pandas dataframe order
	bar_data = bar_data.iloc[::-1]
	
	return chem_list, bar_data

def TopValsBarChart(top_sorted, iFile, xlabel):

	plt.close('all')
	plt.clf()
	plt.rcParams["figure.figsize"] = (4.5,5.5)
	plt.rcParams.update({'font.size': 14})
	#sorted_idx = model.feature_importances_.argsort()
	
	# Format Text
	#print(top_sorted)
	for x in range(0, len(top_sorted.index)):
		tindex = top_sorted.index[x]
		oval = top_sorted.loc[tindex, 'columns']
		nval = oval.title()
		if nval.find('(') > -1:
			fdata = nval.split('(')
			nval = fdata[0]
		#print(oval + "\t" + nval)
		top_sorted.loc[tindex, 'columns'] = nval

	#plt.barh(feature_names[sorted_idx], model.feature_importances_[sorted_idx])
	plt.close('all')
	fig, ax = plt.subplots() 
	fig.patch.set_facecolor('white')
	ax.set_facecolor('white')
	ax.barh(top_sorted['columns'], top_sorted['importances'])

	plt.subplots_adjust(left=0.5, right=0.95, top=0.9, bottom=0.1)
	ax.set_xlabel(xlabel, fontsize=14)
	#ax.set_ylabel("", fontsize=14)
	ax.tick_params(axis='x', labelsize=8)
	ax.tick_params(axis='y', labelsize=9)
	ax.figure.savefig(iFile, dpi=1200)

def variance_threshold_selector(data, threshold=0.5):
	# https://stackoverflow.com/a/39813304/1956309
	selector = VarianceThreshold(threshold)
	selector.fit(data)
	
	#print("Variance threshold")
	odata = data[data.columns[selector.get_support(indices=True)]]
	
	#print(odata)
	
	return odata

def CalcRegression(X, y, subX, suby):

	# grid parameters
	n_folds = 3
	C_values = [0.001, 0.01, 0.05, 0.1, 1., 100.]
	l1_ratio_vals = [0.15, 0.25, 0.5, 0.75]

	univardata = defaultdict(lambda: defaultdict(str))
	
	# replace NaN values
	X = X.fillna(0)
	y = y.fillna(0)
	X = X.replace(np.nan, 0)
	y = y.replace(np.nan, 0)

	subX = subX.fillna(0)
	suby = suby.fillna(0)
	subX = subX.replace(np.nan, 0)
	suby = suby.replace(np.nan, 0)
	
	# remove if all zeros
	#zero_cols = X.columns[(X == 0).all()]
	#X.drop(labels=zero_cols, axis=1, inplace=True)
	#zero_cols = subX.columns[(subX == 0).all()]
	#subX.drop(labels=zero_cols, axis=1, inplace=True)

	X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.5, random_state=7, shuffle=True)
	subX_train,subX_test,suby_train,suby_test = train_test_split(subX, suby, test_size=0.5, random_state=7, shuffle=True)

	log_regression = LogisticRegressionCV(Cs=C_values, cv=n_folds, solver='saga', penalty='elasticnet', refit=True, scoring='roc_auc', l1_ratios=l1_ratio_vals, random_state=0)
	#log_regression = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=0.0001, l1_ratio=0.15)
	#log_regression = LogisticRegression(random_state=1, solver='liblinear')
	#log_regression = LogisticRegression()
	#log_regression = Ridge(alpha=10)
	
	# gets a Hessian error
	log_regression.fit(X_train, y_train)
	#y_pred_proba = log_regression.predict_log_proba(X_test)[::,1]
	#y_pred_proba = log_regression.predict_proba(subX_test)[::,1]
	y_pred_proba = log_regression.predict_proba(X_test)[::,1]
	auc = metrics.roc_auc_score(y_test, y_pred_proba)
	fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
	
	coef_dict = {}
	coef_data = log_regression.coef_[0]
	#print("Coefficients: " + str(coef_data))
	for x in range(0, len(coef_data)):
		#print("coefficient: " + str(coef_data[x]))
		coef_dict[X_train.columns[x]] = coef_data[x]
		univardata[X_train.columns[x]]["beta"] = str(coef_data[x])
	
	'''
	# print out coefficients
	logit_model=sm.Logit(y_train,X_train)
	est2=logit_model.fit()
	#print(est2.summary())
		
	for i, v in est2.params.items():
		univardata[i]["beta"] = str(v)
	for i, v in est2.pvalues.items():
		univardata[i]["p"] = str(v)
	for i, v in est2.conf_int()[0].items():
		univardata[i]["conf_low"] = str(v)
	for i, v in est2.conf_int()[1].items():
		univardata[i]["conf_hi"] = str(v)
		univardata[i]["pseudo_r2"] = str(est2.prsquared)
	'''

	#return fpr, tpr, auc, univardata
	return fpr, tpr, auc, univardata, y_test, y_pred_proba

def RunUnivariateRegression(tdata, tfield, tfile):
    
    # regression results
    univardata = defaultdict(lambda: defaultdict(str))
    multivardata = defaultdict(lambda: defaultdict(str))
    
    # get y data
    ydata = tdata[[tfield]]
    
    # get rid of y data and leave x data
    tcols = tdata.columns
    newdata = tdata
    if "COVID-19 Cases" in tcols:
        newdata = newdata.drop(['COVID-19 Cases'], axis=1)
    if "COVID-19 Deaths" in tcols:
        newdata = newdata.drop(['COVID-19 Deaths'], axis=1)
    newdata = newdata.drop([tfield], axis=1)

    xvars = newdata.columns

    # create dataset keeping variables with highest pca loadings
    # Healthcare Access, Arthritis, Hypertension, Smoking, COPD, Diabetes
    for tcol in xvars:
        
        current_data = newdata[tcol]
        
        #trainX, testX, trainy, testy = train_test_split(current_data, ydata, test_size=0.5, random_state=2)

        # run OLS regression
        #X2 = sm.add_constant(trainX)
        #est = sm.OLS(trainy, X2)
        X2 = sm.add_constant(current_data)
        #missdata = X2.isna().sum()
        #if missdata > 0:
        #    print("RunUnivariateRegression: missing data for " + tcol)
        est = sm.OLS(ydata, X2)
        est2 = est.fit()
        
        # print out values
        univardata[tcol]["r2"] = str(est2.rsquared)
        univardata[tcol]["r2_adj"] = str(est2.rsquared_adj)
        univardata[tcol]["const"] = str(est2.params["const"])
        univardata[tcol]["beta"] = str(est2.params[tcol])
        univardata[tcol]["conf_lower"] = str(est2.conf_int()[0])
        univardata[tcol]["conf_higher"] = str(est2.conf_int()[1])
        
        #print("Univariate regression: " + tcol + " beta: " + str(univardata[tcol]["beta"]))
        
        # get p-values
        unipvals = defaultdict(lambda: defaultdict(str))
        for i, row in est2.pvalues.iteritems():
            if i != "const":
                univardata[i]["pval"] = str(row)
                
        del X2, est, est2        
                
    #for tid in univardata:
    #    print(tid + "\t" + univardata[tid]["r2"] + "\t" + univardata[tid]["r2_adj"] + "\t" + univardata[tid]["const"] + "\t" + univardata[tid]["beta"] + "\t" + univardata[tid]["pval"])
        
    # run multiple linear regression
    trainX, testX, trainy, testy = train_test_split(newdata, ydata, test_size=0.5, random_state=2, shuffle=True)

    # run OLS regression
    #X2 = sm.add_constant(trainX)
    #est = sm.OLS(trainy, X2)
    X2 = sm.add_constant(newdata)
    est = sm.OLS(ydata, X2)
    est2 = est.fit()
    
    # print out properties
    #print(dir(est2))
    
    # get values
    r2 = est2.rsquared
    r2_adj = est2.rsquared_adj
    pvals = est2.pvalues
    betavals = est2.params
    coeff = est2.params
    conf_lower = est2.conf_int()[0]
    conf_higher = est2.conf_int()[1]
    #print("Multivariate regression")
    
    varnames = []
    for i, row in coeff.iteritems():
        multivardata[i]["beta"] = row
    for i, row in pvals.iteritems():
        multivardata[i]["pval"] = row
        #print("Row: " + i + "\t" + " Value: " + str(row))
    #for i, row in r2_adj.iteritems():
    #    multivardata[i]["r2_adj"] = row

    # Run ElasticNet Regression
    # https://machinelearningmastery.com/elastic-net-regression-in-python/
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define model
    ratios = arange(0, 1, 0.01)
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
    model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)

    # fit model
    enet_model = model.fit(trainX, trainy)

    # check prediction
    y_pred = enet_model.predict(testX)
    terror = np.sqrt(mean_squared_error(testy, y_pred))
    r2 = r2_score(testy, y_pred)

    f = open(tfile + ".elastic_net.tsv", "w")
    f.write("Elastic Net coefficients: ")
    for x in range(0, len(enet_model.coef_)):
        f.write(str(tdata.columns[x]) + "\t" + str(enet_model.coef_[x]) + "\n")
    f.write("Error: " + str(terror) + "\n")
    f.write("r2: " + str(r2) + "\n")
    f.write('alpha:' + str(model.alpha_) + "\n")
    f.write('l1_ratio_:' + str(model.l1_ratio_) + "\n")
    f.close()
    #print("Elastic Net Regression")

    #print("Univariate data: " + "\n")
    #for tid in univardata:
    #    print(tid)
    #    print(univardata[tid]["beta"])
    
    # print out univariate and multivariate data
    tBonferroni = 0
    for tid in univardata:
        tBonferroni += 1
    tBonferroni = 0.01 / (tBonferroni * 2)
    
    padj = ""
    f = open(tfile, "w")
    f.write("Variable\tUnivariate Beta\tPvalue\tpadj\tr2\tMultivariate Beta\tPvalue\tpadj\tr2\n")
    for tid in univardata:
        ubeta = '{:.2f}'.format(float(univardata[tid]["beta"]))
        mbeta = '{:.2f}'.format(float(multivardata[tid]["beta"]))
        fupval = float(univardata[tid]["pval"])
        fmpval = float(multivardata[tid]["pval"])
        upval = '{:.2E}'.format(Decimal(fupval))

        uniadj = ""
        if fupval < tBonferroni:
            uniadj = "*"

        multiadj = ""
        if fmpval < tBonferroni:
            multiadj = "*"
            
        mpval = '{:.2E}'.format(Decimal(fmpval))
        ur2 = '{:.2f}'.format(float(univardata[tid]["r2_adj"]))
        mr2 = '{:.2f}'.format(r2_adj)
        f.write(tid + "\t" + ubeta + "\t" + upval + "\t" + uniadj + "\t" + ur2 + "\t" + mbeta + "\t" + mpval + "\t" + multiadj + "\t" + mr2 + "\n")
    f.close()
   
def GridSearch(tmode, X, y, subX, suby):

	X.columns = X.columns.str.replace("[\[,\],<,>]", '')
	subX.columns = subX.columns.str.replace("[\[,\],<,>]", '')

	# subset of data
	X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.5, random_state=104, shuffle=True)	

	# Default
	# Oct 23/24 - change max_depth to 1,2 from 3,4,5; max child weight from 1, 5, 10 to 10, 20, 30
	params_grid = {
		'min_child_weight': [10, 20, 30],
		'gamma': [0.5, 1, 1.5, 2, 5],
		'subsample': [0.6, 0.8, 1.0],
		'colsample_bytree': [0.6, 0.8, 1.0],
		'max_depth': [1, 2]
	}

	# for Stroke Belt overfitting
	# https://stackoverflow.com/questions/69786993/tuning-xgboost-hyperparameters-with-randomizedsearchcv
	# booster=['gbtree', 'gblinear']
	# AUC = 0.95 with 'min_child_weight': [50, 100, 200], 'gamma': [50, 75, 100], 'subsample': [0.6, 0.8, 1.0], 'colsample_bytree': [0.5, 0.75, 1], 'max_depth': [1]
	# AUC = 0.94 with gamma = 75
	# AUC = 0.92, (Hubs = 0.88), 'min_child_weight': [0.1, 1, 2, 5], 'gamma': [70, 75, 80], 'subsample': [0.3, 0.4, 0.5], 'colsample_bytree': [0.8, 0.9, 1], 'max_depth': [1]
	# AUC = 0.91 with colsample_bytree = 1; gamma = 100
	if tmode == "STROKE_BELT":
		params_grid = {
			'min_child_weight': [60],
			'gamma': [1, 10, 20],
			'subsample': [0.3, 0.4, 0.5],
			'colsample_bytree': [0.9, 0.95, 1],
			'max_depth': [1, 2]
		}

	print(tmode + " parameters:" + str(params_grid))
	params_fixed = {'objective':'binary:logistic'}
	xgb = XGBClassifier(**params_fixed)
	grid_search = GridSearchCV(estimator=xgb, param_grid=params_grid, cv=3, n_jobs=10, verbose=True, scoring=['neg_log_loss'], refit='neg_log_loss')
	grid_search.fit(X_train, y_train)
	
	# show best parameters
	best_params = grid_search.best_params_ 
	accuracy = grid_search.best_score_ 	
	print("best_params: " + str(best_params) + ", accuracy: " + str(accuracy))
	
	return best_params

	
def CalcXGBoost(tmode, X, y, subX, suby, xfile, tgamma):

	univardata = defaultdict(lambda: defaultdict(str))
	
	# relabel columns
	X.columns = X.columns.str.replace("[\[,\],<,>]", '')
	subX.columns = subX.columns.str.replace("[\[,\],<,>]", '')
		
	# XGBoost parameters
	ttest_size = 0.5
	num_kfold = 10

	# deal with disease / stroke imbalance using scale_pos_weight
	X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=ttest_size, random_state=104, shuffle=True)
	ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]
	
	# subset of data
	subX_train, subX_test, suby_train, suby_test = train_test_split(subX, suby, test_size=ttest_size, random_state=104, shuffle=True)
	#subratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]
	
	# param from GridSearch
	#gparam = {'colsample_bytree': 0.6, 'gamma': 1, 'max_depth': 3, 'min_child_weight': 1, 'subsample': 1.0}
	gparam = GridSearch(tmode, X, y, subX, suby)
	gparam['objective'] = 'binary:logistic'
	print("Parameter: " + str(gparam))	
	model = XGBClassifier(**gparam)	
	model.fit(subX_train, suby_train)	
	y_pred_proba = model.predict_proba(X_test)[::,1]	
	auc = metrics.roc_auc_score(y_test, y_pred_proba)
	fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
	
	# k-fold CV
	params = {}
	data_dmatrix = xgb.DMatrix(data=X, label=y)	
	xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3, metrics="error", as_pandas=True, seed=123)
	xgb_cv2 = cv(dtrain=data_dmatrix, params=params, nfold=3, metrics="auc", as_pandas=True, seed=123)
	
	#write out accuracy
	aFile = xfile + ".kfold.csv"
	#print("Creating accuracy file: " + aFile)
	f = open(aFile, "w")
	f.write(xgb_cv.to_string())
	f.write("---------------------------------------------------------------------\n")
	f.write("Accuracy Values: " + str(np.array((1 - xgb_cv['test-error-mean'])).round(2)) + "\n")
	f.write("Average Accuracy: " +  str((1 - xgb_cv['test-error-mean']).mean()) + "\n")
	f.write("---------------------------------------------------------------------\n")
	f.write(xgb_cv2.to_string())
	f.write("---------------------------------------------------------------------\n")
	for tid in xgb_cv2:
		f.write(tid + " Mean: " + str(xgb_cv2[tid].mean()) + "\n")
	f.close()
		
	# now calculate xgboost 10 times, and take the average of the k-fold CV
	iFile = xfile + ".importance.jpg"
	#print("Creating importance file: " + iFile)
	importance_vals = pd.DataFrame(columns=['columns', 'importances'])
	#importance_vals = defaultdict(float)
	for kfold in range(0, num_kfold):
	
		#print("K-fold CV iteration " + str(kfold))

		# deal with disease / stroke imbalance using scale_pos_weight
		X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=ttest_size, random_state=104, shuffle=True)
		ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]
		
		# subset of data
		subX_train, subX_test, suby_train, suby_test = train_test_split(subX, suby, test_size=ttest_size, random_state=104, shuffle=True)
		#subratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]
		
		#eval_set = [(X_test,y_test)]
		#model = XGBClassifier(learning_rate=tlearn, max_depth=tmax_depth, scale_pos_weight=ratio, gamma=tgamma)
		#model = XGBClassifier(learning_rate=tlearn, max_depth=tmax_depth, scale_pos_weight=ratio, gamma=tgamma, subsample=tsubsample)
		
		model = XGBClassifier(**gparam)

		# calculate k-fold
		X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=ttest_size, random_state=104, shuffle=True)
		#eval_set = [(X_test,y_test)]
		#model = XGBClassifier(learning_rate=tlearn, max_depth=tmax_depth, scale_pos_weight=ratio)

		#model.fit(X_train, y_train, early_stopping_rounds=tstop, eval_set=eval_set)
		#model.fit(X_train, y_train)
		model.fit(subX_train, suby_train)

		y_pred_proba = model.predict_proba(X_test)[::,1]

		#y_pred_proba = model.predict_proba(X_test)[::,1]
		#auc = metrics.roc_auc_score(y_test, y_pred_proba)
		#fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
					
		kfold_results = pd.DataFrame()
		kfold_results['columns'] = X.columns
		kfold_results['importances'] = model.feature_importances_
				
		importance_vals['columns'] = X.columns
		for xval in range(0, len(importance_vals.index)):
			tval = float(kfold_results.loc[xval, 'importances'])
			#print("value: " + str(tval))
			if pd.isnull(importance_vals.loc[xval, 'importances']) == True:
				importance_vals.loc[xval, 'importances'] = tval
			else:
				importance_vals.loc[xval, 'importances'] += tval
			
	# now take average
	importance_vals['importances'] = importance_vals['importances'] / num_kfold

	# get the top 20
	sorted_data = importance_vals.sort_values(by='importances', ascending=False)
	top_sorted = sorted_data[0:20]

	# save to file
	sorted_data.to_csv(xfile + ".importance.csv")

	# flip order because graphing displays in ascending order for some reason
	top_sorted = top_sorted.sort_values(by='importances', ascending=True)
	
	# plot feature importance
	TopValsBarChart(top_sorted, iFile, "Feature Importance")
	
	return fpr, tpr, auc, univardata, y_test, y_pred_proba, gparam

def MakeRefROCFigure(tmode, merge_data_counties, chemical_data, chemlist, mainfield, tthreshold, out_file):

	tcolors = ["blue", "green", "red", "purple", "orange", "yellow"]
	if tmode != "ejscreen":
		tlabels = ["SDOH Model", "EJSCREEN Model", "Prevention Model", "AirToxScreen Model", "AirToxScreen Model"]
	if tmode == "ejscreen":
		tlabels = ["SDOH Model", "EJSCREEN Model", "Prevention Model", "AirToxScreen Model", "AirToxScreen Model"]
		
	#tfield = "LIFE_EXPECT_THRESHOLD"
	tfield = mainfield + "_THRESHOLD"	
	tfile = out_file
	
	if mainfield == "STROKE_MORT":
		merge_data_counties = LoadStrokeMort(merge_data_counties, "none", tthreshold, "merge_threshold_data")
		chemical_data = LoadStrokeMort(chemical_data, "none", tthreshold, "merge_threshold_data")
		
	if mainfield == "STROKE_BELT":
		merge_data_counties = LoadStrokeBelt(merge_data_counties, "none", 0, "merge_threshold_data")
		chemical_data = LoadStrokeBelt(chemical_data, "none", 0, "merge_threshold_data")

	if mainfield == "LIFE_EXPECT":
		merge_data_counties = LoadLifeExpectancy(merge_data_counties, "none", tthreshold, "merge_threshold_data")
		chemical_data = LoadLifeExpectancy(chemical_data, "none", tthreshold, "merge_threshold_data")

	if (mainfield != "STROKE_MORT") and (mainfield != "LIFE_EXPECT") and (mainfield != "STROKE_BELT"):
		merge_data_counties = MergeDiscreteVals(merge_data_counties[[mainfield]], merge_data_counties, mainfield, tthreshold)
		chemical_data = MergeDiscreteVals(merge_data_counties[[mainfield]], chemical_data, mainfield, tthreshold)
					
	# select only columns with certain variance - need this for logistic regression
	tfract = 0.60
	tgamma = 10
	if mainfield == "STROKE_BELT":
		tgamma = 100
		
	print(mainfield + "\tFraction: " + str(tfract) + "\tGamma: " + str(tgamma))

	# Split the merge_data_counties into a "test" and "train" datasets
	# First, to a train/test split in MakeROCFigure - get a "subsample"
	# Second, do another train/test split with CalcXGBoost - split this
	#subdata = merge_data_counties.sample(frac=tfract)
	merge_data_counties_train, merge_data_counties_test, y_train_dummy, y_test_dummy = train_test_split(merge_data_counties, merge_data_counties, test_size=tfract, random_state=104, shuffle=True)	
	
	current_pollution_model = pollution_model.copy()
	if tmode == "ejscreen":
		current_pollution_model = all_pollution_model.copy()
	
	# use whole dataset
	merge_data_counties_test = merge_data_counties
	merge_data_counties_train = merge_data_counties
	chemical_data_test = chemical_data
	chemical_data_train = chemical_data

	y = merge_data_counties_test[[tfield]]
	X1 = merge_data_counties_test[demographics_model]
	X2 = merge_data_counties_test[current_pollution_model]
	
	# remove disease for y in X
	new_disease_columns = disease_model.copy()
	if mainfield in new_disease_columns:
		new_disease_columns.remove(mainfield)
	X3 = merge_data_counties_test[new_disease_columns]	
	vformatted_chemical_data = variance_threshold_selector(chemical_data, 0.001)
	vformatted_chemical_data_train, vformatted_chemical_data_test, y_train_dummy, y_test_dummy = train_test_split(vformatted_chemical_data, vformatted_chemical_data, test_size=tfract, random_state=104, shuffle=True)

	# use whole dataset
	vformatted_chemical_data_test = vformatted_chemical_data
	vformatted_chemical_data_train = vformatted_chemical_data

	#subdata_chem = chemical_data.sample(frac=tfract)
	chemical_data_train, chemical_data_test, y_train_dummy, y_test_dummy = train_test_split(chemical_data, chemical_data, test_size=tfract, random_state=104, shuffle=True)
	
	y2 = chemical_data_test[[tfield]]
	yv = vformatted_chemical_data_test[[tfield]]
	suby2v = vformatted_chemical_data_train[[tfield]]
	suby = merge_data_counties_train[[tfield]]
	suby2 = chemical_data_train[[tfield]]
	subX1 = merge_data_counties_train[demographics_model]
	subX2 = merge_data_counties_train[current_pollution_model]
	subX3 = merge_data_counties_train[new_disease_columns]						
	X4 = chemical_data_test[chemlist]
	subX4 = chemical_data_train[chemlist]

	if tfield in vformatted_chemical_data.columns:
		vformatted_chemical_data = vformatted_chemical_data.drop(tfield, axis=1)
	Xv = vformatted_chemical_data_test
	
	if tfield in vformatted_chemical_data_train.columns:
		vformatted_chemical_data_train = vformatted_chemical_data_train.drop(tfield, axis=1)		
	subX4v = vformatted_chemical_data_train
	
	# do grid search for stroke_belt
	#if mainfield == 'STROKE_BELT':
	#	# do grid search
	#	GridSearch(X4, y2, subX4, suby2)
	
	#for tgamma in range(10, 110, 10):
	# CalibrationPlot(y_test, prob_pos, cal_file)
	fpr1, tpr1, auc1, dat1, y_test1, prob_pos1 = CalcXGBoost(mainfield, X1, y, subX1, suby, out_file + "." + tlabels[0] + "_model.xgboost.pdf", tgamma)
	fpr2, tpr2, auc2, dat2, y_test2, prob_pos2 = CalcXGBoost(mainfield, X2, y, subX2, suby, out_file + "." + tlabels[1] + "_model.xgboost.pdf", tgamma)
	fpr3, tpr3, auc3, dat3, y_test3, prob_pos3 = CalcXGBoost(mainfield, X3, y, subX3, suby, out_file + "." + tlabels[2] + "_model.xgboost.pdf", tgamma)
	if tmode != "ejscreen":
		fpr4, tpr4, auc4, dat4, y_test4, prob_pos4 = CalcXGBoost("STROKE_MORT", X4, y2, subX4, suby2, out_file + "." + tlabels[3] + "_model.xgboost.pdf", tgamma)
				   
	plt.close("all")
	#fig.clf()	
	#fig = plt.figure(figsize=(5, 4))
	fig, ax = plt.subplots(figsize=(5, 4))
	plt.axis("on")
	ax.xaxis.set_visible(True)
	ax.yaxis.set_visible(True)
	ax.set_facecolor('xkcd:white')

	# order by AUC
	text_labels = defaultdict(float)
	text_labels[str(auc1)] = 0
	text_labels[str(auc2)] = 1
	text_labels[str(auc3)] = 2
	if tmode != "ejscreen":
		text_labels[str(auc4)] = 3
	
	ypos = 0.25
	colorcnt = 3
	tfont = 11
	tleft = 0.4
	#if (pairMatch == True) or (networkMatch == True):
	#	tfont = 8
	#	tleft = 0.2
	for tkey in sorted(text_labels, reverse=True):
		tauc_val = round(float(tkey), 2)
		tindex = text_labels[tkey]
		plt.text(tleft, ypos, "AUC: " + str(tauc_val) + " (" + tlabels[tindex] + ")", fontsize = tfont, color=tcolors[tindex])
		ypos = ypos - 0.05
		colorcnt = colorcnt - 1		

	plt.plot(fpr1, tpr1, color=tcolors[0])
	plt.plot(fpr2, tpr2, color=tcolors[1])
	plt.plot(fpr3, tpr3, color=tcolors[2])
	if tmode != "ejscreen":
		plt.plot(fpr4, tpr4, color=tcolors[3])

	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.margins(x=0, y=0)
	fig.tight_layout()
	
	final_file = tfile
	if tmode == "ejscreen":
		final_file = tfile + ".ejscreen.jpg"
	
	plt.savefig(tfile, dpi=300)
	
	###########################
	### Logistic Regression ###
	###########################
	
	fpr1, tpr1, auc1, dat1, y_test1, prob_pos1 = CalcRegression(X1, y, subX1, suby)
	fpr2, tpr2, auc2, dat2, y_test2, prob_pos2 = CalcRegression(X2, y, subX2, suby)
	fpr3, tpr3, auc3, dat3, y_test3, prob_pos3 = CalcRegression(X3, y, subX3, suby)

	# use variance-thresholded data:
	dropcols = ["STROKE_BELT", "LIFE_EXPECT", "STROKE_MORT"] + disease_columns.copy()
	for tid in dropcols:
		if tid in Xv.columns:
			Xv = Xv.drop(tid, axis=1)
		if tid in subX4v.columns:
			subX4v = subX4v.drop(tid, axis=1)
		nid = tid + "_THRESHOLD"
		if nid in Xv.columns:
			Xv = Xv.drop(nid, axis=1)
		if nid in subX4v.columns:
			subX4v = subX4v.drop(nid, axis=1)

	if tmode != "ejscreen":
		fpr4, tpr4, auc4, dat4, y_test4, prob_pos4 = CalcRegression(Xv, yv, subX4v, suby2v)
	
		# get the top 20
		ttable = []
		for tid in dat4:
			pval = float(dat4[tid]['p'])
			beta = float(dat4[tid]['beta'])
			nlabel = tid
			if (pval <= 0.01):
				nlabel = tid + "*"
			trow = [nlabel, beta]
			ttable.append(trow)
		
		bar_data = pd.DataFrame(ttable, columns=['columns', 'importances'])
		bar_data[['importances']] = bar_data[['importances']].apply(pd.to_numeric)
		#bar_data[['pvals']] = bar_data[['pvals']].apply(pd.to_numeric)	
		bar_data = bar_data.sort_values(by='importances', ascending=False)
		top_sorted = bar_data[0:20]
		#bar_data.to_csv(xfile + ".importance.csv")
		# reverse the order for plotting
		top_sorted = top_sorted.sort_values(by='importances', ascending=True)
		TopValsBarChart(top_sorted, tfile + ".beta.logistic.jpg", "β Coefficient")
	
	plt.close("all")
	fig, ax = plt.subplots(figsize=(5, 4))
	plt.axis("on")
	ax.xaxis.set_visible(True)
	ax.yaxis.set_visible(True)
	ax.set_facecolor('xkcd:white')

	# order by AUC
	text_labels = defaultdict(float)
	text_labels[str(auc1)] = 0
	text_labels[str(auc2)] = 1
	text_labels[str(auc3)] = 2
	
	if tmode != "ejscreen":
		text_labels[str(auc4)] = 3
	
	ypos = 0.25
	colorcnt = 3
	tfont = 11
	tleft = 0.4
	#if (pairMatch == True) or (networkMatch == True):
	#	tfont = 8
	#	tleft = 0.2
	for tkey in sorted(text_labels, reverse=True):
		tauc_val = round(float(tkey), 2)
		tindex = text_labels[tkey]
		plt.text(tleft, ypos, "AUC: " + str(tauc_val) + " (" + tlabels[tindex] + ")", fontsize = tfont, color=tcolors[tindex])
		ypos = ypos - 0.05
		colorcnt = colorcnt - 1		

	plt.plot(fpr1, tpr1, color=tcolors[0])
	plt.plot(fpr2, tpr2, color=tcolors[1])
	plt.plot(fpr3, tpr3, color=tcolors[2])

	if tmode != "ejscreen":
		plt.plot(fpr4, tpr4, color=tcolors[3])

	#plt.majorticks_on()
	#ax.tick_params(tick1On=False)
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.margins(x=0, y=0)
	fig.tight_layout()
	
	final_file = tfile + ".logistic.jpg"
	if tmode == "ejscreen":
		final_file = tfile + ".ejscreen.logistic.jpg"
		
	plt.savefig(final_file, dpi=300)

def MakeMergeROCFigure(merge_data_counties, chemical_data, chemlist, mainfield, tthreshold, out_file):

	tcolors = ["blue", "green", "red", "purple", "orange", "yellow"]
	tlabels = ["SDOH Model", "Pollution Model", "Prevention Model", "Hub Pollutants"]
	#tfield = "LIFE_EXPECT_THRESHOLD"
	tfield = mainfield + "_THRESHOLD"	
	tfile = out_file
	auc_file = out_file + ".auc.tsv"
	
	if mainfield == "STROKE_MORT":
		merge_data_counties = LoadStrokeMort(merge_data_counties, "none", tthreshold, "merge_threshold_data")
		chemical_data = LoadStrokeMort(chemical_data, "none", tthreshold, "merge_threshold_data")
		
	if mainfield == "STROKE_BELT":
		merge_data_counties = LoadStrokeBelt(merge_data_counties, "none", 0, "merge_threshold_data")
		chemical_data = LoadStrokeBelt(chemical_data, "none", 0, "merge_threshold_data")

	if mainfield == "LIFE_EXPECT":
		merge_data_counties = LoadLifeExpectancy(merge_data_counties, "none", tthreshold, "merge_threshold_data")
		chemical_data = LoadLifeExpectancy(chemical_data, "none", tthreshold, "merge_threshold_data")

	if (mainfield != "STROKE_MORT") and (mainfield != "LIFE_EXPECT") and (mainfield != "STROKE_BELT"):
		merge_data_counties = MergeDiscreteVals(merge_data_counties[[mainfield]], merge_data_counties, mainfield, tthreshold)
		chemical_data = MergeDiscreteVals(merge_data_counties[[mainfield]], chemical_data, mainfield, tthreshold)
					
	# Getting Pollution List
	mainfield_match = mainfield.lower()
	if mainfield == "STROKE_MORT":
		mainfield_match = "stroke_data"	
	rdata = tfile.split('.')
	#rauc = rdata[len(rdata) - 3] + '.' + rdata[len(rdata) - 2]
	rauc = rdata[len(rdata) - 4] + '.' + rdata[len(rdata) - 3]

	# select only columns with certain variance - need this for logistic regression
	tfract = 0.50
	tgamma = 10

	chemical_data_train, chemical_data_test, y_train_dummy, y_test_dummy = train_test_split(chemical_data, chemical_data, test_size=tfract, random_state=104, shuffle=True)
	merge_data_counties_train, merge_data_counties_test, y_train_dummy, y_test_dummy = train_test_split(merge_data_counties, merge_data_counties, test_size=tfract, random_state=104, shuffle=True)	

	# use whole dataset
	merge_data_counties_test = merge_data_counties
	merge_data_counties_train = merge_data_counties
	chemical_data_test = chemical_data
	chemical_data_train = chemical_data

	# use variance-thresholded data:
	#vformatted_chemical_data = variance_threshold_selector(chemical_data, 0.00001)	
	vformatted_chemical_data = chemical_data	
	trimmed_chemlist = chemlist.copy()
	for tcol in chemlist:
		if tcol not in vformatted_chemical_data.columns:
			trimmed_chemlist.remove(tcol)
			print("Removed low variance column: " + tcol)

	# get all pollutants selected in merged data
	all_pollution_model = []
	for colnum in range(0, len(chemical_data.columns)):
		tcol = chemical_data.columns[colnum]
		if (tcol.find('_THRESHOLD') == -1) and (tcol.find(mainfield) == -1):
			all_pollution_model.append(tcol)

	vall_pollution_model = []
	for colnum in range(0, len(vformatted_chemical_data.columns)):
		tcol = vformatted_chemical_data.columns[colnum]
		if (tcol.find('_THRESHOLD') == -1):
			vall_pollution_model.append(tcol)

	### ---------------	
	### Append demographics to pollution models - Oct 20, 2024
	#adjcols = ['Percent Minority', 'Percent Low Income', 'Percent Less than HS Education']
	adjcols = ['Percent Minority', 'Percent Linguistic Isolation', 'Percent Unemployed']
	for tcol in adjcols:
		chemlist.append(tcol)
		if tcol not in all_pollution_model:
			all_pollution_model.append(tcol)
		if tcol not in demographics_model:
			demographics_model.append(tcol)
		if tcol not in disease_model:
			disease_model.append(tcol)
	for tfips in chemical_data.index:
		for tcol in adjcols:
			chemical_data.loc[tfips, tcol] = merge_data_counties.loc[tfips, tcol]
	### ---------------
	
	y = merge_data_counties[[tfield]]
	X1 = merge_data_counties[demographics_model]
	X2 = chemical_data[all_pollution_model]
	#X2 = vformatted_chemical_data[vall_pollution_model]
	X4 = chemical_data[chemlist]
	
	# remove disease for y in X
	new_disease_columns = disease_model.copy()
	if mainfield in new_disease_columns:
		new_disease_columns.remove(mainfield)
	X3 = merge_data_counties_test[new_disease_columns]	
	#print("Variance selected columns: " + str(vformatted_chemical_data.columns))
	vformatted_chemical_data_train, vformatted_chemical_data_test, y_train_dummy, y_test_dummy = train_test_split(vformatted_chemical_data, vformatted_chemical_data, test_size=tfract, random_state=104, shuffle=True)

	# use whole dataset
	vformatted_chemical_data_test = vformatted_chemical_data
	vformatted_chemical_data_train = vformatted_chemical_data
	
	y2 = chemical_data[[tfield]]
	yv = vformatted_chemical_data_test[[tfield]]
	suby2v = vformatted_chemical_data_train[[tfield]]
	suby = merge_data_counties[[tfield]]
	suby2 = chemical_data[[tfield]]
	#suby2 = vformatted_chemical_data[[tfield]]
	subX1 = merge_data_counties[demographics_model]
	subX2 = chemical_data[all_pollution_model]
	#subX2 = vformatted_chemical_data[vall_pollution_model]
	subX3 = merge_data_counties[new_disease_columns]						
	subX4 = chemical_data_train[chemlist]

	if tfield in vformatted_chemical_data.columns:
		vformatted_chemical_data = vformatted_chemical_data.drop(tfield, axis=1)
	Xv = vformatted_chemical_data_test

	# drop disease columns in independent variables
	X1 = DropPredCols(X1)
	X2 = DropPredCols(X2)
	X3 = DropPredCols(X3)
	X4 = DropPredCols(X4)
	subX1 = DropPredCols(subX1)
	subX2 = DropPredCols(subX2)
	subX3 = DropPredCols(subX3)
	subX4 = DropPredCols(subX4)
	
	# look for multiple duplicated columns
	print("Columns for X2: " + str(X2.columns))
	print("Columns for y2: " + str(y2.columns))
	print("Columns for subX2: " + str(subX2.columns))
	print("Columns for suby2: " + str(suby2.columns))

	tflag = "OTHER"
	if mainfield == "STROKE_BELT":
		tflag = "STROKE_BELT"
	fpr1, tpr1, auc1, dat1, y_test1, prob_pos1, param1 = CalcXGBoost("OTHER", X1, y, subX1, suby, out_file + "." + tlabels[0] + "_model.xgboost.pdf", tgamma)
	fpr2, tpr2, auc2, dat2, y_test2, prob_pos2, param2 = CalcXGBoost(tflag, X2, y2, subX2, suby2, out_file + "." + tlabels[1] + "_model.xgboost.pdf", tgamma)
	fpr3, tpr3, auc3, dat3, y_test3, prob_pos3, param3 = CalcXGBoost("OTHER", X3, y, subX3, suby, out_file + "." + tlabels[2] + "_model.xgboost.pdf", tgamma)
	fpr4, tpr4, auc4, dat4, y_test4, prob_pos4, param4 = CalcXGBoost(tflag, X4, y2, subX4, suby2, out_file + "." + tlabels[3] + "_model.xgboost.pdf", tgamma)
	
	# output files to D:\bphc\sdi\networks\*.calibration.jpg
	CalibrationPlot([y_test1, y_test2, y_test3, y_test4], [prob_pos1, prob_pos2, prob_pos3, prob_pos4], tfile + ".calibration.jpg", tlabels)
	
	# Oct 21/24 - Calculate DeLong CIs
	#return fpr, tpr, auc, univardata, y_test, y_pred_proba, gparam
	
	# Now make ROC Graph
	auc_list = [auc1, auc2, auc3, auc4]
	tpr_list = [tpr1, tpr2, tpr3, tpr4]
	fpr_list = [fpr1, fpr2, fpr3, fpr4]
	ci_list, delong_pval = delong.calc_ci(prob_pos1, prob_pos2, prob_pos3, prob_pos4, y_test1, y_test2, y_test3, y_test4)
	
	# output files to D:\bphc\sdi\networks\*.xgboost.jpg
	PlotROC(tlabels, tcolors, auc_list, tpr_list, fpr_list, ci_list, delong_pval, tfile + ".xgboost.jpg")
	
	auc_data = ''
	for i in range(0, 4):
		auc_data += "xgboost\t" + tlabels[i] + "\t" + str(auc_list[i]) + "\n"
							
	fpr1, tpr1, auc1, dat1, y_test1, prob_pos1 = CalcRegression(X1, y, subX1, suby)
	fpr3, tpr3, auc3, dat3, y_test3, prob_pos3 = CalcRegression(X3, y, subX3, suby)
	
	Xv4 = vformatted_chemical_data_test[trimmed_chemlist]
	subX4v = vformatted_chemical_data_train[trimmed_chemlist]
	Xv2 = vformatted_chemical_data_test
	subX2v = vformatted_chemical_data_train

	# drop columns
	Xv = DropPredCols(Xv)
	Xv2 = DropPredCols(Xv2)
	subX4v = DropPredCols(subX4v)
	subX2v = DropPredCols(subX2v)

	#dropcols = ["STROKE_BELT", "LIFE_EXPECT", "STROKE_MORT"] + disease_columns.copy()
	#for tid in dropcols:
	#	if tid in Xv.columns:
	#		Xv = Xv.drop(tid, axis=1)
	#	if tid in subX4v.columns:
	#		subX4v = subX4v.drop(tid, axis=1)
	#	nid = tid + "_THRESHOLD"
	#	if nid in Xv.columns:
	#		Xv = Xv.drop(nid, axis=1)
	#	if nid in subX4v.columns:
	#		subX4v = subX4v.drop(nid, axis=1)

	#	if tid in Xv2.columns:
	#		Xv2 = Xv2.drop(tid, axis=1)
	#	if tid in subX2v.columns:
	#		subX2v = subX2v.drop(tid, axis=1)
	#	nid = tid + "_THRESHOLD"
	#	if nid in Xv2.columns:
	#		Xv2 = Xv2.drop(nid, axis=1)
	#	if nid in subX2v.columns:
	#		subX2v = subX2v.drop(nid, axis=1)
	
	fpr2, tpr2, auc2, dat2, y_test2, prob_pos2 = CalcRegression(Xv2, yv, subX2v, suby2v)

	if len(subX4v.columns) >= 2:
		fpr4, tpr4, auc4, dat4, y_test4, prob_pos4 = CalcRegression(Xv4, yv, subX4v, suby2v)

	# Oct 21/24 - Calculate DeLong CIs
	# Now make ROC Graph
	auc_list = [auc1, auc2, auc3, auc4]
	tpr_list = [tpr1, tpr2, tpr3, tpr4]
	fpr_list = [fpr1, fpr2, fpr3, fpr4]
	ci_list, delong_pval = delong.calc_ci(prob_pos1, prob_pos2, prob_pos3, prob_pos4, y_test1, y_test2, y_test3, y_test4)

	# output files to D:\bphc\sdi\networks\*.xgboost.jpg
	PlotROC(tlabels, tcolors, auc_list, tpr_list, fpr_list, ci_list, delong_pval, tfile + ".logistic.jpg")

	for i in range(0, 4):
		auc_data += "elasticnet\t" + tlabels[i] + "\t" + str(auc_list[i]) + "\n"
		
	f = open(auc_file, "w")
	f.write(auc_data)
	f.close()

	# write out betas - get the top 20
	ttable = []
	for tid in dat2:
		#pval = float(dat4[tid]['p'])
		beta = float(dat2[tid]['beta'])
		nlabel = tid
		#if (pval <= 0.01):
		#	nlabel = tid + "*"
		trow = [nlabel, beta]
		ttable.append(trow)
	
	bar_data = pd.DataFrame(ttable, columns=['columns', 'importances'])
	bar_data[['importances']] = bar_data[['importances']].apply(pd.to_numeric)
	#bar_data[['pvals']] = bar_data[['pvals']].apply(pd.to_numeric)	
	bar_data = bar_data.sort_values(by='importances', ascending=False)
	top_sorted = bar_data[0:20]
	bar_data.to_csv(tfile + ".beta.csv")
	# reverse the order for plotting
	top_sorted = top_sorted.sort_values(by='importances', ascending=True)
	
	# output files to D:\bphc\sdi\networks\*.xgboost.jpg
	TopValsBarChart(top_sorted, tfile + ".beta.logistic.jpg", "β Coefficient")

	plt.close("all")
	# output files to D:\bphc\sdi\networks\*.logistic.calibration.jpg
	CalibrationPlot([y_test1, y_test2, y_test3, y_test4], [prob_pos1, prob_pos2, prob_pos3, prob_pos4], tfile + ".logistic.calibration.jpg", tlabels)
	
	# return param data
	param_data = ""
	param_list = [param1, param2, param3, param4]
	x = 0
	for tlabel in tlabels:
		param_data += mainfield + "\t" + tlabel + "\t" + str(param_list[x]) + "\n"
		x += 1

	return param_data

def DropPredCols(datavals):

	dropcols = ["STROKE_BELT", "LIFE_EXPECT", "STROKE_MORT", "Stroke_Belt", "Life_Expect", "Stroke_Mort"] + disease_columns.copy()
	for tid in dropcols:
		if tid in datavals.columns:
			datavals = datavals.drop(tid, axis=1)
		nid = tid + "_THRESHOLD"
		if nid in datavals.columns:
			datavals = datavals.drop(nid, axis=1)

	return datavals

def PlotROC(tlabels, tcolors, auc_list, tpr_list, fpr_list, civals, tpval, tfile):

	#plt.majorticks_on()
	#ax.tick_params(tick1On=False)
	plt.close("all")
	plt.clf()
	fig, ax = plt.subplots(figsize=(5, 4))
	plt.axis("on")
	plt.rcParams.update({'font.size': 14})
	ax.xaxis.set_visible(True)
	ax.yaxis.set_visible(True)
	ax.set_facecolor('xkcd:white')

	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.margins(x=0, y=0)
	fig.tight_layout()

	# order by AUC
	text_labels = defaultdict(float)
	for x in range(0, len(auc_list)):
		nval = str(auc_list[x])
		if nval in text_labels:
			text_labels[nval + '1'] = x
		if nval not in text_labels:
			text_labels[nval] = x
			
	ypos = 0.3
	colorcnt = 3
	tfont = 8
	tleft = 0.25
	plabel = ''
	for tkey in sorted(text_labels, reverse=True):
		tauc_val = round(float(tkey), 2)
		tindex = text_labels[tkey]
		cistr = str(civals[tindex][0]) + '-' + str(civals[tindex][1])
		plabel = ''
		if tpval[tindex] < 0.01:
			plabel = '*'
		if tlabels[tindex].find('Hub') > -1:
			plabel == ''
		print(tlabels[tindex] + "\t" + str(tpval[tindex]) + ": " + plabel)
		plt.text(tleft, ypos, "AUC: " + str(tauc_val) + " (" + cistr + ") " + tlabels[tindex] + ' ' + plabel, fontsize = tfont, color=tcolors[tindex])
		ypos = ypos - 0.055
		colorcnt = colorcnt - 1
	# write out DeLong p-value
	#plt.text(tleft, ypos, "p=" + str(tpval), fontsize = tfont, color='black')

	for x in range(0, 4):
		plt.plot(fpr_list[x], tpr_list[x], color=tcolors[x])

	plt.savefig(tfile, dpi=300)


def MakeChemROCFigure(merge_data_counties, chemical_data, chemlist, mainfield, tthreshold, out_file):

	tcolors = ["blue", "green", "red", "purple", "orange", "yellow"]
	tlabels = ["SDOH Model", "EJSCREEN Model", "Prevention Model", "Top 5 Pollutants", "Top 10 Pollutants"]
	#tfield = "LIFE_EXPECT_THRESHOLD"
	tfield = mainfield + "_THRESHOLD"	
	tfile = out_file
	
	if mainfield == "STROKE_MORT":
		merge_data_counties = LoadStrokeMort(merge_data_counties, "none", tthreshold, "merge_threshold_data")
		chemical_data = LoadStrokeMort(chemical_data, "none", tthreshold, "merge_threshold_data")
		
	if mainfield == "STROKE_BELT":
		merge_data_counties = LoadStrokeBelt(merge_data_counties, "none", 0, "merge_threshold_data")
		chemical_data = LoadStrokeBelt(chemical_data, "none", 0, "merge_threshold_data")

	if mainfield == "LIFE_EXPECT":
		merge_data_counties = LoadLifeExpectancy(merge_data_counties, "none", tthreshold, "merge_threshold_data")
		chemical_data = LoadLifeExpectancy(chemical_data, "none", tthreshold, "merge_threshold_data")

	if (mainfield != "STROKE_MORT") and (mainfield != "LIFE_EXPECT") and (mainfield != "STROKE_BELT"):
		merge_data_counties = MergeDiscreteVals(merge_data_counties[[mainfield]], merge_data_counties, mainfield, tthreshold)
		chemical_data = MergeDiscreteVals(merge_data_counties[[mainfield]], chemical_data, mainfield, tthreshold)
					
	# Getting Pollution List
	chem_list3 = []
	chem_list5 = []
	mainfield_match = mainfield.lower()
	if mainfield == "STROKE_MORT":
		mainfield_match = "stroke_data"	
	rdata = tfile.split('.')
	#rauc = rdata[len(rdata) - 3] + '.' + rdata[len(rdata) - 2]
	rauc = rdata[len(rdata) - 4] + '.' + rdata[len(rdata) - 3]

	# Get top network nodes
	network_file = "d:\\apeer\\networks\\airtox_stats_lancet.canonical." + mainfield_match + "." + str(rauc) + '.binarymap.tsv.txt'
	if mainfield_match == "stroke_belt":
		network_file = "d:\\apeer\\networks\\airtox_stats_lancet.canonical.stroke_belt.binarymap.tsv.txt"
	if (mainfield_match == "stroke_data") or (mainfield_match == "stroke_mort"):
		network_file = "d:\\apeer\\networks\\airtox_stats_lancet.canonical.stroke_data.." + str(rauc) + '.binarymap.tsv.txt'
	if mainfield_match == "life_expect":
		network_file = "d:\\apeer\\networks\\airtox_stats_lancet.canonical.life_expect.0.3.binarymap.tsv.txt"

	tcnt = 0
	print("Loading Network File: " + network_file)
	with open(network_file, "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			tid = ldata[0]
			if tcnt < 5:
				chem_list3.append(tid)
			if tcnt < 10:
				chem_list5.append(tid)
			tcnt += 1
	infile.close()
	print("3 Network nodes found: " + str(chem_list3))
	print("5 Network nodes found: " + str(chem_list5))

	# select only columns with certain variance - need this for logistic regression
	tfract = 0.60
	tgamma = 10
	if mainfield == "STROKE_BELT":
		tfract = 0.5
		tgamma = 47
	print(mainfield + "\tFraction: " + str(tfract) + "\tGamma: " + str(tgamma))

	# Split the merge_data_counties into a "test" and "train" datasets
	# First, to a train/test split in MakeROCFigure - get a "subsample"
	# Second, do another train/test split with CalcXGBoost - split this
	#subdata = merge_data_counties.sample(frac=tfract)
	#subdata_chem = chemical_data.sample(frac=tfract)
	chemical_data_train, chemical_data_test, y_train_dummy, y_test_dummy = train_test_split(chemical_data, chemical_data, test_size=tfract, random_state=104, shuffle=True)

	merge_data_counties_train, merge_data_counties_test, y_train_dummy, y_test_dummy = train_test_split(merge_data_counties, merge_data_counties, test_size=tfract, random_state=104, shuffle=True)	

	# use whole dataset
	merge_data_counties_test = merge_data_counties
	merge_data_counties_train = merge_data_counties
	chemical_data_test = chemical_data
	chemical_data_train = chemical_data

	y = merge_data_counties_test[[tfield]]
	X1 = merge_data_counties_test[demographics_model]
	X2 = merge_data_counties_test[pollution_model]
	X4 = chemical_data_test[chemlist]
	X5 = chemical_data_test[chemlist]
	
	# remove disease for y in X
	new_disease_columns = disease_model.copy()
	if mainfield in new_disease_columns:
		new_disease_columns.remove(mainfield)
	X3 = merge_data_counties_test[new_disease_columns]	
	vformatted_chemical_data = variance_threshold_selector(chemical_data, 0.001)	
	vformatted_chemical_data_train, vformatted_chemical_data_test, y_train_dummy, y_test_dummy = train_test_split(vformatted_chemical_data, vformatted_chemical_data, test_size=tfract, random_state=104, shuffle=True)

	# use whole dataset
	vformatted_chemical_data_test = vformatted_chemical_data
	vformatted_chemical_data_train = vformatted_chemical_data
	
	y2 = chemical_data_test[[tfield]]
	yv = vformatted_chemical_data_test[[tfield]]
	suby2v = vformatted_chemical_data_train[[tfield]]
	suby = merge_data_counties_train[[tfield]]
	suby2 = chemical_data_train[[tfield]]
	subX1 = merge_data_counties_train[demographics_model]
	subX2 = merge_data_counties_train[pollution_model]
	subX3 = merge_data_counties_train[new_disease_columns]						
	subX4 = chemical_data_train[chemlist]
	subX5 = chemical_data_train[chemlist]

	if tfield in vformatted_chemical_data.columns:
		vformatted_chemical_data = vformatted_chemical_data.drop(tfield, axis=1)
	Xv = vformatted_chemical_data_test
	
	#if tfield in vformatted_chemical_data_train.columns:
	#	vformatted_chemical_data_train = vformatted_chemical_data_train.drop(tfield, axis=1)		
	#subX4v = vformatted_chemical_data_train
	#subX5v = vformatted_chemical_data_train
	
	if len(chem_list3) > 1:
		X4 = chemical_data_test[chem_list3]
		subX4 = chemical_data_train[chem_list3]
		X5 = chemical_data_test[chem_list5]
		subX5 = chemical_data_train[chem_list5]

		vchem_list3 = []
		vchem_list5 = []
		for tcol in vformatted_chemical_data.columns:
			if tcol in chem_list3:
				vchem_list3.append(tcol)
			if tcol in chem_list5:
				vchem_list5.append(tcol)

		Xv4 = vformatted_chemical_data_test[vchem_list3]
		subX4v = vformatted_chemical_data_train[vchem_list3]
		Xv5 = vformatted_chemical_data_test[vchem_list5]
		subX5v = vformatted_chemical_data_train[vchem_list5]

	if len(chem_list3) < 2:
		X4 = chemical_data_test[chemlist]
		subX4 = chemical_data_train[chemlist]
		X5 = chemical_data_test[chemlist]
		subX5 = chemical_data_train[chemlist]

		vchem_list = []
		for tcol in vformatted_chemical_data.columns:
			if tcol in chemlist:
				vchem_list.append(tcol)
		
		Xv4 = vformatted_chemical_data_test[vchem_list]
		subX4v = vformatted_chemical_data_train[vchem_list]
		Xv5 = vformatted_chemical_data_test[vchem_list]
		subX5v = vformatted_chemical_data_train[vchem_list]

		tlabels[3] = "(None Found)"
		tlabels[4] = "(None Found)"
			
	#for tgamma in range(10, 110, 10):
	fpr1, tpr1, auc1, dat1, y_test1, prob_pos1 = CalcXGBoost(X1, y, subX1, suby, out_file + "." + tlabels[0] + "_model.xgboost.pdf", tgamma)
	fpr2, tpr2, auc2, dat2, y_test2, prob_pos2 = CalcXGBoost(X2, y, subX2, suby, out_file + "." + tlabels[1] + "_model.xgboost.pdf", tgamma)
	fpr3, tpr3, auc3, dat3, y_test3, prob_pos3 = CalcXGBoost(X3, y, subX3, suby, out_file + "." + tlabels[2] + "_model.xgboost.pdf", tgamma)
	fpr4, tpr4, auc4, dat4, y_test4, prob_pos4 = CalcXGBoost(X4, y2, subX4, suby2, out_file + "." + tlabels[3] + "_model.xgboost.pdf", tgamma)
	fpr5, tpr5, auc5, dat5, y_test5, prob_pos5 = CalcXGBoost(X5, y2, subX5, suby2, out_file + "." + tlabels[4] + "_model.xgboost.pdf", tgamma)
					
	CalibrationPlot([y_test1, y_test2, y_test3, y_test4, y_test5], [prob_pos1, prob_pos2, prob_pos3, prob_pos4, prob_pos5], tfile + ".calibration.jpg", tlabels)
					
	plt.close("all")
	#fig.clf()	
	#fig = plt.figure(figsize=(5, 4))
	fig, ax = plt.subplots(figsize=(5, 4))
	plt.axis("on")
	ax.xaxis.set_visible(True)
	ax.yaxis.set_visible(True)
	ax.set_facecolor('xkcd:white')

	# order by AUC
	text_labels = defaultdict(float)
	text_labels[str(auc1)] = 0
	text_labels[str(auc2)] = 1
	text_labels[str(auc3)] = 2
	text_labels[str(auc4)] = 3
	text_labels[str(auc5)] = 4
	
	ypos = 0.25
	colorcnt = 3
	tfont = 11
	tleft = 0.4
	for tkey in sorted(text_labels, reverse=True):
		tauc_val = round(float(tkey), 2)
		tindex = text_labels[tkey]
		plt.text(tleft, ypos, "AUC: " + str(tauc_val) + " (" + tlabels[tindex] + ")", fontsize = tfont, color=tcolors[tindex])
		ypos = ypos - 0.05
		colorcnt = colorcnt - 1		

	plt.plot(fpr1, tpr1, color=tcolors[0])
	plt.plot(fpr2, tpr2, color=tcolors[1])
	plt.plot(fpr3, tpr3, color=tcolors[2])
	plt.plot(fpr4, tpr4, color=tcolors[3])
	plt.plot(fpr5, tpr5, color=tcolors[4])

	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.margins(x=0, y=0)
	fig.tight_layout()
	plt.savefig(tfile, dpi=300)
	
	fpr1, tpr1, auc1, dat1, y_test1, prob_pos1 = CalcRegression(X1, y, subX1, suby)
	fpr2, tpr2, auc2, dat2, y_test2, prob_pos2 = CalcRegression(X2, y, subX2, suby)
	fpr3, tpr3, auc3, dat3, y_test3, prob_pos3 = CalcRegression(X3, y, subX3, suby)

	# use variance-thresholded data:
	Xv4 = vformatted_chemical_data_test[vchem_list3]
	subX4v = vformatted_chemical_data_train[vchem_list3]

	dropcols = ["STROKE_BELT", "LIFE_EXPECT", "STROKE_MORT"] + disease_columns.copy()
	for tid in dropcols:
		if tid in Xv.columns:
			Xv = Xv.drop(tid, axis=1)
		if tid in subX4v.columns:
			subX4v = subX4v.drop(tid, axis=1)
		nid = tid + "_THRESHOLD"
		if nid in Xv.columns:
			Xv = Xv.drop(nid, axis=1)
		if nid in subX4v.columns:
			subX4v = subX4v.drop(nid, axis=1)

	# sometimes the chemicals with the highest Jaccard are not in the high variance columns - may have 0 columns
	if len(subX4v.columns) >= 2:
		fpr4, tpr4, auc4, dat4, y_test4, prob_pos4 = CalcRegression(Xv4, yv, subX4v, suby2v)
	if len(subX5v.columns) >= 2:
		fpr5, tpr5, auc5, dat5, y_test5, prob_pos5 = CalcRegression(Xv5, yv, subX5v, suby2v)

	CalibrationPlot([y_test1, y_test2, y_test3, y_test4, y_test5], [prob_pos1, prob_pos2, prob_pos3, prob_pos4, prob_pos5], tfile + ".logistic.calibration.jpg", tlabels)

	plt.close("all")
	fig, ax = plt.subplots(figsize=(5, 4))
	plt.axis("on")
	ax.xaxis.set_visible(True)
	ax.yaxis.set_visible(True)
	ax.set_facecolor('xkcd:white')

	# order by AUC
	text_labels = defaultdict(float)
	auc_list = [auc1, auc2, auc3, auc4, auc5]
	for x in range(0, len(auc_list)):
		nval = str(auc_list[x])
		if nval in text_labels:
			text_labels[nval + '1'] = x			
		if nval not in text_labels:
			text_labels[nval] = x
		
	#text_labels[str(auc1)] = 0
	#if str(auc2) in text_labels:
	#	nval = str(auc2) + '1'
	#	text_labels[nval] = 1
		
	#text_labels[str(auc2)] = 1
	#text_labels[str(auc3)] = 2
	#if len(subX4v.columns) > 2:
	#	text_labels[str(auc4)] = 3
	#if len(subX5v.columns) > 2:
	#	text_labels[str(auc5)] = 4
		
	#print("AUCs: " + str(text_labels))
	
	ypos = 0.35
	colorcnt = 3
	tfont = 11
	tleft = 0.4
	for tkey in sorted(text_labels, reverse=True):
		tauc_val = round(float(tkey), 2)
		tindex = text_labels[tkey]
		plt.text(tleft, ypos, "AUC: " + str(tauc_val) + " (" + tlabels[tindex] + ")", fontsize = tfont, color=tcolors[tindex])
		ypos = ypos - 0.05
		colorcnt = colorcnt - 1		

	plt.plot(fpr1, tpr1, color=tcolors[0])
	plt.plot(fpr2, tpr2, color=tcolors[1])
	plt.plot(fpr3, tpr3, color=tcolors[2])
	plt.plot(fpr4, tpr4, color=tcolors[3])
	plt.plot(fpr5, tpr5, color=tcolors[4])

	#plt.majorticks_on()
	#ax.tick_params(tick1On=False)
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.margins(x=0, y=0)
	fig.tight_layout()
	plt.savefig(tfile + ".logistic.jpg", dpi=300)


def MakeEJSCREENROCFigure(merge_data_counties, chemical_data, chemlist, mainfield, tthreshold, out_file):

	# chemlist is the polldata list

	tcolors = ["blue", "green", "red", "purple", "orange", "yellow"]
	#tlabels = ["SDOH Model", "EJSCREEN Model", "Prevention Model", "Top 5 Pollutants", "Top 10 Pollutants"]
	tlabels = ["SDOH Model", "EJSCREEN Model", "Prevention Model", "Top 2 Pollutants", "Top 5 Pollutants"]
	#tfield = "LIFE_EXPECT_THRESHOLD"
	tfield = mainfield + "_THRESHOLD"	
	tfile = out_file
	
	if mainfield == "STROKE_MORT":
		merge_data_counties = LoadStrokeMort(merge_data_counties, "none", tthreshold, "merge_threshold_data")
		chemical_data = LoadStrokeMort(chemical_data, "none", tthreshold, "merge_threshold_data")
		
	if mainfield == "STROKE_BELT":
		merge_data_counties = LoadStrokeBelt(merge_data_counties, "none", 0, "merge_threshold_data")
		chemical_data = LoadStrokeBelt(chemical_data, "none", 0, "merge_threshold_data")

	if mainfield == "LIFE_EXPECT":
		merge_data_counties = LoadLifeExpectancy(merge_data_counties, "none", tthreshold, "merge_threshold_data")
		chemical_data = LoadLifeExpectancy(chemical_data, "none", tthreshold, "merge_threshold_data")

	if (mainfield != "STROKE_MORT") and (mainfield != "LIFE_EXPECT") and (mainfield != "STROKE_BELT"):
		merge_data_counties = MergeDiscreteVals(merge_data_counties[[mainfield]], merge_data_counties, mainfield, tthreshold)
		chemical_data = MergeDiscreteVals(merge_data_counties[[mainfield]], chemical_data, mainfield, tthreshold)
					
	# Getting Pollution List
	chem_list3 = []
	chem_list5 = []
	mainfield_match = mainfield.lower()
	if mainfield == "STROKE_MORT":
		mainfield_match = "stroke_data"	
	rdata = tfile.split('.')
	#rauc = rdata[len(rdata) - 3] + '.' + rdata[len(rdata) - 2]
	rauc = rdata[len(rdata) - 5] + '.' + rdata[len(rdata) - 4]

	# Get top network nodes
	network_file = "d:\\apeer\\networks\\pollution_stats_lancet.canonical." + mainfield_match + "." + str(rauc) + '.binarymap.tsv.txt'
	if (mainfield_match == "stroke_data") or (mainfield_match == "stroke_mort"):
		#network_file = "d:\\apeer\\networks\\pollution_stats_lancet.canonical.stroke_data.." + str(rauc) + '.binarymap.tsv.txt'
		network_file = "d:\\apeer\\networks\\pollution_stats_lancet.canonical.stroke_data..0.6.binarymap.tsv.txt"
	if mainfield_match == "stroke_belt":
		network_file = "d:\\apeer\\networks\\pollution_stats_lancet.canonical.stroke_belt.binarymap.tsv.txt"
	if mainfield_match == "life_expect":
		network_file = "d:\\apeer\\networks\\pollution_stats_lancet.canonical.life_expect.0.3.binarymap.tsv.txt"

	tcnt = 0
	print("Loading Network File: " + network_file)
	with open(network_file, "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			tid = ldata[0]
			if tcnt < 2:
				chem_list3.append(tid)
			if tcnt < 5:
				chem_list5.append(tid)
			tcnt += 1
	infile.close()
	print("3 Network nodes found: " + str(chem_list3))
	print("5 Network nodes found: " + str(chem_list5))

	# select only columns with certain variance - need this for logistic regression
	tfract = 0.60
	tgamma = 10
	if mainfield == "STROKE_BELT":
		tfract = 0.5
		tgamma = 47
	print(mainfield + "\tFraction: " + str(tfract) + "\tGamma: " + str(tgamma))

	# Split the merge_data_counties into a "test" and "train" datasets
	# First, to a train/test split in MakeROCFigure - get a "subsample"
	# Second, do another train/test split with CalcXGBoost - split this
	#subdata = merge_data_counties.sample(frac=tfract)
	#subdata_chem = chemical_data.sample(frac=tfract)
	#chemical_data_train, chemical_data_test, y_train_dummy, y_test_dummy = train_test_split(chemical_data, chemical_data, test_size=tfract, random_state=104, shuffle=True)

	#merge_data_counties_train, merge_data_counties_test, y_train_dummy, y_test_dummy = train_test_split(merge_data_counties, merge_data_counties, test_size=tfract, random_state=104, shuffle=True)	

	# use whole dataset
	merge_data_counties_test = merge_data_counties
	merge_data_counties_train = merge_data_counties
	chemical_data_test = merge_data_counties
	chemical_data_train = merge_data_counties

	y = merge_data_counties_test[[tfield]]
	X1 = merge_data_counties_test[demographics_model]
	X2 = merge_data_counties_test[chemlist]
	X4 = chemical_data_test[chemlist]
	X5 = chemical_data_test[chemlist]
	
	# remove disease for y in X
	new_disease_columns = disease_model.copy()
	if mainfield in new_disease_columns:
		new_disease_columns.remove(mainfield)
	X3 = merge_data_counties_test[new_disease_columns]	
	vformatted_chemical_data = variance_threshold_selector(chemical_data, 0.001)	
	vformatted_chemical_data_train, vformatted_chemical_data_test, y_train_dummy, y_test_dummy = train_test_split(vformatted_chemical_data, vformatted_chemical_data, test_size=tfract, random_state=104, shuffle=True)

	# use whole dataset
	vformatted_chemical_data_test = vformatted_chemical_data
	vformatted_chemical_data_train = vformatted_chemical_data
	
	y2 = chemical_data_test[[tfield]]
	yv = vformatted_chemical_data_test[[tfield]]
	suby2v = vformatted_chemical_data_train[[tfield]]
	suby = merge_data_counties_train[[tfield]]
	suby2 = chemical_data_train[[tfield]]
	subX1 = merge_data_counties_train[demographics_model]
	subX2 = merge_data_counties_train[chemlist]
	subX3 = merge_data_counties_train[new_disease_columns]						
	subX4 = chemical_data_train[chemlist]
	subX5 = chemical_data_train[chemlist]

	if tfield in vformatted_chemical_data.columns:
		vformatted_chemical_data = vformatted_chemical_data.drop(tfield, axis=1)
	Xv = vformatted_chemical_data_test
	
	#if tfield in vformatted_chemical_data_train.columns:
	#	vformatted_chemical_data_train = vformatted_chemical_data_train.drop(tfield, axis=1)		
	#subX4v = vformatted_chemical_data_train
	#subX5v = vformatted_chemical_data_train
	
	if len(chem_list3) > 1:
		X4 = chemical_data_test[chem_list3]
		subX4 = chemical_data_train[chem_list3]
		X5 = chemical_data_test[chem_list5]
		subX5 = chemical_data_train[chem_list5]

		vchem_list3 = []
		vchem_list5 = []
		for tcol in vformatted_chemical_data.columns:
			if tcol in chem_list3:
				vchem_list3.append(tcol)
			if tcol in chem_list5:
				vchem_list5.append(tcol)

		Xv4 = vformatted_chemical_data_test[vchem_list3]
		subX4v = vformatted_chemical_data_train[vchem_list3]
		Xv5 = vformatted_chemical_data_test[vchem_list5]
		subX5v = vformatted_chemical_data_train[vchem_list5]

	if len(chem_list3) < 2:
		X4 = chemical_data_test[chemlist]
		subX4 = chemical_data_train[chemlist]
		X5 = chemical_data_test[chemlist]
		subX5 = chemical_data_train[chemlist]

		vchem_list = []
		for tcol in vformatted_chemical_data.columns:
			if tcol in chemlist:
				vchem_list.append(tcol)
		
		Xv4 = vformatted_chemical_data_test[vchem_list]
		subX4v = vformatted_chemical_data_train[vchem_list]
		Xv5 = vformatted_chemical_data_test[vchem_list]
		subX5v = vformatted_chemical_data_train[vchem_list]

		tlabels[3] = "(None Found)"
		tlabels[4] = "(None Found)"
			
	#for tgamma in range(10, 110, 10):
	fpr1, tpr1, auc1, dat1, y_test1, prob_pos1 = CalcXGBoost(X1, y, subX1, suby, out_file + "." + tlabels[0] + "_model.xgboost.pdf", tgamma)
	fpr2, tpr2, auc2, dat2, y_test2, prob_pos2 = CalcXGBoost(X2, y, subX2, suby, out_file + "." + tlabels[1] + "_model.xgboost.pdf", tgamma)
	fpr3, tpr3, auc3, dat3, y_test3, prob_pos3 = CalcXGBoost(X3, y, subX3, suby, out_file + "." + tlabels[2] + "_model.xgboost.pdf", tgamma)
	fpr4, tpr4, auc4, dat4, y_test4, prob_pos4 = CalcXGBoost(X4, y2, subX4, suby2, out_file + "." + tlabels[3] + "_model.xgboost.pdf", tgamma)
	fpr5, tpr5, auc5, dat5, y_test5, prob_pos5 = CalcXGBoost(X5, y2, subX5, suby2, out_file + "." + tlabels[4] + "_model.xgboost.pdf", tgamma)
					
	CalibrationPlot([y_test1, y_test2, y_test3, y_test4, y_test5], [prob_pos1, prob_pos2, prob_pos3, prob_pos4, prob_pos5], tfile + ".calibration.jpg", tlabels)
					
	plt.close("all")
	#fig.clf()	
	#fig = plt.figure(figsize=(5, 4))
	fig, ax = plt.subplots(figsize=(5, 4))
	plt.axis("on")
	ax.xaxis.set_visible(True)
	ax.yaxis.set_visible(True)
	ax.set_facecolor('xkcd:white')

	# order by AUC
	text_labels = defaultdict(float)
	text_labels[str(auc1)] = 0
	text_labels[str(auc2)] = 1
	text_labels[str(auc3)] = 2
	text_labels[str(auc4)] = 3
	text_labels[str(auc5)] = 4
	
	ypos = 0.25
	colorcnt = 3
	tfont = 11
	tleft = 0.4
	for tkey in sorted(text_labels, reverse=True):
		tauc_val = round(float(tkey), 2)
		tindex = text_labels[tkey]
		plt.text(tleft, ypos, "AUC: " + str(tauc_val) + " (" + tlabels[tindex] + ")", fontsize = tfont, color=tcolors[tindex])
		ypos = ypos - 0.05
		colorcnt = colorcnt - 1		

	plt.plot(fpr1, tpr1, color=tcolors[0])
	plt.plot(fpr2, tpr2, color=tcolors[1])
	plt.plot(fpr3, tpr3, color=tcolors[2])
	plt.plot(fpr4, tpr4, color=tcolors[3])
	plt.plot(fpr5, tpr5, color=tcolors[4])

	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.margins(x=0, y=0)
	fig.tight_layout()
	plt.savefig(tfile, dpi=300)
	
	fpr1, tpr1, auc1, dat1, y_test1, prob_pos1 = CalcRegression(X1, y, subX1, suby)
	fpr2, tpr2, auc2, dat2, y_test2, prob_pos2 = CalcRegression(X2, y, subX2, suby)
	fpr3, tpr3, auc3, dat3, y_test3, prob_pos3 = CalcRegression(X3, y, subX3, suby)

	# use variance-thresholded data:
	dropcols = ["STROKE_BELT", "LIFE_EXPECT", "STROKE_MORT"] + disease_columns.copy()
	for tid in dropcols:
		if tid in Xv.columns:
			Xv = Xv.drop(tid, axis=1)
		if tid in subX4v.columns:
			subX4v = subX4v.drop(tid, axis=1)
		nid = tid + "_THRESHOLD"
		if nid in Xv.columns:
			Xv = Xv.drop(nid, axis=1)
		if nid in subX4v.columns:
			subX4v = subX4v.drop(nid, axis=1)

	# sometimes the chemicals with the highest Jaccard are not in the high variance columns - may have 0 columns
	if len(subX4v.columns) >= 2:
		fpr4, tpr4, auc4, dat4, y_test4, prob_pos4 = CalcRegression(Xv4, yv, subX4v, suby2v)
	if len(subX5v.columns) >= 2:
		fpr5, tpr5, auc5, dat5, y_test5, prob_pos5 = CalcRegression(Xv5, yv, subX5v, suby2v)

	CalibrationPlot([y_test1, y_test2, y_test3, y_test4, y_test5], [prob_pos1, prob_pos2, prob_pos3, prob_pos4, prob_pos5], tfile + ".logistic.calibration.jpg", tlabels)

	plt.close("all")
	fig, ax = plt.subplots(figsize=(5, 4))
	plt.axis("on")
	ax.xaxis.set_visible(True)
	ax.yaxis.set_visible(True)
	ax.set_facecolor('xkcd:white')

	# order by AUC
	text_labels = defaultdict(float)
	auc_list = [auc1, auc2, auc3, auc4, auc5]
	for x in range(0, len(auc_list)):
		nval = str(auc_list[x])
		if nval in text_labels:
			text_labels[nval + '1'] = x			
		if nval not in text_labels:
			text_labels[nval] = x
		
	#text_labels[str(auc1)] = 0
	#if str(auc2) in text_labels:
	#	nval = str(auc2) + '1'
	#	text_labels[nval] = 1
		
	#text_labels[str(auc2)] = 1
	#text_labels[str(auc3)] = 2
	#if len(subX4v.columns) > 2:
	#	text_labels[str(auc4)] = 3
	#if len(subX5v.columns) > 2:
	#	text_labels[str(auc5)] = 4
		
	#print("AUCs: " + str(text_labels))
	
	ypos = 0.35
	colorcnt = 3
	tfont = 11
	tleft = 0.4
	for tkey in sorted(text_labels, reverse=True):
		tauc_val = round(float(tkey), 2)
		tindex = text_labels[tkey]
		plt.text(tleft, ypos, "AUC: " + str(tauc_val) + " (" + tlabels[tindex] + ")", fontsize = tfont, color=tcolors[tindex])
		ypos = ypos - 0.05
		colorcnt = colorcnt - 1		

	plt.plot(fpr1, tpr1, color=tcolors[0])
	plt.plot(fpr2, tpr2, color=tcolors[1])
	plt.plot(fpr3, tpr3, color=tcolors[2])
	plt.plot(fpr4, tpr4, color=tcolors[3])
	plt.plot(fpr5, tpr5, color=tcolors[4])

	#plt.majorticks_on()
	#ax.tick_params(tick1On=False)
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.margins(x=0, y=0)
	fig.tight_layout()
	plt.savefig(tfile + ".logistic.jpg", dpi=300)

def MakeBestMatchFigure():

	tIter = 0
	tfile = "lancet_figure2.tsv"
	tlabel = "pollution"
	figure_table = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
	figure_table_text = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
	
	while tIter < 2:

		if tIter == 1:
			tfile = "lancet_figure3.tsv"
			tlabel = "airtox"

		polldata = GetTopPollutionPair(tfile, tlabel)
		for tid in sorted(polldata):
		
			tid_data = tid.split('.')
			tdisease = tid_data[2]
			tcutoff = tid_data[3] + '.' + tid_data[4]
			
			if (tdisease == "stroke_data"):
				tcutoff = tid_data[4] + '.' + tid_data[5]
			if (tdisease == "stroke_belt"):
				tcutoff = '0'
			
			for tdata in sorted(polldata[tid]):

				ldata = tdata.split("\t")
				tcol1 = ldata[0]
				tcol2 = ldata[1]
				nclust = int(ldata[2])
				jval = str(polldata[tid][tdata])
				jval = jval[0:6]

				# get reference map - lancet.canonical.Stroke.0.6.binarymap.png
				tfile = tid
				tfile = tfile.replace('.tsv', '.png')
				figure_table[tdisease][tcutoff]["reference"] = tfile
				figure_table_text[tdisease][tcutoff]["reference"] = FormatPercentile(tcutoff)
				
				tfile = mappath + "\\" + "lancet_" + tlabel + "_top." + tcol1 + '.' + tcol2 + ".county_cluster." + str(nclust) + '.map.png'
				tfile = FormatFilename(tfile)				
				figure_table[tdisease][tcutoff][tlabel] = tfile
				figure_table_text[tdisease][tcutoff][tlabel] = tcol1 + " and " + tcol2 + "<br>k=" + str(nclust) + " J=" + jval
	
		tIter += 1
		
	# Write out HTML Page
	
	# Get Disease List
	disease_list = []
	for tdisease in figure_table:
		disease_list.append(tdisease)

	mapwidth = "80"
	data_fields = ["reference", "pollution", "airtox"]
	tHTML = "<html>\n"

	for trow in range(0, 3):
	
		# make outer table
		row_diseases = disease_list[0:5]
		if trow == 1:
			row_diseases = disease_list[5:10]
		if trow == 2:
			row_diseases = disease_list[10:15]
		
		tHTML = tHTML + "<table><tr>\n"
		for tdisease in row_diseases:
			tHTML += "<td style=\"text-align: center; font-family: arial; font-size: 10px; background-color: lightblue\">" + tdisease + "</td>\n"
		tHTML = tHTML + "</tr><tr>\n"
		for tdisease in row_diseases:
			
			# make inner tables
			tHTML = tHTML + "<td valign=\"top\"><table>\n"
			tHTML = tHTML + "<tr style=\"text-align: center; font-family: arial; font-size: 6px; color: white; background-color: steelblue\"><td>Reference</td><td>EJSCREEN</td><td>AirToxScreen</td></tr>\n"
			for tcutoff in figure_table[tdisease]:
				tHTML = tHTML + "<tr>\n"
				for tfield in data_fields:
					tHTML = tHTML + "\t<td valign=\"top\" style=\"text-align: center; font-family: arial; font-size: 6px;\"><img src=\"" + figure_table[tdisease][tcutoff][tfield] + "\" style=\"width: " + mapwidth + "px;\"><br>" + figure_table_text[tdisease][tcutoff][tfield]
				tHTML = tHTML + "</tr>\n"
			tHTML = tHTML + "</table></td>\n"
				
		tHTML = tHTML + "</table>\n"
	
	tHTML = tHTML + "</html>\n"

	f = open("lancet_best_matches.html", "w")	
	f.write(tHTML)
	f.close()

def MakeROCFigureChemOld(ttag, mainfile, num_variables, fig_file):

	# Pairwise Pollution Maps
	tHTML = "<html><table style=\"border: 1px solid gray;\">"
	#trow = "<tr><td></td>\n"
	#for pairx in range(0, len(stroke_belt_lib_lancet.pollution_columns)):
	#	tlabel = stroke_belt_lib_lancet.pollution_columns[pairx]
	#	tlabel = tlabel.replace(" ", "\n<br>")
	#	tlabel = tlabel.replace("Air\n<br>", "Air ")
	#	trow = trow + "<td style=\"text-align: center; font-size: 10px; font-family: arial;\">" + tlabel + "</td>\n"
	#trow = trow + "</tr>\n"
	#tHTML = tHTML + trow

	# stroke belt map
	#bfile = "lancet.canonical.stroke_belt.binarymap.png"
	#ofile = "lancet.canonical.stroke_belt.binarymap.tsv"
	#maplist = LoadStrokeBelt(merge_data_counties, ofile, 0, "threshold")
	#ShowBinaryMap(maplist, bfile, json_data_county, light_gray, "blue")
	
	nclust = 3
	if (ttag == "airtox"):
		nclust = 2
	if (ttag == "pollution"):
		nclust = 5
	
	#disease_data = GetTopPollutants(mainfile, ttag, num_variables)
	disease_data = GetTopPollutionPair(mainfile, tlabel)
	for tdisease in disease_data:
	
		chemical_list = []
		for tpollutant in disease_data[tdisease]:
			chemical_list.append(tpollutant)

		chemstr_raw = ', '.join(chemical_list)
		chemstr_wrap = ''
		spacecnt = 0
		for x in range(0, len(chemstr_raw)):
			ttoken = chemstr_raw[x]
			if chemstr_raw[x] == " ":
				if spacecnt == 4:
					ttoken = "\n<br>"
				spacecnt += 1
			chemstr_wrap = chemstr_wrap + ttoken
			
		#for pairx in range(0, len(pollution_columns)):
		tlabel = tdisease
		tlabel_raw = tlabel
		tlabel = tlabel.replace(" ", "\n<br>")
		tlabel = tlabel.replace("Air\n<br>", "Air ")
		trow = "<tr>\n" + "<td style=\"text-align: center; font-size: 10px; font-family: arial;\">" + tlabel + "</td>\n"
				
		# get map
		if (tlabel_raw == "obesity") or (tlabel_raw == "stroke") or (tlabel_raw == "stroke_belt"):
			nclust = 3
		tmap_file = mappath + "\\" + "lancet_" + ttag + "_top." + tdisease + ".county_cluster." + str(nclust) + ".map.png"
		tmap_file = tmap_file.replace(' ', '')
		trow = trow + "<td style=\"text-align: center; font-family: arial; font-size: 8px;\"><img src=\"" + tmap_file + "\" style=\"width: 150px;\"><br>" + chemstr_wrap + "</td>"
		
		# fix stroke_data / stroke_mort issue
		if tlabel_raw == "stroke_data":
			tlabel_raw = "stroke_mort"
		
		quant_cutoffs = [0.6, 0.7, 0.8, 0.9]
		if tlabel_raw == "stroke_mort":
			quant_cutoffs = [0.4, 0.5, 0.6, 0.7]
		if tlabel_raw == "life_expect":
			quant_cutoffs = [0.2, 0.3, 0.4, 0.5]
		
		# get highest AUC
		# stroke_belt_roc.jpg.Stroke Belt_model.xgboost.pdf.kfold.csv
		# Stroke_roc.0.9.jpg.Stroke Belt_model.xgboost.pdf.kfold.csv
		# test-auc-mean Mean: 0.9693330392562711
		
		# get highest AUC
		max_auc = 0
		max_cutoff = 0
		for tcutoff in quant_cutoffs:
			
			tfile = tlabel_raw + "_roc." + str(tcutoff) + '.jpg.Stroke Belt_model.xgboost.pdf.kfold.csv'
			if tlabel_raw == "stroke_belt":
				tfile = tlabel_raw + "_roc.jpg.Stroke Belt_model.xgboost.pdf.kfold.csv"
				
			with open(tfile, "r") as infile:
				for line in infile:
					line = line.strip()
					if line.find("test-auc-mean Mean:") > -1:
						tdata = line.split(':')
						tauc = float(tdata[1])
						print(tdisease + "\t" + str(tcutoff) + "\t" + str(tauc))
						if tauc > max_auc:
							max_auc = tauc
							max_cutoff = tcutoff
			infile.close()
			
		print(tdisease + "\t" + str(max_auc))
		
		# get reference map - lancet.canonical.Arthritis.0.7.binarymap.png
		tmap_file = "lancet.canonical." + tlabel_raw + "." + str(max_cutoff) + '.binarymap.png'
		if (tlabel_raw == "stroke_mort"):
			tmap_file = "lancet.canonical.stroke_data.." + str(max_cutoff) + '.binarymap.png'
		if (tlabel_raw == "stroke_belt"):
			tmap_file = "lancet.canonical.stroke_belt.binarymap.png"
		
		#tmap_file = tmap_file.replace(' ', '')
		trow = trow + "<td style=\"text-align: center; font-family: arial; font-size: 8px;\"><img src=\"" + tmap_file + "\" style=\"width: 150px;\"><br>" + str(max_cutoff) + "</td>"

		tfile = tlabel_raw + "_roc." + str(max_cutoff) + '.jpg'
		if tlabel_raw == "stroke_belt":
			tfile = tlabel_raw + "_roc.jpg"
		
		tcell = "<td style=\"text-align: center; font-face: arial; font-size: 7px;\"><img src=\"" + tfile + "\" style=\"width: 130px;\">AUC = " + str(max_auc)[0:5] + "</td>\n"
		trow = trow + tcell + "\n"
		#print("ROC file: " + tfile)
		#tdata = merge_data_counties[[stroke_belt_lib_lancet.pollution_columns[pairx], stroke_belt_lib_lancet.pollution_columns[pairy]]]
		#stroke_belt_lib_lancet.ShowTractClustersOnMap(tdata, nclust, tfile, "county", json_data_county, showmap=True)
		trow = trow + "</tr>\n"
		tHTML = tHTML + trow
	tHTML = tHTML + "</table></html>"
	
	f = open(fig_file, "w")
	f.write(tHTML)
	f.close()

def MakeReferenceMapsFig(tmode):

	disease_list = ["Stroke", "Coronary Heart Disease", "Hypertension", "Arthritis", "Cancer", "Asthma", "COPD", "Depression", "Diabetes", "Renal Disease", "Obesity"]
	if tmode == "cardiometabolic":
		disease_list = ["Stroke", "Hypertension", "Coronary Heart Disease", "Diabetes", "Obesity"]
		
	twidth = "200"
	theight = "170"
	img_width = "80"
	tHTML = "<html>"
	tHTML += "\t<table>"

	# get life expectancy map
	tHTML += "\t\t<tr><td style=\"font-family: arial; text-align: right;\">Life Expectancy</td>\n"
	quant_cutoffs = [0.2, 0.3, 0.4, 0.5]
	for tquant in quant_cutoffs:
		bfile = "lancet.canonical.life_expect." + str(tquant) + ".binarymap.png"
		roc_file = "life_expect_roc." + str(tquant) + ".jpg"
		if tmode == "network":
			roc_file = "life_expect_roc." + str(tquant) + ".network.jpg"
			bfile = roc_file + ".binarymap.png.map.png"
		plabel = FormatPercentile(str(tquant))
		
		if os.path.exists(bfile):
			tHTML += "\t\t\t<td>\n"
			tHTML += "<div style=\"position: relative; display: block; z-index: 1; width: " + twidth + "px; height: " + theight + "px;\"><img style=\"width: " + twidth + "px;\" src=\"" + roc_file + "\">" 
			tHTML += "<div style=\"position: absolute; z-index: 10; left: 50%; top: 23%;\"><img style=\"width: " + img_width + "px;\" src=\"" + bfile + "\"></div>"
			tHTML += "<div style=\"position: absolute; z-index: 10; left: 60%; top: 51%; font-size: 9px; font-family: arial;\">&lt;" + plabel + "</div>"
			tHTML += "</div>"
			tHTML += "\t\t\t</td>\n"

		if not os.path.exists(bfile):
			tHTML += "\t\t\t<td bgcolor=\"gray\">&nbsp;\n"
			tHTML += "\t\t\t</td>\n"
			

	tHTML += "\t\t</tr>\n"
	
	# stroke map
	tHTML += "\t\t<tr><td style=\"font-family: arial; text-align: right;\">Stroke Mortality</td>\n"
	quant_cutoffs = [0.4, 0.5, 0.6, 0.7]
	for tquant in quant_cutoffs:
		bfile = "lancet.canonical.stroke_data." + "." + str(tquant) + ".binarymap.png"
		roc_file = "stroke_mort_roc." + str(tquant) + ".jpg"
		if tmode == "network":
			roc_file = "stroke_mort_roc." + str(tquant) + ".network.jpg"		
			bfile = roc_file + ".binarymap.png.map.png"
		plabel = FormatPercentile(str(tquant))
		
		if os.path.exists(bfile):
			tHTML += "\t\t\t<td>\n"
			tHTML += "<div style=\"position: relative; display: block; z-index: 1; width: " + twidth + "px; height: " + theight+ "px;\"><img style=\"width: " + twidth + "px;\" src=\"" + roc_file + "\">" 
			tHTML += "<div style=\"position: absolute; z-index: 10; left: 50%; top: 23%;\"><img style=\"width: " + img_width + "px;\" src=\"" + bfile + "\"></div>"
			tHTML += "<div style=\"position: absolute; z-index: 10; left: 60%; top: 51%; font-size: 9px; font-family: arial;\">&gt;" + plabel + "</div>"
			tHTML += "</div></td>\n"

		if not os.path.exists(bfile):
			tHTML += "\t\t\t<td bgcolor=\"gray\">&nbsp;\n"
			tHTML += "\t\t\t</td>\n"
		
	tHTML += "\t\t</tr>\n"
		
	# stroke belt map
	tHTML += "\t\t<tr><td style=\"font-family: arial; text-align: right;\">Stroke Belt</td>\n"
	bfile = "lancet.canonical.stroke_belt.binarymap.png"
	roc_file = "stroke_belt_roc.jpg"
	if tmode == "network":
		roc_file = "stroke_belt_roc.network.jpg"		
		bfile = roc_file + ".binarymap.png.map.png"
	tHTML += "\t\t\t<td>\n"
	tHTML += "<div style=\"position: relative; display: block; z-index: 1; width: " + twidth + "px; height: " + theight + "px;\"><img style=\"width: " + twidth + "px;\" src=\"" + roc_file + "\">" 
	tHTML += "<div style=\"position: absolute; z-index: 10; left: 50%; top: 23%;\"><img style=\"width: " + img_width + "px;\" src=\"" + bfile + "\"></div>"
	tHTML += "</div></td>\n"

	tHTML += "\t\t</tr>\n"
		
	quant_cutoffs = [0.6, 0.7, 0.8, 0.9]
	for tdisease in disease_list:
		tHTML += "\t\t<tr><td style=\"font-family: arial; text-align: right;\">" + tdisease + "</td>\n"
		for tquant in quant_cutoffs:
			bfile = "lancet.canonical." + tdisease + "." + str(tquant) + ".binarymap.png"
			roc_file = tdisease + "_roc." + str(tquant) + ".jpg"
			if tmode == "network":
				roc_file = tdisease + "_roc." + str(tquant) + ".network.jpg"		
				bfile = roc_file + ".binarymap.png.map.png"
			plabel = FormatPercentile(str(tquant))

			if os.path.exists(bfile):
				tHTML += "\t\t\t<td>\n"
				tHTML += "<div style=\"position: relative; display: block; z-index: 1; width: " + twidth + "px; height: " + theight + "px;\"><img style=\"width: " + twidth + "px;\" src=\"" + roc_file + "\">" 
				tHTML += "<div style=\"position: absolute; z-index: 10; left: 50%; top: 23%;\"><img style=\"width: " + img_width + "px;\" src=\"" + bfile + "\"></div>"
				tHTML += "<div style=\"position: absolute; z-index: 10; left: 60%; top: 51%; font-size: 9px; font-family: arial;\">&gt;" + plabel + "</div>"
				tHTML += "</div></td>\n"

			if not os.path.exists(bfile):
				tHTML += "\t\t\t<td bgcolor=\"gray\">&nbsp;\n"
				tHTML += "\t\t\t</td>\n"

		tHTML += "\t\t</tr>\n"
	
	tHTML += "\t</table>\n</html>"

	fig_file = "lancet_reference_maps." + tmode + ".html"
	f = open(fig_file, "w")
	f.write(tHTML)
	f.close()

def ClusterNetworks(ofile, tcutoff, tmode):

	#tcutoff = 0.7
	importance_data = pd.DataFrame()
	ndisease_list = disease_columns.copy() + ['stroke_mort']

	for tdisease in ndisease_list:

		pollution_list = []
		if tmode == "network":

			disease_label = tdisease
			tpath = "d:\\apeer\\networks"
			tfile = tpath + "\\" + "airtox_merge_stats_lancet.canonical." + tdisease + "." + str(tcutoff) + ".binarymap.tsv.txt"
			if tdisease == "stroke_mort":
				tfile = tpath + "\\" + "airtox_merge_stats_lancet.canonical.stroke_data.." + str(tcutoff) + ".binarymap.tsv.txt"
				disease_label = "Stroke Mortality"
			if tdisease == "stroke_belt":
				tfile = tpath + "\\" + "airtox_merge_stats_lancet.canonical.stroke_belt.binarymap.tsv.txt"
				disease_label = "Stroke Belt"
		
			with open(tfile, "r") as infile:
				for line in infile:
					line = line.strip()
					ldata = line.split("\t")
					thub_poll = ldata[2]
					
					if ldata[1] == "Hub":
						#f.write(str(trank) + "\t" + ishub + "\t" + tnode + "\t" + str(sort_degree[tnode]) + "\t" + str(sort_centrality[tnode]) + "\t" + str(hubval)  + "\t" + str(hub_authorities[tnode]) + "\n")
						#pollution_list.append(ldata[2])
						print(tdisease + "\t" + line)
						tval = float(ldata[3])
						importance_data.loc[thub_poll, disease_label] = tval
						
			infile.close()
					
		if (tmode == "logistic") or (tmode == "xgboost"):

			disease_label = tdisease
			tpath = "d:\\apeer\\networks"
			tsuffix = "_roc." + str(tcutoff) + ".merge.network.jpg.Pollution Model_model.xgboost.pdf.importance.csv"
			if tmode == "logistic":
				tsuffix = "_roc." + str(tcutoff) + ".merge.network.jpg.beta.csv"
			
			#STROKE_MORT_roc.0.7.merge.network.jpg.Pollution Model_model.xgboost.pdf.importance.csv
			tfile = tpath + "\\" + tdisease + tsuffix
			if tdisease == "stroke_mort":
				tfile = tpath + "\\STROKE_MORT" + tsuffix
				disease_label = "Stroke Mortality"
			if tdisease == "stroke_belt":
				tfile = tpath + "\\STROKE_BELT" + tsuffix
				disease_label = "Stroke Belt"
			
			print("Processing " + tmode + ": " + tfile)

			# 42,ACETALDEHYDE,4.963783026998435
			icnt = 0
			with open(tfile, "r") as infile:
				csv_reader = csv.reader(infile, delimiter=',')
				for row in csv_reader:
					if (icnt <= 5) and (icnt > 0):
						tpollutant = row[1]
						tbeta = float(row[2])
						trank = icnt + 1
						importance_data.loc[tpollutant, disease_label] = tbeta
						if tmode == "xgboost":
							importance_data.loc[tpollutant, disease_label] = trank
					icnt += 1
			infile.close()

	# drop rows that are all NAs
	importance_data = importance_data.dropna(axis = 0, how = 'all')
	
	# replace NAs
	importance_data = importance_data.fillna(0)

	# format columns
	collist = []
	for tcol in importance_data.index:
		ncol = tcol.title()
		ncol = ncol.replace('_', ' ')
		if ncol.find('(') > -1:
			bdata = ncol.split('(')
			ncol = bdata[0]
		if (ncol == "Diesel Pm"):
			ncol = "Diesel PM"
		collist.append(ncol)
	importance_data.index = collist
	
	# format rows (pollutant names)
	rowlist = []
	for tcol in importance_data.columns:
		ncol = tcol.title()
		if ncol.find('(') > -1:
			bdata = ncol.split('(')
			ncol = bdata[0]
		if (ncol == "Copd"):
			ncol = "COPD"
		rowlist.append(ncol)
	importance_data.columns = rowlist
		
	# create clusters
	sns.set(font_scale=3)
	g = sns.clustermap(importance_data, xticklabels=True, yticklabels=True, figsize=(13,20), tree_kws=dict(linewidths=3, colors=(0.2, 0.2, 0.4)))
	g.fig.subplots_adjust(right=0.6)
	g.ax_cbar.set_position((0.75, 0.05, 0.03, 0.2))
	g.fig.set_figwidth(15)
	g.fig.set_figheight(20)	
	plt.savefig(ofile, format="png")

def MakeSuppClusterFigure():

	cutoff_list = [0.6, 0.7, 0.8, 0.9]
	suffix_list = ['network', 'logistic', 'xgboost']
	tHTML = "<html>\n"
	tHTML += "<table border=0 style=\"font-family: arial;\">\n\t<tr><td></td>\n"
	for tcutoff in cutoff_list:
		tval = tcutoff * 100
		tval = str(round(tval, 0))[0:2] + "th Percentile"
		tHTML += "\t\t<td><center>" + tval + "</center></td>\n"
	tHTML += "\t</tr>\n"
	for tsuffix in suffix_list:
		tlabel = "aPEER Hubs"
		if tsuffix == "logistic":
			tlabel = "Elastic Net β Coefficients"
		if tsuffix == "xgboost":
			tlabel = "Random Forest"
		tHTML += "\t<tr><td><center>" + tlabel + "</center></td>\n"
		for tcutoff in cutoff_list:
			tHTML += "\t\t<td><img style=\"width: 175px;\" src=\"cluster_merged_" + str(tcutoff) + "_" + tsuffix + ".png\"></td>"
		tHTML += "\t</tr>\n"
	tHTML += "</table>\n"
	tHTML += "</html>\n"
	
	f = open("supp_figure_sensitivity.html", "w", encoding="utf-8")
	f.write(tHTML)
	f.close()

def ClusterImportance(tmode, ofile):

	# Arthritis_roc.0.6.jpg.AirToxScreen_model.xgboost.pdf.importance.csv
	#disease_list = ["Arthritis", "Hypertension", "Cancer", "Asthma", "Coronary Heart Disease", "COPD", "Depression", "Diabetes", "Renal Disease", "Obesity", "Stroke", "life_expect", "stroke_mort", "stroke_belt"]
	disease_list = ["Arthritis", "Hypertension", "Cancer", "Asthma", "Coronary Heart Disease", "COPD", "Depression", "Diabetes", "Renal Disease", "Obesity", "Stroke", "stroke_mort"]
	
	#quant_cutoffs = [0.6, 0.7, 0.8, 0.9]
	#quant_cutoffs = [0.7]
	tquant = 0.7
	tmax = 5

	def LoadCSV(tfile, tcol, importance_data):
	
		ncnt = 0
		with open(tfile, "r", encoding="utf-8") as infile:
			csv_reader = csv.reader(infile, delimiter=',')
			tcnt = 0
			for row in csv_reader:
				if tcnt > 0:
					tpoll = row[1]
					npoll = tpoll
					if npoll == "stroke_mort":
						npoll = "Stroke Mortality"
					tval = float(row[2])
					if tval > 0:
						if ncnt < tmax:
							importance_data.loc[npoll, tcol] = tval
						ncnt += 1
				tcnt += 1
		infile.close()
		
		return importance_data

	def LoadTSV(tfile, tcol, importance_data):
	
		ncnt = 0
		with open(tfile, "r", encoding="utf-8") as infile:
			for line in infile:
				line = line.strip()
				ldata = line.split("\t")
				tpoll = ldata[0]
				npoll = tpoll
				if npoll == "stroke_mort":
					npoll = "Stroke Mortality"
				tval = float(ldata[1])
				if ncnt < tmax:
					importance_data.loc[npoll, tcol] = tval
				ncnt += 1
		infile.close()
		
		return importance_data
		

	importance_data = pd.DataFrame()
	for tdisease in disease_list:
	
		#if tdisease == "life_expect":
		#	#quant_cutoffs = [0.2, 0.3, 0.4, 0.5]
		#	quant_cutoffs = [0.3]
			
		#if tdisease == "stroke_mort":
		#	#quant_cutoffs = [0.4, 0.5, 0.6, 0.7]
		#	quant_cutoffs = [0.7]
			
		if tdisease != "stroke_belt":
		
			tfile = tdisease + "_roc." + str(tquant) + ".jpg.AirToxScreen_model.xgboost.pdf.importance.csv"
			if tmode == "network":
				tfile = "d:\\apeer\\networks\\airtox_stats_lancet.canonical." + tdisease + "." + str(tquant) + '.binarymap.tsv.txt'
				if tdisease == "stroke_mort":
					tfile = "d:\\apeer\\networks\\airtox_stats_lancet.canonical.stroke_data.." + str(tquant) + '.binarymap.tsv.txt'
			tcol = tdisease + "." + str(tquant)
			#tcol = tcol.replace('.', '_')
			if tmode == "importance":
				importance_data = LoadCSV(tfile, tcol, importance_data)
			if tmode == "network":
				importance_data = LoadTSV(tfile, tcol, importance_data)
			
		if tdisease == "stroke_belt":
		
			tfile = "stroke_belt_roc.jpg.AirToxScreen_model.xgboost.pdf.importance.csv"
			if tmode == "network":
				tfile = "d:\\apeer\\networks\\airtox_stats_lancet.canonical." + tdisease + '.binarymap.tsv.txt'

			if tmode == "importance":
				importance_data = LoadCSV(tfile, tdisease, importance_data)
			if tmode == "network":
				importance_data = LoadTSV(tfile, tdisease, importance_data)
	
	# drop rows that are all NAs
	importance_data = importance_data.dropna(axis = 0, how = 'all')
	
	# replace NAs
	importance_data = importance_data.fillna(0)

	# create clusters
	sns.set(font_scale=1.5)
	g = sns.clustermap(importance_data, xticklabels=True, yticklabels=True, figsize=(20,20), tree_kws=dict(linewidths=3, colors=(0.2, 0.2, 0.4)))
	g.fig.subplots_adjust(right=0.6)
	g.ax_cbar.set_position((0.8, .6, .03, .2))
	g.fig.set_figwidth(25)
	g.fig.set_figheight(25)	
	plt.savefig(ofile, format="png")


def GetTopPollutantsByJaccard():

	# get p-value threshold
	#mainfile = "lancet_merge_figure3.tsv"
	mainfile = "lancet_merge_0.7_figure3.tsv"
	pval_cutoff = GetPValCutoff(mainfile)

	# output_file = "lancet_figure2_top_pollutants.tsv"	
	print("Loading Jaccard data")
	score_table = defaultdict(lambda: defaultdict(float))
	cluster_table = defaultdict(lambda: defaultdict(float))
	pairwise_count = defaultdict(int)
	top_jaccard_pollutants = defaultdict(str)
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
					pairwise_count[bdisease] += 1

				if tdesc in score_table[bdisease]:
					if tjaccard > score_table[bdisease][tdesc]:
						score_table[bdisease][tdesc] = tjaccard
						cluster_table[bdisease][tdesc] = tclust
						pairwise_count[bdisease] += 1

	for bdisease in score_table:
		maxj = 0
		tcnt = 1
		#rank_cutoff = pairwise_count[bdisease] * 0.01
		rank_cutoff = 100
		top_rank = ""
		top_rank_jaccard = 0
		print("Rank cutoff for " + bdisease + ": " + str(rank_cutoff) + "\t" + str(pairwise_count[bdisease]))
		for titem in sorted(score_table[bdisease], key=score_table[bdisease].get, reverse=True):

			if (tcnt == 1):
				print(bdisease + "\t" + titem + "\t" + str(score_table[bdisease][titem]))

			# Load Reference map data - lancet.canonical.Diabetes.0.7.binarymap.tsv
			disease_str = bdisease.split('.')
			curr_disease = disease_str[2]
			if curr_disease == "stroke_data":
				curr_disease = "STROKE_MORT"
			if curr_disease == "stroke_belt":
				curr_disease = "STROKE_BELT"			
			if tcnt <= rank_cutoff:
				pollutants = titem.split('|')
				tclust = cluster_table[bdisease][titem]
				#tline = str(tcnt) + "\t" + bdisease + "\t" + titem + "\t" + str(tclust) + "\t" + str(score_table[bdisease][titem])
				top_jaccard_pollutants[curr_disease] += titem + '!!'

			tcnt += 1

	return top_jaccard_pollutants

def CheckJaccardValues(tnorm_chemical_data, json_data_county):

	# pull out some sample pre-computed Jaccard Values
	# get p-value threshold
	mainfile = "lancet_merge_figure3.tsv"
	pval_cutoff = GetPValCutoff(mainfile)
	countycnt = 3141

	tdisease = "STROKE_MORT"

	# Load Reference map data
	tcutoff = 0.7
	rfile = "lancet.canonical." + tdisease + "." + str(tcutoff) + ".binarymap.tsv"
	if tdisease == "STROKE_MORT":
		rfile = "lancet.canonical.stroke_data." + "." + str(tcutoff) + ".binarymap.tsv"
	if tdisease == "STROKE_BELT":
		rfile = "lancet.canonical.stroke_belt.binarymap.tsv"

	# Transform clustered data
	rawdata2 = LoadClusterData(rfile)
	refbelt = defaultdict(int)
	for tid in rawdata2:
		if rawdata2[tid] == "1":
			refbelt[tid] = 1

	# create clusters
	tdata = tnorm_chemical_data[["ACETALDEHYDE", "BENZOAPYRENE"]]
	for tclust in range(2, 6):
		tfile = "check_jaccard_cluster" + str(tclust)
		ShowTractClustersOnMap(tdata, tclust, tfile, "county", json_data_county, showmap=False, forceBinary=False)
		ofile = tfile + ".cluster.tsv"
		rawdata = LoadClusterData(ofile)
		tjaccard, tpval = CalcJaccard(rawdata, refbelt, countycnt)
		print("STROKE_MORT" + "\t" + "ACETALDEHYDE" + "\t" + "BENZOAPYRENE" + "\t" + str(tjaccard) + "\t" + str(tpval))

	print("Loading Jaccard data")
	score_table = defaultdict(lambda: defaultdict(float))
	cluster_table = defaultdict(lambda: defaultdict(float))
	pairwise_count = defaultdict(int)
	top_jaccard_pollutants = defaultdict(str)
	with open(mainfile, "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			
			if ldata[0] == "lancet.canonical.stroke_data..0.7.binarymap.tsv":
				if ((ldata[1] == "ACETALDEHYDE") and (ldata[2] == "BENZOAPYRENE")) or ((ldata[2] == "ACETALDEHYDE") and (ldata[1] == "BENZOAPYRENE")):
					print(line)
				
	infile.close()

def ReferenceMapFigure():
	
	tcnt = 1
	rowcnt = 0
	tHTML = "<html>"
	tHTML += "<table><tr>"
	ndisease_columns = disease_columns.copy() + ["STROKE_MORT"]
	for tdisease in sorted(ndisease_columns):
		ttoken = tdisease
		tlabel = tdisease
		if tdisease == "STROKE_MORT":
			ttoken = "stroke_data."
			tlabel = "Stroke Mortality"
		tHTML += "<td><img src=\"lancet.canonical." + ttoken + ".0.7.binarymap.png\" style=\"width: 330px;\"><br><center><font style=\"font-family: arial; font-size: 24px;\">" + tlabel + "</font></center></td>" 
		if ((tcnt % 4) == 0):
			tHTML += "</tr><tr>"
			rowcnt += 1
		tcnt += 1
	
	tHTML += "</tr></table></html>"
	
	f = open("lancet_figure2_ref.html", "w")
	f.write(tHTML)
	f.close()
		
	# load data
	assembly_jaccard = defaultdict(lambda: defaultdict(float))
	assembly_pollutants = defaultdict(str)
	assembly_pvals = defaultdict(str)

	with open("best_assembled_maps.tsv", "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			tdisease = ldata[0]
			tfile = ldata[1]
			tjaccard = float(ldata[2])
			tpollutant = ldata[4]
			assembly_jaccard[tdisease] = tjaccard
			assembly_pollutants[tdisease] = tpollutant
			assembly_pvals[tdisease] = ldata[5]
			#assembly_data[tdisease]['pollutant'] = tpollutant
	infile.close()

	tcnt = 1
	rowcnt = 0
	tHTML = "<html>"
	tHTML += "<table border=0>"
	ndisease_columns = disease_columns.copy() + ["STROKE_MORT"]
	
	# flip around ties - cad, and stroke
	slist = []
	for tdisease in sorted(assembly_jaccard, key=assembly_jaccard.get, reverse=True):
		slist.append(tdisease)
		#if (tdisease != "Renal Disease") and (tdisease != "Stroke Mortality"):
		#	slist.append(tdisease)
		#if (tdisease == "Coronary Heart Disease"):
		#	slist.append("Stroke")
		#if (tdisease == "Stroke"):
		#	slist.append("Coronary Heart Disease")

	for tdisease in slist:
		ttoken = tdisease
		tlabel = tdisease
		pollutant_list = assembly_pollutants[tdisease].replace('!!', ', ')
		if tdisease == "STROKE_MORT":
			ttoken = "stroke_data."
			tlabel = "Stroke Mortality"
		if tdisease != "STROKE_BELT":
			format_pval = formatPVal(assembly_pvals[tdisease])
			tpoll_list = pollutant_list.title()
			tpoll_list = tpoll_list.replace('Diesel Pm', 'Diesel PM')
			tpoll_list = tpoll_list.replace('Rmp Facility Proximity', 'RMP Facility Proximity')			

			tHTML += "<tr><td style=\"width: 70px;\" valign=top><font style=\"font-family: arial; font-size: 20px;\"><center><br>" + str(tcnt) + '. ' + tlabel + "</font><font style=\"font-family: arial; font-size: 18px;\"><br>" + "<br><i>J</i> = " + str(assembly_jaccard[tdisease]) + "<br><i>p</i> = " + format_pval + "</center></font></td>"
			tHTML += "<td style=\"width: 250px; text-align: center;\" valign=top><img src=\"lancet.canonical." + ttoken + ".0.7.binarymap.png\" style=\"width: 250px;\"></center></td>" 
			tHTML += "<td style=\"width: 250px; text-align: center;\" valign=top><img src=\"lancet_final_" + tdisease + "_assembled.map.png\" style=\"width: 250px;\">" + "<center><font style=\"font-family: arial; font-size: 15px;\"><br>" + tpoll_list + "</font></center></td>"
			tHTML += "<td style=\"width: 350px; text-align: center;\" valign=top><img src=\"networks/" + tdisease + "_roc.0.7.merge.network.jpg\" style=\"width: 300px;\"></td>"

			tHTML += "</tr>" 

			tcnt += 1
	
	tHTML += "</table></html>"
	
	f = open("lancet_figure6_compare.html", "w")
	f.write(tHTML)
	f.close()

def SupRegressionResults():
	
	tcnt = 1
	rowcnt = 0

	tHTML = "<html>"
	tHTML += "\t<table border=0>"
	ndisease_columns = disease_columns.copy() + ["STROKE_MORT"]
	
	bpath = "d:\\apeer\\networks\\"
	for tdisease in sorted(ndisease_columns):
		ttoken = tdisease
		tlabel = tdisease
		if tdisease == "STROKE_MORT":
			ttoken = "stroke_data."
			tlabel = "Stroke Mortality"
		roc_file2 = bpath + tdisease + "_roc.0.7.merge.network.jpg.xgboost.jpg"
		roc_file1 = bpath + tdisease + "_roc.0.7.merge.network.jpg.logistic.jpg"
		tHTML += "\t<tr>"
		tHTML += "<td><center><span style=\"font-family: arial; font-size: 24px;\">" + tlabel + "</span></center></td>"
		tHTML += "<td><center><span style=\"font-family: arial; font-size: 16px;\">Elastic Net Regression</span></center><center><img src=\"" + roc_file1 + "\" style=\"width: 400px;\"></center></td>"
		tHTML += "<td><center><span style=\"font-family: arial; font-size: 16px;\">Random Forest Regression</span></center><center><img src=\"" + roc_file2 + "\" style=\"width: 400px;\"></center></td>"
		tHTML += "</tr>\n"

	tHTML += "\t</table>\n</html>"
	
	f = open("supp_regressions.html", "w")
	f.write(tHTML)
	f.close()

def CompareRegressionResults():

	tcnt = 1
	rowcnt = 0

	tHTML = "<html>"
	tHTML += "\t<table border=0>"
	ndisease_columns = disease_columns.copy() + ["STROKE_MORT"]
	
	bpath = "d:\\apeer\\networks\\"
	for tdisease in sorted(ndisease_columns):
		ttoken = tdisease
		tlabel = tdisease
		if tdisease == "STROKE_MORT":
			ttoken = "stroke_data."
			tlabel = "Stroke Mortality"

		# airtox_merge_stats_lancet.canonical.Asthma.0.7.binarymap.tsv.txt.bar.jpg
		roc_file0 = bpath + "airtox_merge_stats_lancet.canonical." + tdisease + ".0.7.binarymap.tsv.txt.bar.jpg"
		# airtox_merge_stats_lancet.canonical.stroke_data..0.7.binarymap.tsv.txt.bar.jpg
		roc_file1 = bpath + tdisease + "_roc.0.7.merge.network.jpg.beta.logistic.jpg"
		# STROKE_BELT_roc.0.7.merge.network.jpg.Pollution Model_model.xgboost.pdf.importance.jpg
		roc_file2 = bpath + tdisease + "_roc.0.7.merge.network.jpg.Pollution Model_model.xgboost.pdf.importance.jpg"
		
		if tdisease == "STROKE_MORT":
			roc_file0 = bpath + "airtox_merge_stats_lancet.canonical.stroke_data..0.7.binarymap.tsv.txt.bar.jpg"

		tHTML += "\t<tr>"
		tHTML += "<td><center><span style=\"font-family: arial; font-size: 24px;\">" + tlabel + "</span></center></td>"
		tHTML += "<td><center><br><span style=\"font-family: arial; font-size: 14px; margin-left: 85px;\">aPEER </span></center><center><img src=\"" + roc_file0 + "\" style=\"width: 250px;\"></center></td>"
		tHTML += "<td><center><br><span style=\"font-family: arial; font-size: 14px; margin-left: 85px;\">Elastic Net</span></center><center><img src=\"" + roc_file1 + "\" style=\"width: 250px;\"></center></td>"
		tHTML += "<td><center><br><span style=\"font-family: arial; font-size: 14px; margin-left: 85px;\">Random Forest</span></center><center><img src=\"" + roc_file2 + "\" style=\"width: 250px;\"></center></td>"
		tHTML += "</tr>\n"

	tHTML += "\t</table>\n</html>"
	
	f = open("supp_compare_regressions.html", "w")
	f.write(tHTML)
	f.close()
	
def Figure4Networks():

	#05/28/2024  12:53 PM         3,102,738 airtox_merge_network_lancet.canonical.Renal Disease.0.7.binarymap.tsv.jpg
	#05/28/2024  12:53 PM         3,087,584 airtox_merge_network_lancet.canonical.Stroke.0.7.binarymap.tsv.jpg
	#05/28/2024  12:53 PM         2,979,770 airtox_merge_network_lancet.canonical.stroke_belt.binarymap.tsv.jpg
	#05/28/2024  12:53 PM         2,196,920 airtox_merge_network_lancet.canonical.stroke_data..0.7.binarymap.tsv.jpg
	
	#05/28/2024  12:53 PM            16,008 airtox_merge_stats_lancet.canonical.Renal Disease.0.7.binarymap.tsv.txt
	#05/28/2024  12:53 PM            16,643 airtox_merge_stats_lancet.canonical.Stroke.0.7.binarymap.tsv.txt
	#05/28/2024  12:53 PM            16,191 airtox_merge_stats_lancet.canonical.stroke_belt.binarymap.tsv.txt
	#05/28/2024  12:53 PM            12,520 airtox_merge_stats_lancet.canonical.stroke_data..0.7.binarymap.tsv.txt

	# Get Top values - column_list, bar_data = stroke_belt_lib_lancet.GetTopNetworkPollutants(tfile2, top_poll)
	# Plot in barchart - TopValsBarChart(top_sorted, iFile, xlabel)

	tcnt = 1
	rowcnt = 0
	topcnt = 20
	tHTML = "<html>"
	tHTML += "\t<table border=0>"
	ndisease_columns = disease_columns.copy() + ["STROKE_MORT"]
	
	bpath = "d:\\apeer\\networks\\"
	for tdisease in sorted(ndisease_columns):
		ttoken = tdisease
		tlabel = tdisease
		if tdisease == "STROKE_MORT":
			ttoken = "stroke_data."
			tlabel = "Stroke Mortality"
		if tdisease == "STROKE_BELT":
			ttoken = "stroke_belt"
			tlabel = "Stroke Belt"
		
		# load hub data
		network_graph = bpath + "airtox_merge_network_lancet.canonical." + ttoken + ".0.7.binarymap.tsv.jpg"
		tfile2 = bpath + 'airtox_merge_stats_lancet.canonical.' + ttoken + '.0.7.binarymap.tsv.txt'
		bar_file = tfile2 + '.bar.jpg'
		#column_list, bar_data = stroke_belt_lib_lancet.GetTopNetworkPollutants(tfile2, top_poll)

		# airtox_merge_stats_lancet.canonical.Arthritis.0.7.binarymap.tsv.txt
		tcnt = 0
		# ACETALDEHYDE	176
		ttable = []
		chem_list = []
		print("Loading Network File: " + tfile2)
		with open(tfile2, "r") as infile:
			for line in infile:
				line = line.strip()
				ldata = line.split("\t")
				tid = ldata[2]
				thub = ldata[1]
				if thub == "Hub":
					tid = tid + '*'
				if tcnt < topcnt:
					chem_list.append(tid)
					trow = [tid, ldata[3]]
					ttable.append(trow)
				tcnt += 1
		infile.close()
		#print("Network nodes found: " + str(chem_list))

		bar_data = pd.DataFrame(ttable, columns=['columns', 'importances'])
		#epa_data.set_index("fips", inplace=True)
		bar_data[['importances']] = bar_data[['importances']].apply(pd.to_numeric)
		
		# reverse pandas dataframe order
		bar_data = bar_data.iloc[::-1]

		# graph
		TopValsBarChart(bar_data, bar_file, "Degree (# Connections)")
		
		tHTML += "\t<tr>"
		tHTML += "<td><center><span style=\"font-family: arial; font-size: 24px;\">" + tlabel + "</span></center></td><td><br><img src=\"" + network_graph + "\" style=\"width: 500px;\"></td><td><img src=\"" + bar_file + "\" style=\"width: 300px;\"></td>"
		tHTML += "</tr>\n"

	tHTML += "\t</table>\n</html>"
	
	f = open("lancet_fig4_regressions.html", "w")
	f.write(tHTML)
	f.close()


def Figure5Regression():
	
	tcnt = 1
	rowcnt = 0

	tHTML = "<html>"
	tHTML += "\t<table border=0>"
	ndisease_columns = disease_columns.copy() + ["STROKE_MORT"]
	
	bpath = "d:\\apeer\\networks\\"
	for tdisease in sorted(ndisease_columns):
		ttoken = tdisease
		tlabel = tdisease
		if tdisease == "STROKE_MORT":
			ttoken = "stroke_data."
			tlabel = "Stroke Mortality"
			
		roc_file1 = bpath + tdisease + "_roc.0.7.merge.network.jpg.logistic.jpg"
		#imp_file = bpath + tdisease + "_roc.0.7.merge.network.jpg.logistic.calibration.jpg"
		imp_file1 = bpath + tdisease + "_roc.0.7.merge.network.jpg.beta.logistic.jpg"
			
		roc_file2 = bpath + tdisease + "_roc.0.7.merge.network.jpg"
		imp_file2 = bpath + tdisease + "_roc.0.7.merge.network.jpg.Pollution Model_model.xgboost.pdf.importance.jpg"
		tHTML += "\t<tr>"
		tHTML += "<td><center><span style=\"font-family: arial; font-size: 24px;\">" + tlabel + "</span></center></td><td><img src=\"" + roc_file1 + "\" style=\"width: 450px;\"></td><td><img src=\"" + roc_file2 + "\" style=\"width: 450px;\"></td>"
		tHTML += "<td><td><img src=\"" + imp_file1 + "\" style=\"width: 300px;\"></td><td><img src=\"" + imp_file2 + "\" style=\"width: 300px;\"></td>"
		tHTML += "</tr>\n"

	tHTML += "\t</table>\n</html>"
	
	f = open("lancet_fig5_regressions.html", "w")
	f.write(tHTML)
	f.close()

def SupLogRegressionResults():
	
	tcnt = 1
	rowcnt = 0

	tHTML = "<html>"
	tHTML += "\t<table border=0>"
	ndisease_columns = disease_columns.copy() + ["STROKE_MORT"]
	
	bpath = "d:\\apeer\\networks\\"
	for tdisease in sorted(ndisease_columns):
		ttoken = tdisease
		tlabel = tdisease
		if tdisease == "STROKE_MORT":
			ttoken = "stroke_data."
			tlabel = "Stroke Mortality"
		roc_file = bpath + tdisease + "_roc.0.7.merge.network.jpg.logistic.jpg"
		#imp_file = bpath + tdisease + "_roc.0.7.merge.network.jpg.logistic.calibration.jpg"
		imp_file = bpath + tdisease + "_roc.0.7.merge.network.jpg.beta.logistic.jpg"
		tHTML += "\t<tr><td><center><span style=\"font-family: arial; font-size: 24px;\">" + tlabel + "</span></center></td><td><br><img src=\"" + roc_file + "\" style=\"width: 400px;\"></td><td><img src=\"" + imp_file + "\" style=\"width: 350px;\"></td></tr>\n"

	tHTML += "\t</table>\n</html>"
	
	f = open("supp_logistic_regressions.html", "w")
	f.write(tHTML)
	f.close()

def GetCorrelationMap(chemical_data, json_data_county):

	# get top jaccard index values
	# get data
	chemical_data = chemical_data.replace(np.nan, 0)
	norm_chemical_data = normalize(chemical_data, axis=0)
	norm_chemical_data = pd.DataFrame(norm_chemical_data, columns=chemical_data.columns, index=chemical_data.index)

	# get top pollutants
	top_pollutants = GetTopPollutantsByJaccard()
	
	# create assembled maps
	countycnt = 3141
	stroke_cutoffs = [0.7]
	xpath = "d:\\apeer\\networks"
	disease_list = ["COPD", "Hypertension", "Asthma", "Diabetes", "Depression", "Arthritis", "Cancer", "Coronary Heart Disease", "Renal Disease", "Obesity", "Stroke"]
	ndisease_list = disease_list.copy() + ["STROKE_MORT", "STROKE_BELT"]
	param_data = ""
	#ndisease_list = ["STROKE_MORT"]
		
	tcutoff = 0.7
	for tdisease in ndisease_list:

		file_list = defaultdict(str)
		jaccard_list = defaultdict(str)
		pval_list = defaultdict(str)

		poll_list = []
		top_list = []
		tstr = top_pollutants[tdisease]
		disease_list = tstr.split('!!')
		for x in range(0, len(disease_list)):
			cdata = disease_list[x]
			disease_pair = cdata.split('|')
			if len(disease_pair) > 1:
				if disease_pair[0] not in poll_list:
					poll_list.append(disease_pair[0])
				if disease_pair[1] not in poll_list:
					poll_list.append(disease_pair[1])
						
		top_list = poll_list[0:10].copy()

		# Load Reference map data
		rfile = "lancet.canonical." + tdisease + "." + str(tcutoff) + ".binarymap.tsv"
		if tdisease == "STROKE_MORT":
			rfile = "lancet.canonical.stroke_data." + "." + str(tcutoff) + ".binarymap.tsv"
		if tdisease == "STROKE_BELT":
			rfile = "lancet.canonical.stroke_belt.binarymap.tsv"

		# Transform clustered data
		rawdata2 = LoadClusterData(rfile)
		refbelt = defaultdict(int)
		for tid in rawdata2:
			if rawdata2[tid] == "1":
				refbelt[tid] = 1

		print(tdisease + "\t" + str(top_list))
		
		for p1 in range(0, 10):
			for p2 in range(p1 + 1, 10):

				# get pairwise data
				pv1 = top_list[p1]
				pv2 = top_list[p2]
				cdata = norm_chemical_data[[pv1, pv2]]
				
				tjaccard = 0
				for nclust in range(2, 6):
				
					tfile = "d:\\apeer\\maps\\lancet_pairwise_fig2_" + pv1 + "_" + pv2 + "_clust" + str(nclust)
					tfile = tfile.replace(' ', '')
					#print(tfile + ".map.png")

					# create clusters
					ShowTractClustersOnMap(cdata, nclust, tfile, "county", json_data_county, showmap=False, forceBinary=False, colorCluster=0)

					# Load Clustered Map
					ofile = tfile + ".cluster.tsv"
					rawdata = LoadClusterData(ofile)
					tjaccard, tpval = CalcJaccard(rawdata, refbelt, countycnt)
					
					tid1 = pv1 + '-' + pv2
					tid2 = pv2 + '-' + pv1
					if (tid1 not in jaccard_list) and (tid2 not in jaccard_list):
						jaccard_list[tid1] = str(tjaccard)
						jaccard_list[tid2] = str(tjaccard)
						file_list[tid1] = tfile
						file_list[tid2] = tfile

					if (float(jaccard_list[tid1]) < tjaccard) and (float(jaccard_list[tid2]) < tjaccard):
						jaccard_list[tid1] = str(tjaccard)
						jaccard_list[tid2] = str(tjaccard)
						file_list[tid1] = tfile
						file_list[tid2] = tfile
						
		# Pairwise Pollution Maps
		tHTML = "<html><table style=\"border: 1px solid gray;\">"
		trow = "<tr><td></td>\n"
		for pairx in range(0, len(top_list)):
			tlabel = top_list[pairx]
			tlabel = tlabel.replace(" ", "\n<br>")
			tlabel = tlabel.replace("Air\n<br>", "Air ")
			tlabel = tlabel.title()
			tlabel = tlabel.replace('Pm', 'PM')
			cdata = tlabel.split('(')
			tlabel = cdata[0]
			trow = trow + "<td style=\"text-align: center; font-size: 10px; font-family: arial; width: 70px;\">" + tlabel + "</td>\n"
		trow = trow + "</tr>\n"
		tHTML = tHTML + trow

		for pairx in range(0, len(top_list)):
			tlabel = top_list[pairx]
			tlabel = tlabel.replace(" ", "\n<br>")
			tlabel = tlabel.replace("Air\n<br>", "Air ")
			tlabel = tlabel.title()
			tlabel = tlabel.replace('Pm', 'PM')
			cdata = tlabel.split('(')
			tlabel = cdata[0]
			trow = "<tr style=\"height: 50px\">\n" + "<td style=\"text-align: center; font-size: 10px; font-family: arial;\">" + tlabel + "</td>\n"

			for pairy in range(0, len(top_list)):
				tjval = 0
				tid1 = top_list[pairx] + '-' + top_list[pairy]
				tid2 = top_list[pairy] + '-' + top_list[pairx]
				if tid1 in jaccard_list:
					tjval = float(jaccard_list[tid1])
				
				fval = "--"
				if tjval > 0:
					fval = str(tjval)[0:6]
				back_color = GetJaccardColorScale(tjval)
				
				if tid1 == tid2:
					fval = "&nbsp;"
					back_color = "#000000"
				
				tcell = "<td style=\"background-color: " + back_color + "; text-align: center; font-size: 12px; font-family: arial; color: white;\">" + fval + "</td>\n"
				if pairy >= (pairx + 1):

					# create clusters
					pv1 = top_list[pairx]
					pv2 = top_list[pairy]
					cdata = norm_chemical_data[[pv1, pv2]]
					tfile = file_list[tid1] + ".map.png"
					file_data = file_list[tid1].split('_clust')
					nclust = int(file_data[1])
					showmap = False
					if os.path.isfile(tfile + ".map.png") == False:
						showmap = True
						print(tdisease + "\t" + "Missing Map: " + tfile)
					#showmap = True
					ShowTractClustersOnMap(cdata, nclust, tfile, "county", json_data_county, showmap, forceBinary=False)
					tcell = "<td style=\"text-align: center;\"><img src=\"" + tfile + ".map.png\" style=\"width: 68px;\"></td>\n"

				trow = trow + tcell
			trow = trow + "</tr>\n"

			tHTML = tHTML + trow

		tHTML = tHTML + "</table></html>"
		
		# Load Hub Data
		f = open("lancet_merge_" + tdisease + "_figure2.html", "w")
		f.write(tHTML)
		f.close()

		
def GetCorrelationMapOld(baseid, ofile):

	# Maps:
	# lancet_epa_pair.Wastewaterdischarge.Ozone.county_cluster.5.map.png

	# Data:
	# load all pairwise Jaccard values from lancet_figure2.tsv
	# lancet.canonical.Arthritis.0.6.binarymap.tsv    Lead Paint      Diesel Particulate Matter       3       0.2615459406903257      0.883015426730431
	# data1 = lancet.canonical.life_expect.0.2.binarymap.tsv
	# data2 = lancet.canonical.stroke_data..0.4.binarymap.tsv
	
	life_expect_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
	stroke_data_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
	
	#ifile = "lancet_figure2.tsv"
	#ifile = "lancet_merge_figure3.tsv"
	ifile = "lancet_merge_0.7_figure3.tsv"
	pval_cutoff = GetPValCutoff(ifile)
	
	tminval = 100
	tmaxval = 0
	with open(ifile, "r") as infile:
	
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			tid = ldata[0]
			id1 = ldata[1]
			id2 = ldata[2]
			tclust = ldata[3]
			jval = float(ldata[4])
			pval = float(ldata[5])
			#print(tid + "\t" + nid + "\t" + str(jval))

			if tid == baseid:
				if pval <= pval_cutoff:
					life_expect_matrix[id1][id2][tclust] = jval
					life_expect_matrix[id2][id1][tclust] = jval
					if (jval < tminval):
						tminval = jval
					if (jval > tmaxval):
						tmaxval = jval

	infile.close()
	
	# get a list of the top 10 Jaccard Values
	
	
	print("Minmax\t" + baseid + "\t" + str(tminval) + "\t" + str(tmaxval))

	# Pairwise Pollution Maps
	tHTML = "<html><table style=\"border: 1px solid gray;\">"
	trow = "<tr><td></td>\n"
	for pairx in range(0, len(pollution_columns)):
		tlabel = pollution_columns[pairx]
		tlabel = tlabel.replace(" ", "\n<br>")
		tlabel = tlabel.replace("Air\n<br>", "Air ")
		trow = trow + "<td style=\"text-align: center; font-size: 10px; font-family: arial; width: 55px;\">" + tlabel + "</td>\n"
	trow = trow + "</tr>\n"
	tHTML = tHTML + trow
	for pairx in range(0, len(pollution_columns)):
		tlabel = pollution_columns[pairx]
		tlabel = tlabel.replace(" ", "\n<br>")
		tlabel = tlabel.replace("Air\n<br>", "Air ")
		trow = "<tr>\n" + "<td style=\"text-align: center; font-size: 10px; font-family: arial;\">" + tlabel + "</td>\n"
		for pairy in range(0, len(pollution_columns)):
			#if pairy <= (pairx + 1):
			# get maximum j from cluster number
			tjval = 0
			tclustval = 2
			tid1 = pollution_columns[pairx]
			tid2 = pollution_columns[pairy]
			for tclust in life_expect_matrix[tid1][tid2]:
				if life_expect_matrix[tid1][tid2][tclust] > tjval:
					tjval = life_expect_matrix[tid1][tid2][tclust]
					tclustval = tclust

			fval = "--"
			if tjval > 0:
				fval = str(tjval)[0:6]
			back_color = GetJaccardColorScale(tjval)
			
			if tid1 == tid2:
				fval = "&nbsp;"
				back_color = "#000000"
			
			tcell = "<td style=\"background-color: " + back_color + "; text-align: center; font-size: 10px; font-family: arial; color: white;\">" + fval + "</td>\n"
			if pairy >= (pairx + 1):
				#for nclust in range(2, 6):
				nclust = tclustval
				tfile = "lancet_epa_pair." + pollution_columns[pairx] + "." + pollution_columns[pairy] + ".county_cluster." + str(nclust) + ".map.png"
				tfile = tfile.replace(" ", "")
				tcell = "<td style=\"text-align: center;\"><img src=\"" + tfile + "\" style=\"width: 55px;\"></td>\n"
				#print("Pairwise Pollution File: " + tfile)
				#tdata = merge_data_counties[[stroke_belt_lib_lancet.pollution_columns[pairx], stroke_belt_lib_lancet.pollution_columns[pairy]]]
				#stroke_belt_lib_lancet.ShowTractClustersOnMap(tdata, nclust, tfile, "county", json_data_county, showmap=True)
			trow = trow + tcell
		trow = trow + "</tr>\n"
		tHTML = tHTML + trow
	tHTML = tHTML + "</table></html>"
	
	f = open(ofile, "w")
	f.write(tHTML)
	f.close()

def GetMaxJaccard():

	idlist = defaultdict(lambda: defaultdict(int))
	life_expect_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
	out_table = defaultdict(lambda: defaultdict(str))

	ifile = "lancet_figure2.tsv"
	pval_cutoff = GetPValCutoff(ifile)

	# get p-value threshold
	n = 0
	jvals = defaultdict(float)
	pvals = defaultdict(float)
	
	with open(ifile, "r") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			colcnt = len(ldata)
			tid = ldata[0] + "\t" + ldata[1] + "\t" + ldata[2] + "\t" + ldata[3]
			jvals[tid] = float(ldata[4])
			pvals[tid] = float(ldata[5])
	infile.close()
	
	for mainid in jvals:
		ldata = mainid.split("\t")
		tid = ldata[0]
		id1 = ldata[1]
		id2 = ldata[2]
		tclust = ldata[3]
		jval = float(jvals[mainid])
		pval = float(pvals[mainid])
		if pval <= pval_cutoff:
			idlist[tid][id1] += 1
			idlist[tid][id2] += 1
			life_expect_matrix[tid][id1][id1 + "\t" + id2 + "\t" + tclust] = float(jval)
			life_expect_matrix[tid][id2][id2 + "\t" + id1 + "\t" + tclust] = float(jval)
	
	# get median values
	#for mainid in idlist:
	#	for tid in idlist[mainid]:
	#		for tfield in life_expect_matrix[mainid][tid]:
	#			tlist = life_expect_matrix[mainid][tid][tfield]
	#			tlist = tlist[:-1]
	#			tlist = tlist.split(',')
	#			tlist = [float(i) for i in tlist]
	#			median_value = median(tlist)
	#		if (mainid.find('0.6') > -1) or (mainid.find('stroke_belt') > -1) or ((mainid.find('life_expect') > -1) and (mainid.find('0.3') > -1)):
	#			print(mainid + "\t" + tid + "\t" + str(idlist[mainid][tid]) + "\t" + str(median_value))
		
	# get average jaccard
	for mainid in idlist:
		mdata = mainid.split('.')
		mid = mdata[2]
		for tid in idlist[mainid]:
			tjaccard = 0
			for tfield in life_expect_matrix[mainid][tid]:
				tjaccard += life_expect_matrix[mainid][tid][tfield]
			taverage = tjaccard / idlist[mainid][tid]
			if (mainid.find('0.6') > -1) or (mainid.find('stroke_belt') > -1) or ((mainid.find('life_expect') > -1) and (mainid.find('0.3') > -1)):
				out_table[mid][tid] = str(idlist[mainid][tid])
				#print(mid + "\t" + tid + "\t" + str(idlist[mainid][tid]) + "\t" + str(taverage))
				
	# print out table
	collist = defaultdict(str)
	for trow in out_table:
		for tcol in out_table[trow]:
			collist[tcol] = ""
	
	outdata = "Disease\t"
	for tcol in sorted(collist):
		outdata += tcol + "\t"
	outdata = outdata[:-1] + "\n"
	
	for trow in sorted(out_table):
		outdata = outdata + trow + "\t"
		for tcol in sorted(out_table[trow]):
			outdata = outdata + str(out_table[trow][tcol]) + "\t"
		outdata = outdata[:-1] + "\n"
	
	f = open("ejscreen_numjaccard.tsv", "w")
	f.write(outdata)
	f.close()

def LoadKFoldFile(tfile):

	tauc = 0
	taccuracy = 0
	if (tfile != "na"):
		with open(tfile, "r") as infile:
			for line in infile:
				line = line.strip()
				if line.find("test-auc-mean Mean:") > -1:
					tdata = line.split(':')
					tauc = float(tdata[1])
					tauc = round(tauc, 4)
				if line.find("Average Accuracy") > -1:
					tdata = line.split(':')
					taccuracy = float(tdata[1])
					taccuracy = round(taccuracy, 4)
		infile.close()
	
	return tauc, taccuracy
	
def GetSensitivityTable(tmode):

	# sensitivity analysis for XGBoost
	#03/24/2024  11:57 AM             1,955 Stroke_roc.0.9.jpg.AirToxScreen Model_model.xgboost.pdf.kfold.csv
	#03/24/2024  11:57 AM             1,956 Stroke_roc.0.9.jpg.Disease Model_model.xgboost.pdf.kfold.csv
	#03/24/2024  11:57 AM             1,954 Stroke_roc.0.9.jpg.EJSCREEN Model_model.xgboost.pdf.kfold.csv
	#03/24/2024  11:57 AM             1,956 Stroke_roc.0.9.jpg.SDOH Model_model.xgboost.pdf.kfold.csv	

	# merge threshholded Stroke, Life-expectancy data
	#chemical_data = LoadChemicalData("county")
	#finalcounties, finaltracts = GetMainLists()
	#merge_data_tract = merge_data_tract[merge_data_tract.index.isin(finaltracts)]
	#chemical_data = chemical_data[chemical_data.index.isin(finalcounties)]
	
	# get master list of chemical data
	#chemlist = list(chemical_data.columns.values)
	def GetColorScale(tlist):

		color_scale = ["#008000", "#75a375", "#dd7777", "#ff0000"]

		sort_list = defaultdict(str)
		tindex = 0
		for titem in sorted(tlist):
			sort_list[titem] = tindex
			tindex += 1

		color_list = defaultdict(str)
		for titem in sort_list:
			tindex = sort_list[titem]
			tcolor = color_scale[tindex]
			color_list[str(titem)] = tcolor

		return color_list
		
	def PrintOutVals(tline, val_list):
	
		getcolors = GetColorScale(val_list)
		for titem in val_list:
			#tline += str(tauc) + "\t" + str(taccuracy) + "\t"
			tline += "<td bgcolor=\"" + getcolors[str(titem)] + "\">" + str(titem) + "</td>"
		tline += "</tr>"
		tline = tline.strip()
		tline = tline + "\n"
		
		return tline
		
	def GetNetworkFile(tdisease, tpct):
	
		ofile = "na"
		
		tpattern = tdisease + "_roc." + str(tpct) + ".network.jpg.*..._model.xgboost.pdf.kfold.csv"
		if tpct == 0:
			tpattern = tdisease + "_roc" + ".network.jpg.*..._model.xgboost.pdf.kfold.csv"
		
		file_list = glob.glob(tpattern)
		print(file_list)
		if len(file_list) > 0:
			ofile = file_list[0]
		
		return ofile
				
	# Model list
	tconcat = '.jpg.'
	if tmode == "network":
		tconcat = '.network.jpg.'
	model_list = ["AirToxScreen Model", "Disease Model", "EJSCREEN Model", "SDOH Model"]
	life_cutoffs = [0.2, 0.3, 0.4, 0.5]
	stroke_cutoffs = [0.4, 0.5, 0.6, 0.7]
	cutoff_list = [0.6, 0.7, 0.8, 0.9]

	# initialize table
	tline = "<html><table style=\"font-family: arial; text-align: center; width: 1000px;\"><tr><td>Disease</td><td>Percentile</td>"
	for tmodel in sorted(model_list):
		tline += "<td>" + tmodel + "</td>"
	tline = tline.strip() + "</tr>\n"

	for tcutoff in life_cutoffs:
		fcutoff = FormatPercentile(str(tcutoff))
		tline += "<tr><td>Life Expectancy</td><td>" + fcutoff + "</td>\n"

		val_list = []
		for tmodel in sorted(model_list):
			tfile2 = "life_expect_roc." + str(tcutoff) + tconcat + tmodel + "_model.xgboost.pdf.kfold.csv"
			if (tmode == "network") and (tmodel == "AirToxScreen Model"):
				tfile2 = GetNetworkFile("life_expect", tcutoff)
				print(str(tcutoff) + "\t" + tmodel + "\t" + tfile2)
			tauc, taccuracy = LoadKFoldFile(tfile2)
			val_list.append(tauc)
			
		tline = PrintOutVals(tline, val_list)
			
	for tcutoff in stroke_cutoffs:
		fcutoff = FormatPercentile(str(tcutoff))
		tline += "<tr><td>Stroke Mortality</td><td>" + fcutoff + "</td>\n"
		
		val_list = []
		for tmodel in sorted(model_list):
			tfile2 = "stroke_mort_roc." + str(tcutoff) + tconcat + tmodel + "_model.xgboost.pdf.kfold.csv"
			if (tmode == "network") and (tmodel == "AirToxScreen Model"):
				tfile2 = GetNetworkFile("stroke_mort", tcutoff)
			tauc, taccuracy = LoadKFoldFile(tfile2)
			val_list.append(tauc)

		tline = PrintOutVals(tline, val_list)

	disease_list = ["COPD", "Hypertension", "Asthma", "Diabetes", "Depression", "Arthritis", "Cancer", "Coronary Heart Disease", "Renal Disease", "Obesity", "Stroke"]
	for tdisease in disease_list:
		for tcutoff in cutoff_list:
			fcutoff = FormatPercentile(str(tcutoff))
			tline += "<tr><td>" + tdisease + "</td><td>" + fcutoff + "</td>\n"
			val_list = []
			for tmodel in sorted(model_list):
				tfile2 = tdisease + "_roc." + str(tcutoff) + tconcat + tmodel + "_model.xgboost.pdf.kfold.csv"
				if (tmode == "network") and (tmodel == "AirToxScreen Model"):
					tfile2 = GetNetworkFile(tdisease, tcutoff)
				tauc, taccuracy = LoadKFoldFile(tfile2)
				val_list.append(tauc)
				
			tline = PrintOutVals(tline, val_list)

	tline += "<tr><td>Stroke Belt</td><td>" + "NA" + "</td>\n"
	val_list = []
	for tmodel in sorted(model_list):
		tfile2 = "stroke_belt_roc" + tconcat + tmodel + "_model.xgboost.pdf.kfold.csv"
		if (tmode == "network") and (tmodel == "AirToxScreen Model"):
			tfile2 = GetNetworkFile("stroke_belt", 0)
		tauc, taccuracy = LoadKFoldFile(tfile2)
		val_list.append(tauc)

	tline = PrintOutVals(tline, val_list)
	
	tline += "</table></html>"

	f = open("supplemental_sensitivity" + tconcat + "html", "w")
	f.write(tline)
	f.close()

	plt.close("all")
	#fig.clf()	
	#fig = plt.figure(figsize=(5, 4))
	fig, ax = plt.subplots(figsize=(5, 4))
	plt.axis("on")
	ax.xaxis.set_visible(True)
	ax.yaxis.set_visible(True)
	ax.set_facecolor('xkcd:white')

	# order by AUC
	text_labels = defaultdict(float)
	text_labels[str(auc1)] = 0
	text_labels[str(auc2)] = 1
	text_labels[str(auc3)] = 2
	text_labels[str(auc4)] = 3
	text_labels[str(auc5)] = 4
	
	ypos = 0.25
	colorcnt = 3
	tfont = 11
	tleft = 0.4
	for tkey in sorted(text_labels, reverse=True):
		tauc_val = round(float(tkey), 2)
		tindex = text_labels[tkey]
		plt.text(tleft, ypos, "AUC: " + str(tauc_val) + " (" + tlabels[tindex] + ")", fontsize = tfont, color=tcolors[tindex])
		ypos = ypos - 0.05
		colorcnt = colorcnt - 1		

	plt.plot(fpr1, tpr1, color=tcolors[0])
	plt.plot(fpr2, tpr2, color=tcolors[1])
	plt.plot(fpr3, tpr3, color=tcolors[2])
	plt.plot(fpr4, tpr4, color=tcolors[3])
	plt.plot(fpr5, tpr5, color=tcolors[4])

	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.margins(x=0, y=0)
	fig.tight_layout()
	plt.savefig(tfile, dpi=300)


def CalibrationPlot(y_test, prob_pos, cal_file, tlabels):

	tcolors = ["blue", "green", "red", "purple", "orange", "yellow"]
	#tlabels = ["SDOH Model", "EJSCREEN Model", "Prevention Model", "Top 5 Pollutants", "Top 10 Pollutants"]

	#fig = plt.figure(figsize=(5, 4))
	fig, ax = plt.subplots(figsize=(5, 4))
	plt.rcParams.update({'font.size': 14})
	plt.axis("on")
	ax.xaxis.set_visible(True)
	ax.yaxis.set_visible(True)
	ax.set_facecolor('xkcd:white')

	#Plot the Perfectly Calibrated by Adding the 45-degree line to the plot
	plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')

	# Set the title and axis labels for the plot
	tdata = cal_file.split('_')
	ttitle = tdata[0]
	if ttitle == "stroke_belt":
		ttitle = "Stroke Belt"
	if ttitle == "life_expect":
		ttitle = "Life Expectancy"
	#plt.title(ttitle)
	plt.xlabel('Predicted Probability')
	plt.ylabel('True Probability')

	# Brier Score
	b_score = []
	text_labels = defaultdict(int)
	for x in range(0, len(y_test)):
		tval = brier_score_loss(y_test[x], prob_pos[x])
		if tval in b_score:
			nval = str(tval)[0:6] + "1"
			tval = float(nval)
		b_score.append(tval)
		print(str(x) + "\t" + str(tval))

		# True and Predicted Probabilities
		true_pos, pred_pos = calibration_curve(y_test[x], prob_pos[x], n_bins=10)
		plt.plot(pred_pos, true_pos, color=tcolors[x])
		text_labels[str(tval)] = x

	print(str(text_labels))

	ypos = 0.9
	colorcnt = 3
	tfont = 11
	tleft = 0.05
	for tkey in sorted(text_labels, reverse=True):
		tauc_val = round(float(tkey), 2)
		tindex = text_labels[tkey]
		plt.text(tleft, ypos, "Brier: " + str(tauc_val) + " (" + tlabels[tindex] + ")", fontsize = 9, color=tcolors[tindex])
		ypos = ypos - 0.05
		colorcnt = colorcnt - 1		

	plt.margins(x=0, y=0)
	fig.tight_layout()
	plt.savefig(cal_file, dpi=300)

	plt.close("all")
	fig.clf()	


def MakeCalibrationFigure(tmode):

	# Stroke_roc.0.6.jpg.AirToxScreen Model_model.xgboost.pdf.calibration.jpg
	tHTML = "<table style=\"font-family: arial;\"><tr>\n"
	newlist = disease_columns.copy() + ["stroke_mort"]
	for x in range(0, len(newlist)):
		rval = (x + 1) % 3
		tdisease = newlist[x]
		disease_text = tdisease
		if (tdisease == "stroke_mort"):
			disease_text = "Stroke Mortality"
		if (tdisease == "life_expect"):
			disease_text = "Life Expectancy"
		tpercent = "0.7"
		if tdisease == "life_expect":
			tpercent = "0.3"
		tfile = tdisease + '_roc.' + tpercent + '.merge.network.jpg.calibration.jpg'
		if tmode == "logistic":
			tfile = tdisease + '_roc.' + tpercent + '.merge.network.jpg.logistic.calibration.jpg'
		tHTML += "\t" + '<td><img style="width: 350px;" src="' + tfile + '"></td>' + "\n"
		if (rval == 0):
			tHTML += "</tr>\n"
			tHTML += "<tr>\n"

	tHTML += "</tr></table></html>\n"

	ofile = "supplemental_calibration_curves.html"
	if tmode == "logistic":
		ofile = "supplemental_logistic_calibration_curves.html"
		
	f = open(ofile, "w")
	f.write(tHTML)
	f.close()

def MakeBaselineRegTable():

	ndisease_list = disease_columns.copy() + ["STROKE_MORT"]

	cols = ["Variable", "Univariate Beta", "Pvalue", "padj", "r2", "Multivariate Beta", "Pvalue", "padj", "r2"]
	tHTML = "<html>\n"

	for tdisease in ndisease_list:
		#ofile = tdisease + ".county.regression.tsv"
		ofile = tdisease + ".lancet_merge.regression.tsv"
		filedata = defaultdict(float)
		#print("Processing Regression for " + tdisease)
		#stroke_belt_lib_lancet.RunUnivariateRegression(merge_data_counties, tdisease, ofile)
		with open(ofile, "r") as infile:
			tline = infile.readline()
			tline = tline.strip()
			theader = tline.split("\t")
			for line in infile:
				tline = line.strip()
				tdata = tline.split("\t")
				tval = tdata[len(tdata) - 4]
				padj = tdata[len(tdata) - 2]
				if padj == '*':
					#print(line)
					filedata[tline] = float(tval)
		infile.close()
		
		# print out top values
		disease_label = tdisease
		if tdisease == "STROKE_MORT":
			disease_label = "Stroke Mortality"
		if tdisease == "LIFE_EXPECT":
			disease_label = "Life Expectancy"

		tcnt = 0
		tHTML += "<br><center><span style=\"font-size: 16; font-family: arial;\">" + disease_label + "</span></center>\n"
		tHTML += "<br>\n<br><table border=1><tr>\n"
		for tcol in cols:
			tHTML += "\t" + '<td style="text-align: center; font-family: arial;">' + tcol + '</td>' + "\n"
		tHTML += "</tr>\n"
		for tid in sorted(filedata, key=filedata.get, reverse=True):
			if tcnt < 20:
				ldata = tid.split("\t")
				tHTML += "<tr>\n"
				#tHTML += "\t<td style=\"text-align: center; font-family: arial;\">" + disease_label + "</td>\n"
				for x in range(0, len(ldata)):
					tHTML += "\t<td style=\"text-align: center; font-family: arial;\">" + ldata[x].title() + "</td>\n"
				tHTML += "</tr>\n"
			tcnt += 1
		tHTML += "</table>\n"

	tHTML += "</html>\n"

	f = open("supplemental_regression_table.html", "w")
	f.write(tHTML)
	f.close()

def GetRegModels(tmode, ofile):

	current_diseases = disease_columns.copy()
	if tmode == "logistic":
		current_diseases = ['stroke_mort', 'life_expect'] + current_diseases

	# Get XGBoost and basic predictions
	tHTML = "<html>\n"
	tHTML += "<table>\n"
	
	ttop = "80"
	for tdisease in sorted(current_diseases):
		# Hypertension_roc.0.6.network.jpg.binarymap.png.map.png
		troc_file = tdisease + "_roc.0.6.jpg"
		timportance_file = tdisease + "_roc.0.6.jpg.AirToxScreen Model_model.xgboost.pdf.importance.jpg"
		tmap = "lancet.canonical." + tdisease + ".0.6.binarymap.png"
		if tdisease == "life_expect":
			troc_file = tdisease + "_roc.0.3.jpg"
			timportance_file = tdisease + "_roc.0.3.jpg.AirToxScreen Model_model.xgboost.pdf.importance.jpg"
			tmap = "lancet.canonical." + tdisease + ".0.3.binarymap.png"
		if tdisease == "stroke_mort":
			tmap = "lancet.canonical.stroke_data..0.6.binarymap.png"
		if (tmode == "logistic"):
			troc_file = troc_file + ".logistic.jpg"
			timportance_file = tdisease + "_roc.0.6.jpg.beta.logistic.jpg"
			if tdisease == "life_expect":
				timportance_file = tdisease + "_roc.0.3.jpg.beta.logistic.jpg"
		
		# Set disease label
		disease_label = tdisease
		if tdisease == "stroke_mort":
			disease_label = "Stroke Mortality"
		if tdisease == "life_expect":
			disease_label = "Life Expectancy"			
		
		# lancet.canonical.Obesity.0.6.binarymap.png
		# Diabetes_roc.0.6.jpg.AirToxScreen Model_model.xgboost.pdf.importance.jpg
		tHTML = tHTML + "\t<tr>\n"
		tHTML = tHTML + "\t\t<td style=\"text-align: right; font-family: arial; font-size: 20px;\">" + disease_label + "</td>\n"
		tHTML = tHTML + "\t\t<td><div style=\"width: 300px; position: relative;\"><img style=\"width: 100%;\" src=\"" + troc_file + "\">"
		tHTML = tHTML + "\t\t\t<div style=\"z-index: 10; position: absolute; left: 170px; top: " + ttop + "px; width: 150px;\"><img style=\"width: 110px\" src=\"" + tmap + "\"></div>"
		tHTML = tHTML + "\t\t\t</div></td>\n"
		tHTML = tHTML + "\t\t<td><img style=\"width: 250px;\" src=\"" + timportance_file + "\"></td>\n"
		tHTML = tHTML + "\t</tr>\n"
	tHTML += "</table>\n</html>\n"
		
	#f = open("supplemental_xgboost_pred.html", "w")
	f = open(ofile, "w")
	f.write(tHTML)
	f.close()

def GetMergedRegModels(tmode, ofile):

	current_diseases = disease_columns.copy()
	if tmode == "logistic":
		current_diseases = ['stroke_mort'] + current_diseases

	# Get XGBoost and basic predictions
	tHTML = "<html>\n"
	tHTML += "<table>\n"
	
	ttop = "80"
	for tdisease in sorted(current_diseases):
		# Hypertension_roc.0.6.network.jpg.binarymap.png.map.png
		troc_file = tdisease + "_roc.0.7.jpg"
		timportance_file = tdisease + "_roc.0.7.jpg.AirToxScreen Model_model.xgboost.pdf.importance.jpg"
		tmap = "lancet.canonical." + tdisease + ".0.7.binarymap.png"
		if tdisease == "stroke_mort":
			tmap = "lancet.canonical.stroke_data..0.7.binarymap.png"
		if (tmode == "logistic"):
			troc_file = troc_file + ".logistic.jpg"
			timportance_file = tdisease + "_roc.0.7.jpg.beta.logistic.jpg"
		
		# Set disease label
		disease_label = tdisease
		if tdisease == "stroke_mort":
			disease_label = "Stroke Mortality"
		
		# lancet.canonical.Obesity.0.6.binarymap.png
		# Diabetes_roc.0.6.jpg.AirToxScreen Model_model.xgboost.pdf.importance.jpg
		tHTML = tHTML + "\t<tr>\n"
		tHTML = tHTML + "\t\t<td style=\"text-align: right; font-family: arial; font-size: 20px;\">" + disease_label + "</td>\n"
		tHTML = tHTML + "\t\t<td><div style=\"width: 300px; position: relative;\"><img style=\"width: 100%;\" src=\"" + troc_file + "\">"
		tHTML = tHTML + "\t\t\t<div style=\"z-index: 10; position: absolute; left: 170px; top: " + ttop + "px; width: 150px;\"><img style=\"width: 110px\" src=\"" + tmap + "\"></div>"
		tHTML = tHTML + "\t\t\t</div></td>\n"
		tHTML = tHTML + "\t\t<td><img style=\"width: 250px;\" src=\"" + timportance_file + "\"></td>\n"
		tHTML = tHTML + "\t</tr>\n"

	tHTML += "</table>\n</html>\n"
		
	#f = open("supplemental_xgboost_pred.html", "w")
	f = open(ofile, "w")
	f.write(tHTML)
	f.close()

def GetNetworkRegModels(tmode, ofile):

	ttop = "80"
	if tmode == "logistic":
		ttop = "60"

	# XGBoost and Network Connectedness
	tHTML = "<html>\n"
	tHTML += "<table>\n"
	
	current_diseases = disease_columns.copy()
	#if tmode == "logistic":
	current_diseases = ['stroke_mort', 'life_expect'] + current_diseases

	# get order by Jaccard
	jaccard_list = defaultdict(float)
	for tdisease in current_diseases:
		tquant = "0.6"
		if tdisease == "life_expect":
			tquant = "0.3"
		jval, pval = GetNetworkMapJaccard(tdisease, tquant)
		fjval = round(jval, 2)
		if len(str(fjval)) == 3:
			fjval = round(jval, 3)
		jaccard_list[tdisease] = fjval
	
	#for tdisease in sorted(current_diseases):
	trank = 1
	for tdisease in sorted(jaccard_list, key=jaccard_list.get, reverse=True):
		# Hypertension_roc.0.6.network.jpg.binarymap.png.map.png
		troc_file = tdisease + "_roc.0.6.network.jpg"
		timportance_file = tdisease + ".0.6.network.importance.jpg"
		tmap = tdisease + "_roc.0.6.network.jpg.binarymap.png.map.png"
		
		# get refmap
		tquant = "0.6"
		if tdisease == "life_expect":
			tquant = "0.3"
		refmap = "lancet.canonical." + tdisease + "." + str(tquant) + ".binarymap.png"
		if tdisease == "stroke_mort":
			refmap = "lancet.canonical.stroke_data..0.6.binarymap.png"

		if tdisease == "life_expect":
			troc_file = tdisease + "_roc.0.3.network.jpg"
			# life_expect.0.3.importance.jpg
			timportance_file = "life_expect.0.3.importance.jpg"
			tmap = tdisease + "_roc.0.3.network.jpg.binarymap.png.map.png"
		if tdisease == "stroke_mort":
			timportance_file = "stroke_data.0.6.importance.jpg"
		if (tmode == "logistic"):
			troc_file = troc_file + ".logistic.jpg"

		# Set disease label
		tquant = "0.6"
		disease_label = tdisease
		if tdisease == "stroke_mort":
			disease_label = "Stroke Mortality"
		if tdisease == "life_expect":
			disease_label = "Life Expectancy"			
			tquant = "0.3"

		jval, pval = GetNetworkMapJaccard(tdisease, tquant)
		fjval = round(jval, 2)
		if len(str(fjval)) == 3:
			fjval = round(jval, 3)
		fpval = formatPVal(pval)

		tHTML = tHTML + "\t<tr>\n"
		tHTML = tHTML + "\t\t<td style=\"text-align: center; font-family: arial; font-size: 20px;\">" + str(trank) + '.' + disease_label + "<br>\n"
		tHTML = tHTML + "<center><table><tr><td><img style=\"width: 100px\" src=\"" + refmap + "\"></td>"
		tHTML = tHTML + "<td><img style=\"width: 100px\" src=\"" + tmap + "\"></td></tr>"
		tHTML = tHTML + "<tr><td valign=\"top\" style=\"text-align: center;\"><span style=\"font-family: arial; font-size: 9px;\">Reference Map</span></td>"
		tHTML = tHTML + "<td valign=\"top\" style=\"text-align: center;\"><span style=\"font-family: arial; font-size: 9px;\">aPEER <br>Reassembled Map</span>"
		tHTML = tHTML + "<br><span style=\"font-family: arial; font-size: 9px;\">J=" + str(fjval) + " <i>(p=" + str(fpval) + ")" + "</i></span>\n"
		tHTML = tHTML + "</td></tr></table></center>"

		tHTML = tHTML + "</td>\n"

		tHTML = tHTML + "\t\t<td><div style=\"width: 300px; position: relative;\">\n"
		tHTML = tHTML + "<img style=\"width: 100%;\" src=\"" + troc_file + "\">"
		#tHTML = tHTML + "\t\t\t<div style=\"z-index: 10; position: absolute; left: 175px; top: " + ttop + "px; width: 150px;\">"
		#tHTML = tHTML + "<div style=\"z-index: 12; font-size: 8px; font-family: arial; position: absolute; left: 10px; top: -9px;\">J=" + str(fjval) + " <i>(p=" + str(fpval) + ")</i></div>"
		#tHTML = tHTML + "<img style=\"width: 100px\" src=\"" + tmap + "\"></div>"
		tHTML = tHTML + "\t\t\t</div></td>\n"
		tHTML = tHTML + "\t\t<td><img style=\"width: 250px;\" src=\"" + timportance_file + "\"></td>\n"
		tHTML = tHTML + "\t</tr>\n"
		
		trank += 1
		
	tHTML += "</table>\n</html>\n"

	#f = open("supplemental_xgboost_network.html", "w")
	f = open(ofile, "w")
	f.write(tHTML)
	f.close()

def MakeViolinPlots(disease_data, chemical_data, json_data_county):

	# https://stackoverflow.com/questions/75818679/violin-plot-with-categorization-using-two-different-columns-of-data-for-one-vio
	# https://www.geeksforgeeks.org/python-pandas-melt/
	
	# merge data
	disease_cols = ["STROKE_MORT", "LIFE_EXPECT", "Stroke", "Hypertension", "Diabetes", "Ozone"]
	chem_cols = ["ACETALDEHYDE", "FORMALDEHYDE", "METHANOL", "BENZENE", "CARBON TETRACHLORIDE"]
	all_list = disease_cols + chem_cols
	select_disease_data = disease_data[disease_cols]
	select_chemical_data = chemical_data[chem_cols]	

	# create chemical cluster
	cluster_data_file = "final_cluster.tsv"
	ShowTractClustersOnMap(select_chemical_data, 2, cluster_data_file, "county", json_data_county, showmap=False, forceBinary=False)
	cluster_data = pd.read_csv(cluster_data_file + ".cluster.tsv", sep="\t", dtype={'county': str})
	cluster_data.index = cluster_data['county']
	
	#print(cluster_data)

	main_data = pd.concat([select_disease_data, select_chemical_data], axis = 1)
	#main_data = main_data.drop('county', axis=1)

	# copy the data 
	main_data_scaled = main_data.copy()
	main_data_scaled.apply(pd.to_numeric)

	for column in main_data.columns: 
		main_data_scaled[column] = (main_data[column] - main_data[column].min()) / (main_data[column].max() - main_data[column].min())

	main_data_scaled['fips'] = main_data_scaled.index
	pdf = pd.melt(main_data_scaled, id_vars=['fips'], value_vars=all_list)

	#print(pdf)
	
	# combine melted dataframe with clusters
	for tid in range(0, len(pdf.index)):
		tfips = pdf.iloc[tid]['fips']
		tval = cluster_data.loc[tfips, 'cluster']
		#pdf.iloc[tid]['cluster'] = tval
		pdf.loc[tid, 'cluster'] = tval
		#print(str(tfips) + "\t" + str(tval))
		#pdf.iloc[tid, pdf.columns.get_loc('cluster')] = cluster_data.loc[tfips, 'cluster']
	
	fig, ax = plt.subplots(figsize=(12, 6))
	sns.set(font_scale=0.2)
	sns.set_theme(style='white')
	violin_fig = sns.violinplot(data=pdf, x="Variable", y="Normalized Value", hue="cluster", split=False, cut=0, scale='width', linewidth=1, saturation=1)
	#sns.stripplot(x='Manufacturer', y='Combined MPG', data=df_mpg_filtered,
              #jitter=True, linewidth=1, order=ordered)
	#violin_fig.tick_params(axis='x', labelrotation=90)
	plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)
	plt.xticks(rotation=45, ha='right')
	plt.subplots_adjust(bottom=0.3)
	fig = violin_fig.get_figure()
	fig.savefig("violins.png", dpi=300) 

def GetNetworkMapJaccard(tdisease, tquant):

	countycnt = 3141

	def ReformatBeltData(rawdata):	
		refbelt = defaultdict(int)
		for tid in rawdata:
			if rawdata[tid] == "1":
				refbelt[tid] = 1
		
		return refbelt
		
	def GetDiseaseJaccard(tlabel, clustfile, ofile):
		clustdata = LoadClusterData(clustfile)
		rawdata = LoadClusterData(ofile)
		refbelt = ReformatBeltData(rawdata)
		tjaccard, tpval = CalcJaccard(clustdata, refbelt, countycnt)
		#print(tlabel + "\t" + str(tjaccard) + "\t" + str(tpval))

		return tjaccard, tpval
			
	ofile = "lancet.canonical." + tdisease + "." + str(tquant) + ".binarymap.tsv"
	if tdisease == "stroke_mort":
		ofile = "lancet.canonical.stroke_data..0.6.binarymap.tsv"
	clustfile = tdisease + '_roc.' + str(tquant) + '.network.jpg.binarymap.png.cluster.tsv'
	tjaccard, tpval = GetDiseaseJaccard(tdisease, clustfile, ofile)
	
	return tjaccard, tpval
	
	'''
	#quant_cutoffs = [0.6, 0.7, 0.8, 0.9]
	quant_cutoffs = [0.6]
	for tquant in quant_cutoffs:
		for tdisease in disease_columns:
			ofile = "lancet.canonical." + tdisease + "." + str(tquant) + ".binarymap.tsv"
			clustfile = tdisease + '_roc.' + str(tquant) + '.network.jpg.binarymap.png.cluster.tsv'
			GetDiseaseJaccard(tdisease, clustfile, ofile)

	# get life expectancy map
	#quant_cutoffs = [0.2, 0.3, 0.4, 0.5]
	quant_cutoffs = [0.3]
	for tquant in quant_cutoffs:
		ofile = "lancet.canonical.life_expect." + str(tquant) + ".binarymap.tsv"
		clustfile = 'life_expect_roc.' + str(tquant) + '.network.jpg.binarymap.png.cluster.tsv'
		GetDiseaseJaccard("life_expect", clustfile, ofile)

	# stroke map
	#quant_cutoffs = [0.4, 0.5, 0.6, 0.7]
	quant_cutoffs = [0.6]
	for tquant in quant_cutoffs:
		ofile = "lancet.canonical.stroke_data." + "." + str(tquant) + ".binarymap.tsv"
		clustfile = 'stroke_mort_roc.' + str(tquant) + '.network.jpg.binarymap.png.cluster.tsv'
		GetDiseaseJaccard("stroke_mort", clustfile, ofile)


	# stroke belt
	#ofile = "lancet.canonical.stroke_belt.binarymap.tsv"
	#filelist.append(ofile)

	#return filelist
	'''

def ClusterPipeline(datatable):

	# random_state = 0
	# PCA
	datatable = datatable.replace(np.nan, 0)
	pca = PCA()
	Xt = pca.fit_transform(datatable)

	#epa_clust = PCA_Kmeans_Tracts(Xt_epa, numclust, datatable, kfile, cluster_file)
	PCA_components = pd.DataFrame(Xt)
	model = KMeans(n_clusters = numclust, random_state = 0)
	model.fit(PCA_components.iloc[:,:2])
	labels = model.predict(PCA_components.iloc[:,:2])
	
	return labels

def GetIdealCluster(datatable, numclust):

	labels = ClusterPipeline()

	# store clusters in dataframe
	ideal_cluster = defaultdict(int)
	clustertable = []
	for i, name in enumerate(datatable.index):
		tfips = str(name)
		if len(tfips) == 10:
			tfips = '0' + tfips
		if len(tfips) == 4:
			tfips = '0' + tfips
		if labels[i] == 0:
			ideal_cluster[name] = 0
		clustertable.append([name, labels[i]])

	clusterframe = pd.DataFrame(clustertable, columns=["county", "cluster"])
	clusterframe = clusterframe.set_index("county")
	clusterframe.to_csv("reference_map.tsv", sep="\t")

	return ideal_cluster
	
def LoadReferenceMap(tindex):

	clustertable = defaultdict(int)
	with open("reference_map.tsv", "r") as infile:
		theader = infile.readline()
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			tval = int(ldata[1])
			#if tval == 0:
			clustertable[ldata[0]] = tval
	infile.close()

	return clustertable

def BestMatchClusterOnMap(datatable, numclust, tfile, tmode, json_data, compare_data):

	print("Making File: " + tfile)

	# make filenames
	pcafile = tfile + ".pca.png"
	kfile = tfile + ".kmeans.pca.png"
	cluster_file = tfile + ".kmeans.pca.clusters.png"
	mapfile = tfile + ".map.png"
	cluster_data_file = tfile + ".cluster.tsv"
	
	# PCA
	datatable = datatable.replace(np.nan, 0)
	pca = PCA()
	Xt = pca.fit_transform(datatable)
	PCA_components = pd.DataFrame(Xt)
	model = KMeans(n_clusters = numclust, random_state = 0)
	model.fit(PCA_components.iloc[:,:2])
	labels = model.predict(PCA_components.iloc[:,:2])

	# store clusters in dataframe
	clustertable = []
	for i, name in enumerate(datatable.index):
		clustertable.append([name, labels[i]])

	clusterframe = pd.DataFrame(clustertable, columns=["county", "cluster"])
	clusterframe = clusterframe.set_index("county")

	# find the cluster that best overlaps with the reference cluster
	tcolors = ["#f5f5f5", "blue"]
	jaccard_scores = defaultdict(int)
	
	# calculate jaccard
	cluster_list = defaultdict(int)
	for tfips in clusterframe.index:
		tclust = clusterframe.loc[tfips, "cluster"]
		cluster_list[tclust] = 1
	
	for tclust in cluster_list:
		tintersect = 0
		tunion = 0
		for tfips in clusterframe.index:
			#print(tfips + "\t" + str(tclust))
			curr_clust = clusterframe.loc[tfips, "cluster"]
			if (tfips in compare_data) and (curr_clust == tclust):
				tintersect += 1
			if (tfips in compare_data) or (curr_clust == tclust):
				tunion += 1
		tjaccard = tintersect / tunion
		jaccard_scores[tclust] = tjaccard
			
	maxscore = 0
	bestclust = -1
	for tclust in jaccard_scores:
		if (jaccard_scores[tclust] > maxscore):
			bestclust = tclust
			maxscore = jaccard_scores[tclust]

	cluster_list = defaultdict(str)
	final_list = defaultdict(int)
	for tfips in clusterframe.index:
		cluster_list[tfips] = "#f5f5f5"
		if clusterframe.loc[tfips, "cluster"] == bestclust:
			cluster_list[tfips] = "blue"
			final_list[tfips] = 1
	
	#json_data = LoadMapJSON(tmode)
	plt.close()
	fig = plt.figure(figsize=(25, 16))
	ax = fig.gca()
	for titem in json_data["features"]:
		tfips = titem["properties"]["GEOID"]
		if tfips in clusterframe.index:
			#print("FIPS Code: " + tfips)
			if int(tfips[0:1]) < 6:
				#print("FIPS found")
				poly = titem["geometry"]
				if str(poly) != "None":
					#print(tfips + "\t" + str(get_color))
					ax.add_patch(PolygonPatch(poly, fc=cluster_list[tfips], ec=cluster_list[tfips], alpha=1, zorder=2))
	
	#ax.axis('scaled')
	#ax.set_xlim(-180, -60)
	#ax.set_ylim(20, 80)
	ax.set_facecolor('xkcd:white')
	ax.set_xlim(-130, -60)
	ax.set_ylim(23, 50)

	#plt.xlabel("Longitude")
	#plt.ylabel("Latitude")
	plt.axis("off")

	fig.tight_layout()
	print("Writing out cluster map file: " + mapfile)
	plt.savefig(mapfile)
	
	return final_list

def SuppEJNetworks():

	tHTML = "<html><table>\n"
	nlist = disease_columns.copy() + ["life_expect", "stroke_mort"]
	for tdisease in nlist:
		ndisease = tdisease		
		# pollution_network_lancet.canonical.Coronary Heart Disease.0.6.binarymap.tsv.jpg
		imgfile = 'networks/pollution_network_lancet.canonical.' + tdisease + '.0.6.binarymap.tsv.jpg'
		statfile = "networks/pollution_stats_lancet.canonical." + tdisease + ".0.6.binarymap.tsv.txt"
		if tdisease == "life_expect":
			imgfile = 'networks/pollution_network_lancet.canonical.' + tdisease + '.0.3.binarymap.tsv.jpg'
			statfile = "networks/pollution_stats_lancet.canonical." + tdisease + ".0.3.binarymap.tsv.txt"
			ndisease = "Life Expectancy"
		if tdisease == "stroke_mort":
			imgfile = 'networks/pollution_network_lancet.canonical.stroke_data..0.6.binarymap.tsv.jpg'
			statfile = "networks/pollution_stats_lancet.canonical.stroke_data..0.6.binarymap.tsv.txt"
			ndisease = "Stroke Mortality"

		# stats_lancet.canonical.Arthritis.0.6.binarymap.tsv.txt
		tstats = "<table style=\"text-align: center; font-family: arial; font-size: 10px;\"><tr><td><b>Pollutant</b></td><td><b>Connectedness</b></td></tr>"
		with open(statfile, "r") as infile:
			for line in infile:
				line = line.strip()
				ldata = line.split("\t")
				tstats += "<tr><td>" + ldata[0] + "</td><td>" + ldata[1] + "</td></tr>\n"
		infile.close()
		tstats += "</table>"

		tHTML += '<tr><td style="font-family: arial; font-size: 20px; text-align: center;">' + ndisease + '</td><td><img width="450" src="' + imgfile + '"></td><td>' + tstats + '</td></tr>\n'
	
	tHTML += "</table></html>\n"
	
	f = open("supp_ejscreen_networks.html", "w")
	f.write(tHTML)
	f.close()

def EJScreenJaccardVals(merge_data_counties, json_data_county):

	current_diseases = disease_columns.copy()
	#if tmode == "logistic":
	current_diseases = ['stroke_mort', 'life_expect'] + current_diseases

	jaccard_list = defaultdict(float)
	pval_list = defaultdict(float)
	for tdisease in current_diseases:
		tquant = "0.6"
		if tdisease == "life_expect":
			tquant = "0.3"
		#jval, pval = GetNetworkMapJaccard(tdisease, tquant)
		jval, pval = GetEJRefMap(tdisease, float(tquant), merge_data_counties, json_data_county)
		fjval = round(jval, 2)
		if len(str(fjval)) == 3:
			fjval = round(jval, 3)
		jaccard_list[tdisease] = fjval
		pval_list[tdisease] = pval

	f = open("ejscreen_jaccard_vals.tsv", "w")
	for tid in jaccard_list:
		f.write(tid + "\t" + str(jaccard_list[tid]) + "\t" + str(pval_list[tid]) + "\n")
	f.close()

def GetEJNetworkRegModels(tmode, ofile, merge_data_counties, json_data_county):

	ttop = "80"
	if tmode == "logistic":
		ttop = "60"

	# XGBoost and Network Connectedness
	tHTML = "<html>\n"
	tHTML += "<table>\n"
	
	current_diseases = disease_columns.copy()
	#if tmode == "logistic":
	current_diseases = ['stroke_mort', 'life_expect'] + current_diseases

	jaccard_list = defaultdict(float)
	pval_list = defaultdict(float)
	with open("ejscreen_jaccard_vals.tsv") as infile:
		for line in infile:
			line = line.strip()
			ldata = line.split("\t")
			jaccard_list[ldata[0]] = float(ldata[1])
			pval_list[ldata[0]] = float(ldata[2])
	infile.close()
	
	#for tdisease in sorted(current_diseases):
	trank = 1
	for tdisease in sorted(jaccard_list, key=jaccard_list.get, reverse=True):
		# Hypertension_roc.0.6.network.jpg.binarymap.png.map.png

		# write out regression models
		# stroke_mort_roc.0.6.ejscreen.network.jpg
		# stroke_mort_roc.0.6.ejscreen.network.jpg.EJSCREEN Model_model.xgboost.pdf.importance.jpg
		# pollution_stats_lancet.canonical.Arthritis.0.6.binarymap.tsv.txt.map.png
		# lancet.canonical.Obesity.0.6.binarymap.png

		troc_file = tdisease + "_roc.0.6.ejscreen.network.jpg"
		timportance_file = tdisease + "_roc.0.6.ejscreen.network.jpg.EJSCREEN Model_model.xgboost.pdf.importance.jpg"
		#tmap = tdisease + "_roc.0.6.network.jpg.binarymap.png.map.png"
		tmap = "pollution_stats_lancet.canonical." + tdisease + ".0.6.binarymap.tsv.txt.map.png"
		
		# get refmap
		tquant = "0.6"
		if tdisease == "life_expect":
			tquant = "0.3"
		refmap = "lancet.canonical." + tdisease + "." + str(tquant) + ".binarymap.png"
		if tdisease == "stroke_mort":
			refmap = "lancet.canonical.stroke_data..0.6.binarymap.png"
			timportance_file = "stroke_mort_roc.0.6.ejscreen.network.jpg.EJSCREEN Model_model.xgboost.pdf.importance.jpg"
		if tdisease == "life_expect":
			troc_file = tdisease + "_roc.0.3.network.jpg"
			timportance_file = "life_expect_roc.0.3.ejscreen.network.jpg.EJSCREEN Model_model.xgboost.pdf.importance.jpg"
			#tmap = tdisease + "_roc.0.3.network.jpg.binarymap.png.map.png"
			tmap = "pollution_stats_lancet.canonical." + tdisease + ".0.3.binarymap.tsv.txt.map.png"
		if (tmode == "logistic"):
			troc_file = troc_file + ".logistic.jpg"

		# Set disease label
		tquant = "0.6"
		disease_label = tdisease
		if tdisease == "stroke_mort":
			disease_label = "Stroke Mortality"
		if tdisease == "life_expect":
			disease_label = "Life Expectancy"			
			tquant = "0.3"

		#jval, pval = GetNetworkMapJaccard(tdisease, tquant)
		jval = jaccard_list[tdisease]
		fjval = round(jval, 2)
		if len(str(fjval)) == 3:
			fjval = round(jval, 3)
		pval = pval_list[tdisease]
		fpval = formatPVal(pval)

		tHTML = tHTML + "\t<tr>\n"
		tHTML = tHTML + "\t\t<td style=\"text-align: center; font-family: arial; font-size: 20px;\">" + str(trank) + '.' + disease_label + "<br>\n"
		tHTML = tHTML + "<center><table><tr><td><img style=\"width: 100px\" src=\"" + refmap + "\"></td>"
		tHTML = tHTML + "<td><img style=\"width: 100px\" src=\"" + tmap + "\"></td></tr>"
		tHTML = tHTML + "<tr><td valign=\"top\" style=\"text-align: center;\"><span style=\"font-family: arial; font-size: 9px;\">Reference Map</span></td>"
		tHTML = tHTML + "<td valign=\"top\" style=\"text-align: center;\"><span style=\"font-family: arial; font-size: 9px;\">aPEER <br>Reassembled Map</span>"
		tHTML = tHTML + "<br><span style=\"font-family: arial; font-size: 9px;\">J=" + str(fjval) + " <i>(p=" + str(fpval) + ")" + "</i></span>\n"
		tHTML = tHTML + "</td></tr></table></center>"

		tHTML = tHTML + "</td>\n"

		tHTML = tHTML + "\t\t<td><div style=\"width: 300px; position: relative;\">\n"
		tHTML = tHTML + "<img style=\"width: 100%;\" src=\"" + troc_file + "\">"
		#tHTML = tHTML + "\t\t\t<div style=\"z-index: 10; position: absolute; left: 175px; top: " + ttop + "px; width: 150px;\">"
		#tHTML = tHTML + "<div style=\"z-index: 12; font-size: 8px; font-family: arial; position: absolute; left: 10px; top: -9px;\">J=" + str(fjval) + " <i>(p=" + str(fpval) + ")</i></div>"
		#tHTML = tHTML + "<img style=\"width: 100px\" src=\"" + tmap + "\"></div>"
		tHTML = tHTML + "\t\t\t</div></td>\n"
		tHTML = tHTML + "\t\t<td><img style=\"width: 250px;\" src=\"" + timportance_file + "\"></td>\n"
		tHTML = tHTML + "\t\t<td>"
		
		# stats_lancet.canonical.Arthritis.0.6.binarymap.tsv.txt
		statfile = "networks/pollution_stats_lancet.canonical." + tdisease + ".0.6.binarymap.tsv.txt"
		if tdisease == "life_expect":
			statfile = "networks/pollution_stats_lancet.canonical." + tdisease + ".0.3.binarymap.tsv.txt"
		if tdisease == "stroke_mort":
			statfile = "networks/pollution_stats_lancet.canonical.stroke_data..0.6.binarymap.tsv.txt"

		# generate bar charts
		bar_file = statfile + ".jpg"
		ttable = []
		bar_data = defaultdict(int)
		with open(statfile, "r") as infile:
			for line in infile:
				line = line.strip()
				ldata = line.split("\t")
				ttable.append([ldata[0], int(ldata[1])])
		infile.close()

		bar_data = pd.DataFrame(ttable, columns=['columns', 'importances'])
		bar_data[['importances']] = bar_data[['importances']].apply(pd.to_numeric)
		# reverse order
		bar_data = bar_data.iloc[::-1]
				
		TopValsBarChart(bar_data, bar_file, "Network Connectedness")

		#tstats = "<table style=\"text-align: center; font-family: arial; font-size: 10px;\"><tr><td><b>Pollutant</b></td><td><b>Connectedness</b></td></tr>"
		#with open(statfile, "r") as infile:
		#	for line in infile:
		#		line = line.strip()
		#		ldata = line.split("\t")
		#		tstats += "<tr><td>" + ldata[0] + "</td><td>" + ldata[1] + "</td></tr>\n"
		#infile.close()
		#tstats += "</table>"
		#tHTML = tHTML + tstats + "</td>\n"
		
		tHTML = tHTML + '<img style="width: 250px;" src="' + bar_file + '">' + "\n"
		
		tHTML = tHTML + "\t</tr>\n"
		
		trank += 1
		
	tHTML += "</table>\n</html>\n"

	#f = open("supplemental_xgboost_network.html", "w")
	f = open(ofile, "w")
	f.write(tHTML)
	f.close()

def MakeEJRegs(baseid, tcutoff, merge_data_counties, chemical_data, chemlist, polldata):
	baseid2 = baseid
	if (baseid == "life_expect") or (baseid == "stroke_mort"):
		baseid2 = baseid.upper()
	tfile = baseid + "_roc." + str(tcutoff) + ".jpg"
	tfile2 = baseid + "_roc." + str(tcutoff) + ".network.jpg"
	tfile3 = baseid + "_roc." + str(tcutoff) + ".ejscreen.network.jpg"		
	MakeRefROCFigure("ejscreen", merge_data_counties, chemical_data, chemlist, baseid2, tcutoff, tfile)
	MakeEJSCREENROCFigure(merge_data_counties, chemical_data, polldata, baseid2, tcutoff, tfile3)

def GetEJRefMap(baseid, tcutoff, merge_data_counties, json_data_county):

	tclust = 3
	top_poll = 2
	four_clust = ["Renal Disease", "Coronary Heart Disease", "Cancer", "Arthritis", "COPD"]

	# reference map
	refdata = LoadReferenceMap(0)
	
	# get reference map
	bfile = "lancet.canonical." + baseid + "." + str(tcutoff) + ".binarymap.png"
	ofile = "lancet.canonical." + baseid + "." + str(tcutoff) + ".binarymap.tsv"
	tfile2 = "pollution_stats_lancet.canonical." + baseid + "." + str(tcutoff) + ".binarymap.tsv.txt"

	# reference map
	if (baseid != "life_expect") and (baseid != "stroke_mort"):
		tdata = merge_data_counties[[baseid]]
		maplist = GetReferenceMap(tdata, ofile, tcutoff)

	if baseid == "life_expect":
		maplist = LoadLifeExpectancy(merge_data_counties, ofile, tcutoff, "threshold")

	if baseid == "stroke_mort":
		bfile = "lancet.canonical.stroke_data." + "." + str(tcutoff) + ".binarymap.png"
		ofile = "lancet.canonical.stroke_data." + "." + str(tcutoff) + ".binarymap.tsv"
		#tfile2 = "pollution_stats_lancet.canonical." + baseid + "." + str(tcutoff) + ".binarymap.tsv.txt"
		tfile2 = "pollution_stats_lancet.canonical.stroke_data.." + str(tcutoff) + ".binarymap.tsv.txt"

		maplist = LoadStrokeMort(merge_data_counties, ofile, tcutoff, "threshold")

	#ShowBinaryMap(maplist, bfile, json_data_county, "#f5f5f5", "blue")

	# Get binary clustered map
	column_list, bar_data = GetTopNetworkPollutants(tfile2, top_poll)
	tdata = merge_data_counties[column_list]
	print("Columns: " + str(column_list))
	#ShowTractClustersOnMap(tdata, nclust, tfile2 + ".binarymap.png", "county", json_data_county, showmap=True, forceBinary=False)
	clusterdata = BestMatchClusterOnMap(tdata, tclust, tfile2, "county", json_data_county, refdata)
	
	# calculate Jaccard Index between best match and reference map
	countycnt = 3141
	tjaccard, tpval = CalcJaccard(clusterdata, maplist, countycnt)
	print(baseid + "\t" + "Jaccard: " + str(tjaccard) + "\t" + str(tpval))
	
	return tjaccard, tpval

def CompareReferenceMaps():

	total_list = ["Arthritis", "Hypertension", "Coronary Heart Disease", "COPD", "Depression", "Diabetes", "Renal Disease", "Obesity", "Stroke", "STROKE_MORT", "Cancer", "Asthma"]
	tHTML = "<html><table border=1><tr><td><center>Disease</center></td><td><center>Life Expect Adj with PCA/Kmeans</center></td><td><center>60th Percentile</center></td></tr>\n"
	for pairx in range(0, len(total_list)):
		#for nclust in range(2, 6):
		nclust = 3
		tfile = "maps/lancet_disease_pair." + total_list[pairx] + ".LIFE_EXPECT_2019.county_cluster." + str(nclust) + ".map.png"
		tfile2 = "lancet.canonical." + total_list[pairx] + ".0.6.binarymap.png"
		tfile = tfile.replace(" ", "")
		tHTML += "<tr><td><center>" + total_list[pairx] + "</center></td><td><img src=\"" + tfile + "\" width=\"300\"></td>"
		if total_list[pairx] != "STROKE_MORT":
			tHTML += "<td><img src=\"" + tfile2 + "\" width=\"300\"></td></tr>"
		if total_list[pairx] == "STROKE_MORT":
			tHTML += "<td><img src=\"lancet.canonical.stroke_data..0.7.binarymap.png\" width=\"300\"></td></tr>"

	tHTML += "</table></html>\n"
	
	f = open("compare_ref_maps.html", "w")
	f.write(tHTML)
	f.close()

def GetDensityPlot(tdata, tcolumns, collabels, ofile, tmax):

	# https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0
	tcnt = 0
	tdataitem = 0
	
	ciparm = False
	if (ofile == "ci_density_pollutant.png") or (ofile == "ci_density_disease.png"):
		ciparm = True
	
	labelcnt = 0
	dotted_cutoff = int(len(tcolumns) / 2)	
	for tcol in tcolumns:
		if (ciparm == False):
			tsubset = tdata[[tcol]]
		if (ciparm == True):
			tsubset = np.log2(tdata[[tcol]])
		tlabel = tcol
		if ciparm == True:
			tlabel = collabels[tcnt]
		kde_parm = {'linewidth': 2}
		if tcnt > dotted_cutoff:
			kde_parm = {'linewidth': 2, 'linestyle': '--'}		
		sns.distplot(tsubset, hist=False, kde=True, kde_kws=kde_parm, label=tlabel)
		tcnt += 1

	plt.legend(prop={'size': 7}, title = 'Measure')
	plt.title('')
	if (ciparm == False):
		plt.xlabel('Normalized Value (Concentration or Rate)')
	if (ciparm == True):
		plt.xlabel('Ln(Confidence Interval Width)')
		
	plt.ylabel('Density')
	
	if (ofile != "ci_density_disease.png"):
		plt.xlim([-2, 4])
	if (ofile == "ci_density_pollutant.png"):
		plt.xlim([-40, 10])

	plt.savefig(ofile, dpi=1200)
	plt.close()

