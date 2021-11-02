import pandas as pd
import numpy as np
import os
import pydicom
import matplotlib.pyplot as plt
import gzip
import argparse
import seaborn as sns

from lungmask import mask
import SimpleITK as sitk
import radiomics

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

def calcFirstOrderFeatures(segmentation, pixel_array):
	extractor = radiomics.firstorder.RadiomicsFirstOrder(pixel_array, segmentation)
	return np.array([v for _, v in extractor.execute().items()])


def calcShapeBased3DFeatures(segmentation, pixel_array):
	extractor = radiomics.shape.RadiomicsShape(pixel_array, segmentation)
	return np.array([v for _, v in extractor.execute().items()])


def calcShapeBased2DFeatures(segmentation, pixel_array):
	extractor = radiomics.shape2D.RadiomicsShape2D(pixel_array, segmentation, force2D=True)
	return np.array([v for _, v in extractor.execute().items()])


def calcGLCMFeatures(segmentation, pixel_array):
	extractor = radiomics.glcm.RadiomicsGLCM(pixel_array, segmentation)
	return np.array([v for _, v in extractor.execute().items()])


def calcGLRLMFeatures(segmentation, pixel_array):
	extractor = radiomics.glrlm.RadiomicsGLRLM(pixel_array, segmentation)
	return np.array([v for _, v in extractor.execute().items()])


def calcGLSZMFeatures(segmentation, pixel_array):
	extractor = radiomics.glszm.RadiomicsGLSZM(pixel_array, segmentation)
	return np.array([v for _, v in extractor.execute().items()])


def calcNGTDMFeatures(segmentation, pixel_array):
	extractor = radiomics.ngtdm.RadiomicsNGTDM(pixel_array, segmentation)
	return np.array([v for _, v in extractor.execute().items()])


def calcGLDMFeatures(segmentation, pixel_array):
	extractor = radiomics.gldm.RadiomicsGLDM(pixel_array, segmentation)
	return np.array([v for _, v in extractor.execute().items()])


def calcFeatures(segmentation, pixel_array):
	return np.concatenate([calcFirstOrderFeatures(segmentation, pixel_array), calcShapeBased3DFeatures(segmentation, pixel_array), calcShapeBased2DFeatures(segmentation, pixel_array), calcGLCMFeatures(segmentation, pixel_array), calcGLRLMFeatures(segmentation, pixel_array), calcGLSZMFeatures(segmentation, pixel_array), calcNGTDMFeatures(segmentation, pixel_array), calcGLDMFeatures(segmentation, pixel_array)])


def segmentLungs(imagepath):
	# imagepath = '/Volumes/Extended/drexelai/radiogenomics/NSCLC/NSCLC_Radiogenomics-6-1-21 Version 4/manifest-1622561851074/NSCLC Radiogenomics/AMC-001/1.3.6.1.4.1.14519.5.2.1.4334.1501.227933499470131058806289574760/1.3.6.1.4.1.14519.5.2.1.4334.1501.131836349235351218393791897864/1-092.dcm'

	segimagepath = os.path.join(os.path.split(imagepath)[0], 'seg_' + os.path.split(imagepath)[1].split('.')[0]) + '.dcm'

	if os.path.isfile(segimagepath):
		segmentation = sitk.ReadImage(segimagepath)
	else:
		pixel_array = sitk.ReadImage(imagepath)
		segmentation = mask.apply(pixel_array)
		# segmentation_ = segmentation
		segmentation = sitk.GetImageFromArray(segmentation)  
		sitk.WriteImage(segmentation, segimagepath)
		# np.save(segimagepath, segmentation)
	# plt.imshow(pydicom.dcmread(imagepath).pixel_array, cmap=plt.cm.bone)
	# plt.imshow(segmentation, cmap='jet', alpha=0.2)
	# plt.show()
	return segmentation


def preprocessImagingData(rootdir):
	imagemeta = pd.read_csv(os.path.join(rootdir, 'NSCLC_Radiogenomics-6-1-21 Version 4/manifest-1622561851074/metadata.csv'))
	patientmeta = pd.read_csv(os.path.join(rootdir, 'NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv'))
	patientsdir = os.path.join(rootdir, 'NSCLC_Radiogenomics-6-1-21 Version 4/manifest-1622561851074/NSCLC Radiogenomics')

	results = {}

	for patientid in os.listdir(patientsdir):
		if patientid == '.DS_Store' or patientid not in patientmeta['Case ID'][patientmeta['rnaseq']]:
			continue
		patientdir = os.path.join(patientsdir, patientid)
		for studyid in os.listdir(patientdir):
			if studyid == '.DS_Store':
				continue
			studydir = os.path.join(patientdir, studyid)
			for studyid2 in os.listdir(studydir):
				if studyid2 == '.DS_Store':
					continue
				study2dir = os.path.join(studydir, studyid2)
				for image in os.listdir(study2dir):
					if image == '.DS_Store':
						continue

					# TODO: select the most representative image per person

					imagepath = os.path.join(study2dir, image)
					segmentation = segmentLungs(imagepath)
					pixel_array = pydicom.dcmread(imagepath).pixel_array

					results[patientid] = calcFeatures(segmentation, pixel_array)

	return results


def preprocessRNASeq(rootdir):
	rnaseqloc = os.path.join(rootdir, 'GSE103584_R01_NSCLC_RNAseq.txt.gz')
	rnaseqdata = pd.read_csv(gzip.open(rnaseqloc), sep='\t', index_col=0)
	rnaseqdata = rnaseqdata.fillna(0)
	rnaseqdata = rnaseqdata.drop(index=rnaseqdata.index[np.where(rnaseqdata.sum(axis=1) == 0)[0]])
	rnaseqdata = rnaseqdata.apply(lambda x: (x - np.mean(x)) / np.std(x), axis=1)
	return rnaseqdata

genome_patients = ['R01-023', 'R01-024', 'R01-006', 'R01-153', 'R01-031', 'R01-032',
       'R01-033', 'R01-034', 'R01-035', 'R01-037', 'R01-005', 'R01-147',
       'R01-051', 'R01-043', 'R01-028', 'R01-052', 'R01-056', 'R01-057',
       'R01-059', 'R01-060', 'R01-061', 'R01-062', 'R01-063', 'R01-066',
       'R01-067', 'R01-068', 'R01-072', 'R01-080', 'R01-081', 'R01-154',
       'R01-083', 'R01-084', 'R01-048', 'R01-077', 'R01-078', 'R01-003',
       'R01-007', 'R01-012', 'R01-013', 'R01-015', 'R01-016', 'R01-017',
       'R01-018', 'R01-021', 'R01-022', 'R01-026', 'R01-004', 'R01-014',
       'R01-027', 'R01-029', 'R01-038', 'R01-039', 'R01-040', 'R01-041',
       'R01-042', 'R01-046', 'R01-156', 'R01-049', 'R01-160', 'R01-054',
       'R01-055', 'R01-064', 'R01-065', 'R01-069', 'R01-071', 'R01-073',
       'R01-076', 'R01-148', 'R01-149', 'R01-079', 'R01-150', 'R01-089',
       'R01-157', 'R01-158', 'R01-151', 'R01-152', 'R01-091', 'R01-159',
       'R01-093', 'R01-094', 'R01-096', 'R01-097', 'R01-098', 'R01-099',
       'R01-100', 'R01-101', 'R01-102', 'R01-103', 'R01-104', 'R01-105',
       'R01-106', 'R01-107', 'R01-108', 'R01-109', 'R01-110', 'R01-111',
       'R01-112', 'R01-113', 'R01-114', 'R01-115', 'R01-116', 'R01-117',
       'R01-118', 'R01-119', 'R01-120', 'R01-121', 'R01-122', 'R01-123',
       'R01-124', 'R01-125', 'R01-126', 'R01-127', 'R01-128', 'R01-129',
       'R01-130', 'R01-131', 'R01-132', 'R01-133', 'R01-134', 'R01-135',
       'R01-136', 'R01-137', 'R01-138', 'R01-139', 'R01-140', 'R01-141',
       'R01-142', 'R01-144', 'R01-145', 'R01-146']


def preprocessClinicalData(data):
	"""Fills in missing values, standardizes, one-hot & categorically encodes, and returns a dataframe ready to be split into train and test sets"""
	#Missing/improper value replacement
	data["Weight (lbs)"].replace("Not Collected", 0, inplace=True)
	data["Weight (lbs)"] = pd.to_numeric(data["Weight (lbs)"])
	data["Weight (lbs)"].replace(0, data["Weight (lbs)"].mean(), inplace=True)
	data["Pack Years"].replace("Not Collected", 0, inplace=True)
	data["Pack Years"] = pd.to_numeric(data["Pack Years"])
	data["Pack Years"].replace(0, data["Pack Years"].mean(), inplace=True) 
	data["Pack Years"].replace(np.NaN, data["Quit Smoking Year"].mean(), inplace=True) 
	data["%GG"].replace("Not Assessed", "0%", inplace=True)
	recurr_dates = pd.to_datetime(data["Date of Recurrence"])
	#Binning the Recurrence dates
	data["Date of Recurrence"] = recurr_dates
	r_dates = []
	for date in data["Date of Recurrence"]:
		if pd.isna(date.year):
			r_dates.append("None")
		elif date.year <= 1992:
			r_dates.append("1-2")   
		elif date.year > 1992 and date.year <= 1994:
			r_dates.append("2-4")
		elif date.year > 1994 and date.year <= 1996:
			r_dates.append("4-6")
		elif date.year > 1996 and date.year <= 1998:
			r_dates.append("6-8")
		else:
			r_dates.append("8-10")
	#Binning the Death dates
	data["Date of Recurrence"] = r_dates
	death = []
	for days in data["Time to Death (days)"]:
		if pd.isna(days):
			death.append("None")
		elif days/365 <= 2:
			death.append("1-2")
		elif days/365 > 2 and days/365 <=3:
			death.append("2-3")
		elif days/365 > 3 and days/365 <=4:
			death.append("3-4")
		elif days/365 > 4 and days/365 <=5:
			death.append("4-5")
		elif days/365 > 5 and days/365 <=6:
			death.append("5-6")
		elif days/365 > 6:
			death.append("6-7")
		else:
			death.append("8-10")
	data["Time to Death (years)"] = death
	data.drop('Time to Death (days)', axis=1, inplace=True)
	data = data[data["Case ID"].isin(genome_patients)]
	#Encoding & Normalizing
	ordinal_feats = ["Former smoker", "Non smoker","Current smoker", "%GG","Pathological T stage", "Pathological N stage", "Pathological M stage", "Histopathological Grade", "Lymphovascular invasion","Time to Death (years)", "Date of Recurrence", "Survival Status", "Recurrence", "Recurrence Location", "Chemotherapy", "Radiation", "EGFR mutation status", "KRAS mutation status", "ALK translocation status"]
	hotenc_feats = ["Patient affiliation", "Gender", "Ethnicity","Tumor Location (choice=RUL)", "Tumor Location (choice=RML)", "Tumor Location (choice=RLL)", "Tumor Location (choice=LUL)","Tumor Location (choice=LLL)", "Tumor Location (choice=L Lingula)", "Tumor Location (choice=Unknown)", "Histology ", "Pleural invasion (elastic, visceral, or parietal)"]
	scaled_feats = ["Weight (lbs)", "Age at Histological Diagnosis","Pack Years", "Quit Smoking Year", "Days between CT and surgery"]

	for o in ordinal_feats:
		ordenc = OrdinalEncoder()
		data[o] = ordenc.fit_transform(data[[o]])

	for o in hotenc_feats:
		hotenc = OneHotEncoder(handle_unknown='ignore') 
		data[o] = hotenc.fit_transform(data[[o]])

	for o in scaled_feats:
		scaler = StandardScaler()
		data[o] = scaler.fit_transform(data[[o]])      

		
	fm = []
	ns = []
	cs = []

	for status in data["Smoking status"]:
		if status == "Former":
			fm.append(1)
			ns.append(0)
			cs.append(0)
		elif status == "Nonsmoker":
			fm.append(0)
			ns.append(1)
			cs.append(0)
		elif status == "Current":
			fm.append(0)
			ns.append(0)
			cs.append(1)
	data["Former smoker"] = fm
	data["Non smoker"] = ns
	data["Current smoker"] = cs
	data.drop('Smoking status', axis=1, inplace=True)
	data["Recurrence Location"].replace(np.NaN, "none", inplace=True) 

def display_correlation_matrix(data):
	""" Displays a correlation matrix for a dataset """
	corr = data.corr()
	mask = np.triu(np.ones_like(corr, dtype=bool))
	f, ax = plt.subplots(figsize=(50, 50))
	cmap = sns.diverging_palette(20, 230, as_cmap=True)
	sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
	            square=True, annot=True,linewidths=.5, cbar_kws={"shrink": .5})



def preprocessData(rootdir):
	patientloc = os.path.join(rootdir, 'NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv')
	patientmeta = pd.read_csv(patientloc)

	patientmeta = preprocessClinicalData(patientmeta)

	rnaseqdata = preprocessRNASeq(rootdir)

	if 'rnaseq' not in patientmeta.columns:
		patientmeta['rnaseq'] = [e in rnaseqdata.columns for e in patientmeta['Case ID']]
		patientmeta.to_csv(patientloc, index=False)

	imagedata = preprocessImagingData(rootdir)
	clinicaldata = preprocessClinicalData(rootdir)

	data = pd.concat([rnaseqdata, imagedata, clinicaldata])
	data = data.dropna(axis=1, how='any')
	y = (patientmeta.loc[[e in data.columns.values for e in patientmeta.loc[:, 'Case ID']], 'Recurrence'] == 'yes').values.astype(int)
	X = np.transpose(data.values)

	return X, y



def runRandomForest(X_train, X_test, y_train, y_test): 
	model = RandomForestClassifier()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	precision, recall, fbeta_score, _ = precision_recall_fscore_support(y_test, y_pred)
	return precision, recall, fbeta_score

def main(rootdir):
	X, y = preprocessData(rootdir)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
	runRandomForest(X_train, X_test, y_train, y_test)



if __name__ == '__main__':
	rootdir = '/Volumes/Extended/drexelai/radiogenomics/NSCLC/'

	main(rootdir)




