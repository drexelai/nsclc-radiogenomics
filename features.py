import pandas as pd
import numpy as np
import os
import pydicom
import matplotlib.pyplot as plt
import gzip
import argparse

from lungmask import mask
import SimpleITK as sitk
import radiomics

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import xgboost as xgb

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


def preprocessClinicalData(rootdir):
	pass



def preprocessData(rootdir):
	patientloc = os.path.join(rootdir, 'NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv')
	patientmeta = pd.read_csv(patientloc)

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




