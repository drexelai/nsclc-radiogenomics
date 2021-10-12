import pandas as pd
import numpy as np
import os
import pydicom
import matplotlib.pyplot as plt

from lungmask import mask
import SimpleITK as sitk

def caclVolumne(pixel_array):
	pass


def calcSurfaceArea(pixel_array):
	pass


def calcSphericity(pixel_array):
	pass


def calcMean(pixel_array):
	pass


def calcSRD(pixel_array):
	pass
	

def calcPercentiles(pixel_array):
	pass


def calcKurtosis(pixel_array):
	pass


def calcSkewness(pixel_array):
	pass


def calcFrequency(pixel_array):
	pass


def caclVesselDesnity(pixel_array):
	pass


def calcAutoCorrelatioin(pixel_array):
	pass


def calcClusterProminence(pixel_array):
	pass


def calcDifferenceVariance(pixel_array):
	pass


def calcHomogenity(pixel_array):
	pass


def segmentLungs(imagepath):
	# imagepath = '/Volumes/Extended/drexelai/radiogenomics/NSCLC/NSCLC_Radiogenomics-6-1-21 Version 4/manifest-1622561851074/NSCLC Radiogenomics/AMC-001/1.3.6.1.4.1.14519.5.2.1.4334.1501.227933499470131058806289574760/1.3.6.1.4.1.14519.5.2.1.4334.1501.131836349235351218393791897864/1-092.dcm'

	segimagepath = os.path.join(os.path.split(imagepath)[0], 'seg_' + os.path.split(imagepath)[1].split('.')[0]) + '.npy'

	if os.path.isfile(segimagepath):
		segmentation = np.load(segimagepath)
	else:
		pixel_array = sitk.ReadImage(imagepath)
		segmentation = mask.apply(pixel_array)[0]
		np.save(segimagepath, segmentation)
	# plt.imshow(pydicom.dcmread(imagepath).pixel_array, cmap=plt.cm.bone)
	# plt.imshow(segmentation, cmap='jet', alpha=0.2)
	# plt.show()
	return segmentation


def calcFeatures(segmentation, pixel_array):
	return [caclVolumne(pixel_array_seg), calcSurfaceArea(pixel_array_seg), calcSphericity(pixel_array_seg), calcMean(pixel_array_seg), calcSRD(pixel_array_seg), calcPrecentiles(pixel_array_seg), calcPercentiles(pixel_array_seg), calcKurtosis(pixel_array_seg), calcSkewness(pixel_array_seg), calcFrequency(pixel_array_seg), caclVesselDesnity(pixel_array_seg), calcAutoCorrelatioin(pixel_array_seg), calcClusterProminence(pixel_array_seg), calcDifferenceVariance(pixel_array_seg), calcHomogenity(pixel_array_seg)]


def main(rootdir):
	
	meta = pd.read_csv(os.path.join(rootdir, 'metadata.csv'))
	patientsdir = os.path.join(rootdir, 'NSCLC Radiogenomics')

	results = {}

	for patientid in os.listdir(patientsdir):
		if patientid == '.DS_Store':
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
					imagepath = os.path.join(study2dir, image)
					# TODO: figure out which images to actually encode for features
					segmentation = segmentLungs(imagepath)
					pixel_array = pydicom.dcmread(imagepath).pixel_array

					results[patientid] = calcFeatures(segmentation, pixel_array)

if __name__ == '__main__':
	rootdir = '/Volumes/Extended/drexelai/radiogenomics/NSCLC/NSCLC_Radiogenomics-6-1-21 Version 4/manifest-1622561851074'

	main(rootdir)




