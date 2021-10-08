import pandas as pd
import numpy as np
import os
import pydicom

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


def calcPrecentiles(pixel_array):
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


def segment_lungs(pixel_array):
	pass


def find_pix_dim(ct_img):
    """
    Get the pixdim of the CT image.
    A general solution that gets the pixdim indicated from the image dimensions. From the last 2 image dimensions, we get their pixel dimension.
    Args:
        ct_img: nib image

    Returns: List of the 2 pixel dimensions
    """
    pix_dim = ct_img.header["pixdim"] # example [1,2,1.5,1,1]
    dim = ct_img.header["dim"] # example [1,512,512,1,1]
    max_indx = np.argmax(dim)
    pixdimX = pix_dim[max_indx]
    dim = np.delete(dim, max_indx)
    pix_dim = np.delete(pix_dim, max_indx)
    max_indy = np.argmax(dim)
    pixdimY = pix_dim[max_indy]
    return [pixdimX, pixdimY] # example [2, 1.5]
    

def calcFeatures(pixel_array):

	pixel_array_seg = segment_lungs(pixel_array)

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
					results[patientid] = calcFeatures(pydicom.dcmread(imagepath).pixel_array)

if __name__ == '__main__':
	rootdir = '/Volumes/Extended/drexelai/radiogenomics/NSCLC/NSCLC_Radiogenomics-6-1-21 Version 4/manifest-1622561851074'

	main(rootdir)




