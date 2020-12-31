import numpy as np
import pickle
import scvelo as scv

def load_adata(file):

	# INPUT
	# file = path to file containing a pickle object
	# OUTPUT
	# AnnData object

	with open(file, "rb") as inF:
		obj = pickle.load(inF)

		return obj


def save_adata(obj, filename):

	# IPUT
	# obj = python object
	# filename = path to save object

	with open(filename, "wb") as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def unique(list1):

	# INPUT
	# list1 = python list
	# OUTPUT:
	# numpy array with unique elements in the list

	x = np.array(list1)
	return np.unique(x)
