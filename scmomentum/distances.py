import copy
import numpy as np
import pandas as pd
import random
import scvelo as scv
import skbio as sk
from sklearn import preprocessing
from scmomentum.utilities import unique

def rescale(df):

	# INPUT
	#  df =  distance matrix data frame
	# OUTPUT
	# scaled_df = scaled distance matrix

	scaled_df = df
	values = []
	for i in range(0, len(df.index)):
		for j in range(i, len(df.columns)):
			values.append(df.iloc[i, j])

	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(np.array(values).reshape(-1, 1))

	k = 0
	for i in range(0, len(df.index)):
		for j in range(i, len(df.columns)):
			scaled_df.iloc[i, j] = x_scaled[k]
			scaled_df.iloc[j, i] = x_scaled[k]
			k = k + 1
	return scaled_df


def equal_dimentions(cluster_networks):

	# Recieves a list of two matrices
	# Returns a list of two matrices in which all the genes of one exist in the other
	# with random small if not present before
	# values re-scaled to each columns minimum and maximum

	newmats = []
	allgenes = unique(
		cluster_networks[0].columns.values.tolist()
		+ cluster_networks[1].columns.values.tolist()
	)

	for mat in mats:

		l = [not g in cluster_networks.columns.values.tolist() for g in allgenes]
		missing = allgenes[l].tolist()

		if len(missing) == 0:

			mat.sort_index(inplace=True)
			mat = mat.reindex(sorted(mat.columns), axis=1)
			newmats.append(mat)

		else:
			nm = mat

			for i in range(0, len(missing)):
				nm[missing[i]] = random.uniform(0, 0.001)
				nm.loc[missing[i]] = random.uniform(0, 0.001)

			nm.sort_index(inplace=True)
			nm = nm.reindex(sorted(nm.columns), axis=1)
			newmats.append(nm)

	return (np.array(newmats[0]), np.array(newmats[1]))


def expression_distance(adata, clustcol, resc=True, copy=False):

	# INPUT
	# adata - AnnData object
	# clustcol - name of the column used to cluster cells
	# resc - whether to normalize and rescale distances (recommended)
	# copy - whether a copy of the distance matrix should be returned. By default copy=True and adata.uns['expression_distances'] is updated

	clusters = [c for c in adata.obs.dropna()[clustcol].unique() if c != "nan"]
	centroids = [
		np.array(
			np.mean(
				adata.layers["spliced"][(adata.obs[clustcol] == c).values, :], axis=0
			)[:,].tolist()[
				0
			]
		)
		for c in clusters
	]

	nc = len(centroids)
	dist = pd.DataFrame(0, index=range(0, nc), columns=range(0, nc), dtype=np.float64)
	for i in range(0, nc):
		for j in range(i, nc):
			dist.at[i, j] = np.linalg.norm(centroids[i] - centroids[j])
			dist.at[j, i] = dist.at[i, j]

	if resc:
		dist = rescale(dist)

	dist = sk.DistanceMatrix(dist.values)

	if copy == False:
		adata.uns["expression_distances"] = dist
	else:
		return dist


def network_distance(networks, resc=True,dis_type='euclidean'):

	# Inputs:
	#  - cluter_mats = list with two network adjacency matrices (each a data frame with gene names as index and columns)
	#  - resc = bool, wether to rescale the distances using min to max scaling, default False
	#  - dis_type = type of distance metric to compute. Options:
	#    * 'euclidean' - default, euclidean norm of adjacency matrices (after matching dimensions)
	#    * 'ji_all_genes' - jaccard index of original adjacency matrices
	#    * 'ji_top_edges' - jaccard index of edges with weeigths above t quantile(default 0.5)
	#    * 'lp_spectral' - euclidean norm of all eigenvalues of laplacian matrices of networks
	#    * 'lp_egvectors' - euclidean norm of first eigenvector of laplacian matrices of networks
	#    * 'induced2norm' - induced two norm of adjacency matrices

	nc = len(networks)
	dist = pd.DataFrame(0, index=range(0, nc), columns=range(0, nc), dtype=np.float64)

	for i in range(0, nc):
		for j in range(i, nc):

			m1, m2 = copy.copy(networks[i]), copy.copy(networks[j])
			w1, w2 = equal_dimentions([m1, m2])

			if dis_type == 'euclidean':
				dist.at[i, j] = np.linalg.norm(w1 - w2)
				dist.at[j, i] = dist.at[i, j]

	if resc:
		dist = rescale(dist)

	dist = sk.DistanceMatrix(dist.values)
	
	return dist
