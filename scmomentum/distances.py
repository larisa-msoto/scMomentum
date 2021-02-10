import copy as cp
import numpy as np
import pandas as pd
import random
import scvelo as scv
import skbio as sk
from sklearn import preprocessing
from scmomentum.utilities import unique
import networkx as nx

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

	for mat in cluster_networks:

		l = [g not in mat.columns.values.tolist() for g in allgenes]
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

	dist = sk.DistanceMatrix(dist.values).to_data_frame()
	dist.columns,dist.index = clusters,clusters

	if copy:
		return dist
	else:
		adata.uns["expression_distances"] = dist
		print('--> Updated adata.uns with key ','\'expression_distances\'')
		
def network_distance(adata,net_type,resc=True,dis_type='euclidean',copy=False):

	# Inputs:
	#  - cluter_mats = list with two network adjacency matrices (each a data frame with gene names as index and columns)
	#  - resc = bool, wether to rescale the distances using min to max scaling, default False
	#  - dis_type = type of distance metric to compute. Options:
	#    * 'euclidean' - default, euclidean norm of adjacency matrices (after matching dimensions)

	clusters = adata.uns[net_type].keys()
	networks = [adata.uns[net_type][key] for key in clusters]

	nc = len(networks)
	dist = pd.DataFrame(0, index=range(0, nc), columns=range(0, nc), dtype=np.float64)
	#dist = pd.DataFrame(0, index=clusters, columns=clusters, dtype=np.float64)

	for i in range(0, nc):
		for j in range(i, nc):

			m1, m2 = cp.copy(networks[i]), cp.copy(networks[j])
			w1, w2 = equal_dimentions([m1, m2])

			if dis_type == 'euclidean':
				dist.at[i, j] = np.linalg.norm(w1 - w2)
				dist.at[j, i] = dist.at[i, j]

	if resc:
		dist = rescale(dist)

	dist = sk.DistanceMatrix(dist.values).to_data_frame()
	dist.columns,dist.index = clusters,clusters

	if copy:
		return dist
	else:
		adata.uns["network_distances"] = dist
		print('--> Updated adata.uns with key ','\'network_distances\'')
		

def jaccard_index(gset1, gset2):
    
    inter = len(list(set(gset1).intersection(gset2)))
    union = (len(gset1) + len(gset2)) - inter
    ji = float(inter)/union

    return ji

def network_jaccard(w1,w2,t=0.5):
    
    
    # Description:
    # First all edges with weigth above t are selected in each network separately. 
    # Then the jaccard index is computed for that filtered set of edges.
    # Input:
    #  - w1 = numpy array, network 1
    #  - w2 = numpy array, network 2
    #  - t = minimum threshold value to filter weights in both networks
    # Returns:
    #  - jaccard index of edges with weights above t in both networks
   
    n1 = np.quantile(w1,t)
    n2 = np.quantile(w2,t)
    
    G1,G2 = nx.from_numpy_matrix(w1,create_using=nx.DiGraph()),nx.from_numpy_matrix(w2,create_using=nx.DiGraph())
    g1_top = [(u,v) for (u,v,d) in G1.edges(data=True) if d['weight'] >= n1]
    g2_top = [(u,v) for (u,v,d) in G2.edges(data=True) if d['weight'] >= n2]
    
    ji_edges_top = jaccard_index(g1_top,g2_top)

    return ji_edges_top














