import time
import os
import csv
import h5py
import numpy as np
import crystallography as xtal
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
import click
from Graph import GraphSet
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
from scipy.stats import ks_2samp as Kolmogorov_Smirnov
from scipy.stats import mode,gaussian_kde
from scipy.sparse.csgraph import laplacian
from numpy.linalg import eig

def graph_difference(X,Y,n):
	'''Compute Laplacian Matrices'''
	X,Y = -1*X,-1*Y
	for i in range(len(X)):
		''' assumes symmetric adjacency matrix '''
		X[i,i] = np.sum(np.abs(X[:,i]))
		Y[i,i] = np.sum(np.abs(Y[:,i]))
		
	X_w,X_v = np.linalg.eig(X)
	Y_w,Y_v = np.linalg.eig(Y)
	return np.abs(np.sort([x for x in X_w if not x == 0])[1] -\
		   np.sort([x for x in X_w if not x == 0])[1])
	
def compute_dist_matrix(X,metric):
	dist_matrix = np.zeros((len(X),len(X)))
	for i in range(len(X)):
		print('\r{}/{}'.format(i+1,len(X)),end='')
		for j in range(i+1,len(X)):
			dist = metric(X[i],X[j])
			dist_matrix[i,j] = dist
			dist_matrix[j,i] = dist

	return dist_matrix

def load_dataset(filepath,clustered = False,symmetry=xtal.Symmetry('Cubic'),synthetic=True):
    f = h5py.File(filepath,'r')['DataContainers']
    if not synthetic:
        data = f['ImageDataContainer']['CellFeatureFeatureData']
    else:
        data = f['SyntheticVolumeDataContainer']['CellFeatureData']
    neighbor_list = data['NeighborList']
    num_neighbors = data['NumNeighbors']
    try:
        quats = data['Quats']
    except KeyError:
        quats = data['AvgQuats']
    volume = data['Volumes']
    diameters = data['EquivalentDiameters']
    surface_features = None
    centroids = data['Centroids']
    if clustered:
        cluster_ids = data['ClusteringList']
    else:
        cluster_ids = None

    return GraphSet(neighbor_list,num_neighbors,quats,volume,
           diameters,surface_features,centroids,cluster_ids,symmetry)

def twin_distances(graph):
	twins = graph.neighbor_list[graph.sigma_3_list]
	parents = np.searchsorted(graph.node_indices,graph.sigma_3_list,side='right')-1
	twin_centroids = graph.centroids[twins]
	parent_centroids = graph.centroids[parents]
	distances = []
	for i in range(len(twin_centroids)):
		dist = np.sqrt(np.sum(np.square(twin_centroids[i]-parent_centroids[i])))
		if not dist == 0.0:
			distances.append(dist)
	return np.array(distances)	
	

def get_spectral_features(adj_matr):
	laplac_matr = laplacian(adj_matr)
	eigenvalues = np.sort(eig(adj_matr)[0])
	laplac_eigenvalues = np.sort(eig(laplac_matr)[0])
	return np.concatenate((eigenvalues,laplac_eigenvalues))

def attribute_matrices(walk):
	Synthetic = load_dataset('../rene88/final-dataset.dream3d',synthetic=True)
	Experimental = load_dataset('../rene88/final_dataset.dream3d',synthetic=False)
	Synthetic.set_node_features([Synthetic.volume])
	Experimental.set_node_features([Experimental.volume])
	Synthetic.set_edge_features([Synthetic.misori_angle_list])
	Experimental.set_edge_features([Experimental.misori_angle_list])
	
	iters=1000
	synths = []
	expers = []
	for a in range(iters):
		Synth = Synthetic.random_walk(walk)
		Exper = Experimental.random_walk(walk)
		synth = np.zeros(Synth.shape)
		exper = np.zeros(Exper.shape)
		for i in range(len(Synth)):
			for j in range(len(Synth[i])):
				synth[i] = int(Synth[i,j].get(0) is not None)
				exper[i] = int(Exper[i,j].get(0) is not None)
		
		synths.append(get_spectral_features(synth))
		expers.append(get_spectral_features(exper))
		print('\r{}/{}'.format(a+1,iters),end='')

	synths,expers = np.array(synths),np.array(expers)
					
	print('\nDone.')
	'''
	data = np.concatenate((synths,expers),axis=0)
	kmeans = KMeans(n_clusters=2,n_init=500,max_iter=5000)
	
	print('KMeans...')
	kmeans.fit(data)
	exp_mode = int(mode(kmeans.predict(expers))[0][0])
	synth_cluster = kmeans.transform(synths)[:,exp_mode]
	exper_cluster = kmeans.transform(expers)[:,exp_mode]

	p = Kolmogorov_Smirnov(synth_cluster,exper_cluster)[1]
	print('p={} for N: The Experimental and Synthetic spectral features are not distinctly clustered.'.format(p))
	ps = [p]
	'''
	ps = []
	for feature in range(synths.shape[1]):
		synth_features = synths[:,feature]
		exper_features = expers[:,feature]
		p = Kolmogorov_Smirnov(synth_features,exper_features)[1]
		print('p={} for N: Spectral Feature {} are the same for synthetic and experimental'.format(p,feature))
		ps.append(p)
	ps = np.array(ps)
	return np.round((1-ps)*100,4)

def walk_distributions(walk,Synthetic,Experimental):
	ps = []
	#voxels_to_microns = FACTOR
	''' FILL THIS IN '''
	Synthetic.set_node_features([Synthetic.volume,Synthetic.quats])
	Experimental.set_node_features([Experimental.volume,Experimental.quats])
	Synthetic.set_edge_features([Synthetic.misori_angle_list,
								 Synthetic.distance_list,
								 Synthetic.misori_list])	
	Experimental.set_edge_features([Experimental.misori_angle_list,
									Experimental.distance_list,
									Experimental.misori_list])
	
	''' Plot Twinned Centroid Distances '''
	colors = ['green','blue']
	plt.style.use('ggplot')
	matplotlib.rc('font', family = 'serif', serif = 'cmr10',size=20)
	matplotlib.rcParams.update({'axes.titlepad': 20})
	matplotlib.rcParams.update({'axes.labelpad': 20})
	exp_distances = twin_distances(Experimental)
	synth_distances = twin_distances(Synthetic)
	synth_distances = synth_distances
	exp_distances = exp_distances
	bins = np.linspace(min(min(synth_distances),min(exp_distances)),\
					   max(max(synth_distances),max(exp_distances)),\
					   int(np.sqrt(max(len(synth_distances),len(exp_distances)))))
	
	fig = plt.figure(0,figsize=(16,16),dpi=600)
	plt.hist(synth_distances,bins,alpha=0.4,
			 label='Synthetic',normed=True,facecolor='b')
	plt.hist(exp_distances,bins,alpha=0.4,
			 label='Experimental',normed=True,facecolor='g')

	exp_density = gaussian_kde(exp_distances)
	synth_density = gaussian_kde(synth_distances)
	
	xs = np.linspace(min(min(synth_distances),min(exp_distances)),\
					 max(max(synth_distances),max(exp_distances)),\
					 500)
	
	plt.plot(xs,exp_density.pdf(xs),color='g',alpha=0.7)
	plt.plot(xs,synth_density.pdf(xs),color='b',alpha=0.7)
	plt.title('Distance between Twin-Related Grain Centroids')
	plt.xlabel('Centroid Distance (voxels)')
	plt.ylabel('Frequency')
	plt.legend(loc='upper right')
	fig.savefig('figures/centroid_distances.pdf')
	plt.clf()
	
	synth_volumes, exp_volumes = [],[]
	synth_angles, exp_angles = [],[]	
	synth_distances, exp_distances = [],[]
	synth_misori, exp_misori = [],[]

	print('Computing Random Walks...')
	n = 1000
	for i in range(n):
		print('\r{}/{}'.format(i+1,n),end="")
		x = Synthetic.random_walk_stats(n=walk)
		synth_nodes,synth_edges = x[0],x[1]
		x = Experimental.random_walk_stats(n=walk)
		exp_nodes,exp_edges = x[0],x[1]
		for j in range(len(synth_nodes[0])):
			if not synth_nodes[0][j] == 0.0:
				synth_volumes.append(synth_nodes[0][j])
		
		for j in range(len(exp_nodes[0])):
			if not exp_nodes[0][j] == 0.0:
				exp_volumes.append(exp_nodes[0][j])

		[synth_angles.append(angle) for angle in synth_edges[0]]
		[synth_distances.append(distance) for distance in synth_edges[1] if not distance == 0.0]
		[synth_misori.append(misori) for misori in synth_edges[2]]
	
		[exp_angles.append(angle) for angle in exp_edges[0]]
		[exp_distances.append(distance) for distance in exp_edges[1] if not distance == 0.0]
		[exp_misori.append(misori) for misori in exp_edges[2]]

	 

	synth_angles,exp_angles = np.array(synth_angles).flatten(),\
							  np.array(exp_angles).flatten()
	synth_distances,exp_distances = np.array(synth_distances).flatten(),\
									np.array(exp_distances).flatten()
	synth_misori,exp_misori = np.array(synth_misori).reshape(-1,4),\
							  np.array(exp_misori).reshape(-1,4)
	
	''' Plot Random Walk Angles '''
	minimum = 0
	maximum = 65
	bins = np.linspace(minimum,maximum,50)
	
	fig = plt.figure(0,figsize=(16,16),dpi=600)
	sx,sbins,sptch = plt.hist(synth_angles,bins,alpha=0.4,
							  label='Synthetic',normed=True,facecolor='b')
	ex,ebins,eptch = plt.hist(exp_angles,bins,alpha=0.4,
							  label='Experimental',normed=True,facecolor='g')

	exp_density = gaussian_kde(exp_angles)
	synth_density = gaussian_kde(synth_angles)
	
	xs = np.linspace(0,65,500)
	
	plt.plot(xs,exp_density.pdf(xs),color='g',alpha=0.7)
	plt.plot(xs,synth_density.pdf(xs),color='b',alpha=0.7)
	plt.xlabel('Misorientation Angle')
	plt.ylabel('Frequency')
	plt.title('Misorientation Angles in Random Walk of Length {} from Twin-Related Grains'.format(walk))
	plt.legend(loc='upper left')
	fig.savefig('figures/random_walk_misori_angles_n={}.pdf'.format(walk))
	plt.clf()

	p = Kolmogorov_Smirnov(synth_angles,exp_angles)[1]
	print('\np={} for N: Random Walk Misorientations are from same distribution'.format(p))
	ps.append(p)
	
	angles_local = exp_angles
	angles_global = Experimental.misori_angle_list

	angles_local = np.array([angle for angle in angles_local if not angle == 0.0])
	angles_global = np.array([angle for angle in angles_global if not angle == 0.0])

	minimum = 0.0
	maximum = 65.0
	bins = np.linspace(minimum,maximum,50)
	fig = plt.figure(0,figsize=(16,16),dpi=600)
	sx,sbins,sptch = plt.hist(angles_local,bins,alpha=0.4,
							  label='Local',normed=True,facecolor='C1')
	ex,ebins,eptch = plt.hist(angles_global,bins,alpha=0.4,
							  label='Global',normed=True,facecolor='r')
	global_density = gaussian_kde(angles_global)
	local_density = gaussian_kde(angles_local)
		
	plt.plot(xs,global_density.pdf(xs),color='r',alpha=0.7)
	plt.plot(xs,local_density.pdf(xs),color='C1',alpha=0.7)
	plt.xlabel('Misorientation Angle')
	plt.ylabel('Frequency')
	plt.title('Experimental Misorientation Distribution - Global vs. Around Twin-Related Grains')
	plt.legend(loc='upper left')
	fig.savefig('figures/global_local_misori_angles_experimental_n={}.pdf'.format(walk))
	plt.clf()
	p_global_local = [Kolmogorov_Smirnov(angles_local,angles_global)[1]]

	angles_local = synth_angles
	angles_global = Synthetic.misori_angle_list

	angles_local = np.array([angle for angle in angles_local if not angle == 0.0])
	angles_global = np.array([angle for angle in angles_global if not angle == 0.0])

	minimum = 0.0
	maximum = 65.0
	bins = np.linspace(minimum,maximum,50)
	fig = plt.figure(0,figsize=(16,16),dpi=600)
	sx,sbins,sptch = plt.hist(angles_local,bins,alpha=0.4,
							  label='Local',normed=True,facecolor='C1')
	ex,ebins,eptch = plt.hist(angles_global,bins,alpha=0.4,
							  label='Global',normed=True,facecolor='r')

	local_density = gaussian_kde(angles_local)
	global_density = gaussian_kde(angles_global)

	plt.plot(xs,local_density.pdf(xs),color='C1',alpha=0.7)
	plt.plot(xs,global_density.pdf(xs),color='r',alpha=0.7)
	
	plt.xlabel('Misorientation Angle')
	plt.ylabel('Frequency')
	plt.title('Synthetic Misorientation Distribution - Global vs. Around Twin-Related Grains')
	plt.legend(loc='upper left')
	fig.savefig('figures/global_local_misori_angles_synthetic_n={}.pdf'.format(walk))
	plt.clf()
	p_global_local.append(Kolmogorov_Smirnov(angles_local,angles_global)[1])
	print('p={} for N: Global and Local Misorientation Angles are from same distribution'.format(p_global_local))

	
	''' Plot Random Walk Distances '''
	minimum = min(min(exp_distances),min(synth_distances))
	maximum = max(max(exp_distances),max(synth_distances))
	print('Minimum Walk Distance: {} Maximum Walk Distance: {}'.format(minimum,maximum))
	bins = np.linspace(minimum,maximum,100)

	fig = plt.figure(0,figsize=(16,16),dpi=600)	
	sx,sbins,sptch = plt.hist(synth_distances,bins,normed=True,alpha=0.4,facecolor='b',
							  label='Synthetic')
	ex,ebins,eptch = plt.hist(exp_distances,bins,normed=True,alpha=0.4,facecolor='g',
							  label='Experimental')
	
	xs = np.linspace(minimum,maximum,500)
	
	exp_density = gaussian_kde(exp_distances)
	synth_density = gaussian_kde(synth_distances)
	
	plt.plot(xs,exp_density.pdf(xs),color='g',alpha=0.7)
	plt.plot(xs,synth_density.pdf(xs),color='b',alpha=0.7)

	plt.title('Centroid Distances in Random Walk of Length {} from Twins'.format(walk))
	plt.xlabel('Distance between Grain Centroids (microns)')
	plt.ylabel('Frequency')
	plt.legend(loc='upper left')
	fig.savefig('figures/random_walk_centroid_distances_n={}.pdf'.format(walk))
	plt.clf()
	p = Kolmogorov_Smirnov(synth_distances,exp_distances)[1]
	print('p={} for N: Random Walk Distances are from same distribution'.format(p))
	ps.append(p)
	print('\nDone.')
	return (1-np.array(ps))*100,(1-np.array(p_global_local))*100

@click.command()
@click.argument('min_walk',type=int)
@click.argument('max_walk',type=int)
def main(min_walk,max_walk):
	walks = np.array(np.arange(min_walk,max_walk),dtype=int)
	Synthetic = load_dataset('rene88/GAN_input.dream3d',synthetic=True)
	Experimental = load_dataset('rene88/final_dataset.dream3d',synthetic=False)
	with open('results.csv','w') as f:
		writer = csv.writer(f, delimiter=',')
		header = ['Number of Steps','Global vs. Local Misorientation Experimental KS-Test',\
				  'Global vs. Local Misorientation Synthetic KS-Test',\
				  'Angle Distribution KS-Test','Centroid Distance Distribution KS-Test']
		writer.writerow(header)
		# Perform random walks
		for walk in walks:
			print('Performing random walk of length n={}'.format(walk))
			#spectral_tests = attribute_matrices(walk)
			#spectral_tests = np.concatenate((spectral_tests,np.zeros(max_walk*2-len(spectral_tests))))
			distr_tests,p_global_local = walk_distributions(walk,Synthetic,Experimental)
			row = list(np.concatenate(([walk],p_global_local,distr_tests)))
			writer.writerow(row)
			os.system('clear')

if __name__ == '__main__':
	main()
