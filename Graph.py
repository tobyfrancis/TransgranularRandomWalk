import h5py
import crystallography as xtal
import numpy as np
import warnings
import math
class Feature:
	def __init__(self):
		self.features = []
		self.feature_lengths = []
		self.length = 0

	def append(self,feature):
		try:
			self.feature_lengths.append(len(feature))
			self.features.append(np.array(feature))
		except TypeError:
			self.feature_lengths.append(1)
			self.features.append(np.array([feature]))

		self.length = self.length + 1

	def get(self,index):
		try:
			return self.features[index]
		except IndexError:
			pass


class GraphSet:
	def __init__(self,neighbor_list,num_neighbors,quats,volume,
				 diameters,surface_features,centroids,cluster_ids,symmetry):
		self.neighbor_list = np.array(neighbor_list)
		self.num_neighbors = np.array(num_neighbors,dtype=int)
		self.quats = np.roll(np.array(quats),-1,1)
		self.diameters = np.array(diameters)
		self.volume = np.array(volume)
		self.surface_features = np.array(surface_features)
		self.centroids = np.array(centroids)
		if cluster_ids:
			self.cluster_ids = np.array(cluster_ids)
		self.symmetry = symmetry
	
		print('Computing Neighbor List Statistics...')
		self.node_indices = np.cumsum(self.num_neighbors)[:-1]
		self.node_indices = np.insert(self.node_indices,0,0)
		print('Computing Misorientation Neighbor List...')
		self.misori_list,self.sigma_3_list,self.misori_angle_list = self.construct_misori_list()
		self.distance_list = self.construct_distance_list()
		
		self.node_features = []
		self.edge_features = []

	def set_node_features(self,array):
		self.node_features = array
		
	def set_edge_features(self,array):
		self.edge_features = array		
	
	def within_tolerance(self,a,b,tolerance):
		try:
			if not (a.shape == b.shape):
				raise Exception('Tolerance function only takes arrays of same shape')
			elif len(a.shape)>2:
				raise Exception('Tolerance function implemented only for up to 1D arrays')
		except AttributeError:
				pass
		try:
			if any(math.isnan(a)) or any(np.isnan(a)):
				print(a)
				raise Exception
			if any(math.isnan(b)) or any(np.isnan(b)):
				print(b)
				raise Exception
			try:
				return all(np.less(np.abs(a-b),tolerance))
			except ValueError:
				print(a,b)
		except TypeError:
			return np.abs(a-b) < tolerance

	def get_angle(self,misori):
		misori = xtal.qu2ax(xtal.do2qu(misori))
		
		if np.isnan(misori[0,3]):
			return 0.0
			
		return misori[0,3]*(180/np.pi)

	def construct_misori_list(self):
		node_indices = self.node_indices
		neighbor_list = self.neighbor_list
		edge_indices = np.arange(len(neighbor_list))
		num_neighbors = self.num_neighbors
		quats = self.quats
			
		symmetry = self.symmetry
		sigma3 = 60 * (np.pi/180)
		ijk_tolerance = 0.1
		angle_tolerance = 2.0 * (np.pi/180)

		def misori(edge_index):
			search_index = np.searchsorted(node_indices,edge_index)
			start_node_feature = search_index-1
			end_node_feature = neighbor_list[edge_index]
			quat_1 = xtal.do2qu(quats[start_node_feature])
			quat_2 = xtal.do2qu(quats[end_node_feature])

			disori = xtal.qu2do(symmetry.disoQu(quat_1,quat_2))
			return disori 

		def is_sigma_3(edge_index):
			search_index = np.searchsorted(node_indices,edge_index)
			start_node_feature = search_index-1
			end_node_feature = neighbor_list[edge_index]
			quat_1 = xtal.do2qu(quats[start_node_feature])
			quat_2 = xtal.do2qu(quats[end_node_feature])

			disori = xtal.qu2ax(symmetry.disoQu(quat_1,quat_2))
			ax 	= 3*(disori[0,:3]/np.sum(np.abs(disori[0,:3])))
			angle = disori[0,3]
			
			if any(np.isnan(ax)):
				return False	
			
			ax =  all(np.less(np.abs(ax-np.ones(3)),ijk_tolerance))
			angle = np.abs(angle-sigma3)<angle_tolerance
			return ax and angle
		
		def misori_angle(edge_index):
			search_index = np.searchsorted(node_indices,edge_index)
			start_node_feature = search_index-1
			end_node_feature = neighbor_list[edge_index]
			quat_1 = xtal.do2qu(quats[start_node_feature])
			quat_2 = xtal.do2qu(quats[end_node_feature])

			disori = xtal.qu2ax(symmetry.disoQu(quat_1,quat_2))
			angle = disori[0,3]*(180/np.pi)
		
			if np.isnan(angle):
				return 0.0	
			
			return 	angle
	
		misori_list = [misori(edge_index) for edge_index in edge_indices]
		sigma_3_list = [is_sigma_3(edge_index) for edge_index in edge_indices]
		misori_angle_list = [misori_angle(edge_index) for edge_index in edge_indices]
		return np.array(misori_list).reshape(-1,4),np.where(sigma_3_list)[0],np.array(misori_angle_list).flatten()

	def construct_distance_list(self):
		node_indices = self.node_indices
		neighbor_list = self.neighbor_list
		centroids = self.centroids
		neighbor_iter = np.arange(len(neighbor_list))
		
		def get_distance(index):
			feature_2 = neighbor_list[index]
			feature_1 = np.searchsorted(node_indices,index)

			if feature_1 == len(node_indices):
				feature_1 = feature_1-1

			elif not node_indices[feature_1] == index:
				feature_1 = feature_1 - 1

			feature_1 = np.searchsorted(node_indices,index)-1
			dist = centroids[feature_2]-centroids[feature_1]
			dist = np.square(np.sum(np.square(dist)))
			return dist
	
		return np.array(list(map(get_distance,neighbor_iter)))

	def random_walk(self,n,
					in_cluster=False,exclude_surface=False,intersecting=False):
		nodes = []
		adj_matrix = np.empty((n,n),dtype=Feature)
		vFeature = np.vectorize(Feature)
		adj_matrix[:,:] = vFeature()
		
		neighbor_list = self.neighbor_list
		num_neighbors = self.num_neighbors
		node_indices = self.node_indices

		#edge_index = np.random.choice(self.sigma_3_list)
		edge_index = np.random.choice(np.arange(len(neighbor_list)))
		edge = neighbor_list[edge_index]
		node = np.searchsorted(node_indices,edge_index)-1
			
		for feature in self.node_features:
			adj_matrix[0,0].append(feature[node])
		
		for feature in self.edge_features:
			adj_matrix[0,1].append(feature[edge_index])
			adj_matrix[1,0].append(feature[edge_index])
		
		if in_cluster:
			cluster_id = self.cluster_ids[start_node]

		nodes.append(node)
			
		for i in range(1,n-1):
			''' set node to edge '''
			node = edge
			for feature in self.node_features:
				adj_matrix[i,i].append(feature[node])
			
			''' get possible edges from neigh bor list '''
			choices = np.arange(node_indices[node],node_indices[node]+num_neighbors[node])
			feature_choices = neighbor_list[choices]			
			
			probs = np.ones(len(choices))

			''' check for connections with previous edges in the walk '''

			connections = [np.where(np.array(nodes) == node) for node in feature_choices]
			for h,connects in enumerate(connections):
				edge_index = choices[h]
				for j in connects[0]:
					if adj_matrix[i,j].length == 0:
						adj_matrix[i,j].append(feature[edge_index])
						adj_matrix[j,i].append(feature[edge_index])
			
			''' apply constraints to edge probabilities '''

			if not intersecting:
				#restricts possible edges to those not already encountered
				probs = np.array([len(np.where(np.array(nodes) == node)[0]) == 0 for node in feature_choices],dtype=int)
		
			if in_cluster:
				#restricts possible edges to those in the same twin cluster as starting node
				incluster = np.array(self.cluster_ids[feature_choices] == \
							cluster_id,dtype=int).flatten()
				probs = np.logical_and(probs,incluster)
				
			if exclude_surface:
				#restricts possible edges to those that don't lead to the surface
				probs = np.logical_and(probs,np.logical_not(\
					    np.array(self.surface_features[feature_choices]).flatten()))
			
			''' if none of the edge choices satisfiy conditions, re-generate random walk '''
			if np.sum(probs) == 0.0:
				return self.random_walk(n,in_cluster=in_cluster,
				       exclude_surface=exclude_surface)
			else:
				probs = probs/np.sum(probs)
			
			''' pick edge, add it to adjacency matrix '''

			edge_index = np.random.choice(choices,p=probs)
			edge = neighbor_list[edge_index]
			
			for feature in self.edge_features:
				adj_matrix[i,i+1].append(feature[edge_index])
				adj_matrix[i+1,i].append(feature[edge_index])
			
			nodes.append(node)
			
		''' add final node '''
		node = edge
		for feature in self.node_features:
			adj_matrix[n-1,n-1].append(feature[node])
		
		return adj_matrix

	def random_walk_stats(self,n,
					in_cluster=False,exclude_surface=False,intersecting=False):
		
		nodes = []
		node_feats = [[] for _ in range(len(self.node_features))]
		edge_feats = [[] for _ in range(len(self.edge_features))]
		neighbor_list = self.neighbor_list
		num_neighbors = self.num_neighbors
		node_indices = self.node_indices

		edge_index = np.random.choice(self.sigma_3_list)
		#edge_index = np.random.choice(np.arange(len(neighbor_list)))
		edge = neighbor_list[edge_index]
		node = np.searchsorted(node_indices,edge_index)-1

		for index,feature in enumerate(self.node_features):
			try:
				length = len(feature[node])
				if length == 1:
					node_feats[index].append(feature[node][0])
				else:
					node_feats[index].append(feature[node])
			except TypeError:
				node_feats[index].append(feature[node])
		for index,feature in enumerate(self.edge_features):
			try:
				length = len(feature[node])
				if length == 1 and not(feature[edge_index][0] == 0.0):
					edge_feats[index].append(feature[edge_index][0])
				elif not all(feature[node]==0.0):
					edge_feats[index].append(feature[edge_index])
			except TypeError:
				if not feature[node] == 0.0:
					edge_feats[index].append(feature[node])
		if in_cluster:
			cluster_id = self.cluster_ids[start_node]
		
		nodes.append(node)
			
		for i in range(1,n-1):
			''' set node to edge '''
			node = edge

			for index,feature in enumerate(self.node_features):
				try:
					length = len(feature[node])
					if length == 1:
						node_feats[index].append(feature[node][0])
					else:
						node_feats[index].append(feature[node])
				except TypeError:
					node_feats[index].append(feature[node])


			''' get possible edges from neighbor list '''
			choices = np.arange(node_indices[node],node_indices[node]+num_neighbors[node])
			feature_choices = neighbor_list[choices]			
			
			probs = np.ones(len(choices))

			''' check for connections with previous edges in the walk '''

			connections = [np.where(np.array(nodes) == node) for node in feature_choices]
			for h,connects in enumerate(connections):
				edge_index = choices[h]
				for index,feature in enumerate(self.edge_features):
					if (not (np.array(feature[edge_index]) == np.array(edge_feats[index])).any()) and\
						(not all(np.in1d(feature[edge_index],0.0))):
						edge_feats[index].append(feature[edge_index])	

			''' apply constraints to edge probabilities '''

			if not intersecting:
				#restricts possible edges to those not already encountered
				probs = np.array([len(np.where(np.array(nodes) == node)[0]) == 0 for node in feature_choices],dtype=int)
		
			if in_cluster:
				#restricts possible edges to those in the same twin cluster as starting node
				incluster = np.array(self.cluster_ids[feature_choices] == \
							cluster_id,dtype=int).flatten()
				probs = np.logical_and(probs,incluster)
				
			if exclude_surface:
				#restricts possible edges to those that don't lead to the surface
				probs = np.logical_and(probs,np.logical_not(\
					    np.array(self.surface_features[feature_choices]).flatten()))
			
			''' if none of the edge choices work, re-generate random walk '''
			if np.sum(probs) == 0.0:
				return self.random_walk_stats(n,in_cluster=in_cluster,
				       exclude_surface=exclude_surface)
			else:
				probs = probs/np.sum(probs)
			
			''' pick edge, add it to edges '''

			edge_index = np.random.choice(choices,p=probs)
			edge = neighbor_list[edge_index]
			
			for index,feature in enumerate(self.edge_features):
				edge_feats[index].append(feature[edge_index])
	
			nodes.append(node)
			
		''' add final node '''
		node = edge
		for index,feature in enumerate(self.node_features):
			node_feats[index].append(feature[node])	
	
		return node_feats,edge_feats


	def generate_full_twin_network(self,n,in_cluster=True):
		#ToDo
		pass

	def vtk_random_twin_network(self,cutoff):
		#ToDo
		pass
	
		
		


# helper class for cluster connectivity calculations
class UndirectedGraph:
	# create a graph object from a Neighborlist
	def __init__(self,NeighborList,GrainFeatures):
		self.adjacency_matrix = self.construct_adjacency_matrix(NeighborList,GrainFeatures)
		self.edges = self.construct_edge_list(self.adjacency_matrix)
		self.num_nodes = self.adjacency_matrix.shape[0]

	# computes the minimum number of edges to connect each node to the root node
	def computeDistance(self, root):
		if self.numNodes is 0:
			return []
		distance = [-1 for i in range(self.numNodes)]
		distance[root] = 0
		changed = True
		while changed:
			changed = False
			for e in self.edges:
				d0 = distance[e[0]]
				d1 = distance[e[1]]
				if d0 < 0 and d1 >= 0: # e0 unassigned, e1 assigned
					distance[e[0]] = d1+1
					changed = True
				elif d1 < 0 and d0 >= 0: # e0 assigned, e1 unassigned
					distance[e[1]] = d0+1
					changed = True
				elif d0 >= 0 and d1 >= 0: # e0 assigned, e1 assigned
					if d0+1 < d1:
						distance[e[1]] = d0+1
						changed = True
					elif d1+1 < d0:
						distance[e[0]] = d1+1
						changed = True
				else: # e0 unassigned, e1 unassigned
					pass
		if min(distance) < 0:
			print('warning, isolated nodes exist')
		return distance

	def distanceBetween(self, n1, n2):
		return self.computeDistance(n1)[n2]

	# computes the root(s) to minimize the maximum distance from the root to any other node
	def findMinRoot(self):
		if self.numNodes is 0:
			return None
		root = [0]
		maxDist = max(self.computeDistance(0))
		for i in range(1, self.numNodes):
			dist = max(self.computeDistance(i))
			if dist < maxDist:
				root = [i]
				maxDist = dist
			elif dist == maxDist:
				root.append(i)
		return root

	# computes the root(s) to maximize the maximum distance from the root to any other node
	def findMaxRoot(self):
		if self.numNodes is 0:
			return None
		root = [0]
		maxDist = max(self.computeDistance(0))
		for i in range(1, self.numNodes):
			dist = max(self.computeDistance(i))
			if dist > maxDist:
				root = [i]
				maxDist = dist
			elif dist == maxDist:
				root.append(i)
		return root

	def maxDist(self):
		if self.numNodes is 0:
			return None
		return max(self.computeDistance(self.findMaxRoot()[0]))
