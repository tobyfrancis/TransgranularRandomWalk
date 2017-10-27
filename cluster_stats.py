import h5py
import numpy as np
import colorsys
import rotations
from symmetry import Symmetry
from quaternion import Quaternion

# helper class for cluster connectivity calculations
class UndirectedGraph:
	# create a network object from a list of edges (pairs of connected nodes)
	def __init__(self, edges):
		# build list of nodes
		nodes = set()
		for c in edges:
			nodes.add(c[0])
			nodes.add(c[1])
		nodes = list(nodes)
		self.edges = [(nodes.index(e[0]), nodes.index(e[1])) for e in edges] # handle spare node labeling
		self.numNodes = len(nodes)

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


f = h5py.File('5_5_cluster_data.dream3d', 'r')

# get largest interior twin cluster
voxels_interior = f['DataContainers/ImageDataContainer/ClusterData/NumCells'][:] * (1-f['DataContainers/ImageDataContainer/ClusterData/SurfaceFeatures'][:])
clusterId = np.argsort(voxels_interior[:,0])
internalClusters = [i for i in clusterId if voxels_interior[i] > 0]
print(len(internalClusters))

clusterCount = 0
with open('cluster_stats2.csv', 'w') as outfile:
	outfile.write('clusterId, numGrains, numVariants, twinArea, totalArea, clusterVolume, maxVolume, minDist, maxDist, numGenerations, numLines, numVarLines\n')
	for clusterId in internalClusters:
	# # clusterId = clusterId[-2]
	# clusterId = clusterId[-50]
	# clusterId = 769
	# print(clusterId)

		# get grains belonging to cluster
		clusterGrains = np.nonzero(f['DataContainers/ImageDataContainer/CellFeatureData/ClusterIds'][:]==clusterId)[0]
		if 1 is len(clusterGrains):
			continue
		clusterCount += 1

		# get orientations of grains in cluster and convert from xyzw to wxyz
		quats = np.roll(f['DataContainers/ImageDataContainer/CellFeatureData/AvgQuats'][:][clusterGrains], 1, axis = 1)

		# get sizes of grains in cluster
		diams = f['DataContainers/ImageDataContainer/CellFeatureData/EquivalentDiameters'][:][clusterGrains]

		# get neighborss of grains in cluster
		neighbors = f['DataContainers/ImageDataContainer/CellFeatureData/NeighborList']
		neighborCount = f['DataContainers/ImageDataContainer/CellFeatureData/NumNeighbors'][:,0]
		neighborInds = np.copy(neighborCount)
		for i in range(1, len(neighborInds)):
			neighborInds[i] += neighborInds[i-1]
		neighborInds -= neighborCount

		# get shared surface areas
		sharedAreas = f['DataContainers/ImageDataContainer/CellFeatureData/SharedSurfaceAreaList']

		# build list of neighbors within cluster
		lines = []
		areas = []
		for i in range(clusterGrains.shape[0]):
			grain = clusterGrains[i]
			offset = neighborInds[grain]
			count = neighborCount[grain]
			grainNeighbors = neighbors[offset:offset+count]
			for n in range(len(grainNeighbors)):
				ind = np.argwhere(clusterGrains == grainNeighbors[n])[:,0]
				if ind.shape[0] > 0:
					lines.append((i,ind[0]))
					areas.append(sharedAreas[offset+n])

		# segment cluster into variants
		variants = np.zeros_like(clusterGrains)
		variantId = 1
		for i in range(clusterGrains.shape[0]):
			for j in range(i):
				miso = Symmetry.Cubic.disoQuat(quats[i], quats[j])
				if 2*np.math.acos(min(miso[0],1))*180/np.pi <= 5: # within 5 degrees
					variants[i] = variants[j]
					break
			if 0 == variants[i]:
				variants[i] = variantId
				variantId += 1
		variantId -= 1
		variants -= 1

		# move quats so that they are nearby eachother
		fzQuats = [Symmetry.Cubic.fzQu(Quaternion(q)) for q in quats]
		baseQuats = [None] * variantId
		for i in range(len(variants)):
			for j in range(variantId):
				if baseQuats[variants[j]] is None:
					baseQuats[variants[j]] = Quaternion(fzQuats[i])
				else:
					fzQuats[i] = Symmetry.Cubic.nearbyQuat(baseQuats[variants[j]], fzQuats[i])

		# compute volume weighted average orientation of each variant
		grainCus = np.array([rotations.qu2cu(Symmetry.Cubic.fzQu(q.wxyz)) for q in fzQuats])

		avgCu = np.zeros((variantId, 3))
		vol = np.zeros((variantId,))
		for i in range(clusterGrains.shape[0]):
			v = variants[i]
			avgCu[v] += grainCus[i,:] * (diams[i]**3)
			vol[v] += diams[i]**3

		avgCu[:,:] /= vol[:,np.newaxis]

		avgQu = []
		for cu in avgCu:
			avgQu.append(Quaternion(rotations.cu2qu(list(cu))))

		# build connections between variants
		variantLines = []
		variantAreas = []
		lineCounts = []
		for i in range(len(lines)):
			line = lines[i]
			variantPair = sorted((variants[line[0]], variants[line[1]]))
			if variantPair[0] != variantPair[1]:
				if variantPair in variantLines:
					ind = variantLines.index(variantPair)
					variantAreas[ind] += areas[i]
					lineCounts[ind] += 1
				else:
					variantLines.append(variantPair)
					variantAreas.append(0)
					lineCounts.append(1)
		variantAreas = np.sqrt(variantAreas)

		variantMisos = []
		twinLine = []
		twinLines = []
		for line in variantLines:
			miso = Symmetry.Cubic.disoQuat(avgQu[line[0]], avgQu[line[1]])
			variantMisos.append(miso)
			nDot = sum(miso[1:]) / (np.math.sqrt(3) * np.math.sqrt(1 - miso[0]*miso[0]))
			rot_60 = abs(2*np.math.acos(miso[0])-np.pi/3) * 180 / np.pi <= 10 # within 5 degrees of 60 rotation
			axis_111 = np.math.acos(min(nDot, 1)) * 180 / np.pi <= 10 # within 5 degrees of 111 axis
			if rot_60 and axis_111:
				twinLine.append(True)
				twinLines.append(line)
			else:
				twinLine.append(False)

		# find largest variant
		varMax = np.argmax(vol)
		network = UndirectedGraph(twinLines)

		# compute network metrics
		maxDist = network.maxDist()
		volRootDist = network.computeDistance(varMax)
		distRoots = network.findMinRoot()
		minDist = -1
		numGenerations = -1
		if distRoots is not None:
			distRootDist = network.computeDistance(distRoots[0])
			seperation = distRootDist[varMax]
			equivRoots = len(distRoots)

			generations = volRootDist
			# print(generations)
			numGenerations = max(generations)
			minDist = max(distRootDist)

			numGrains = len(variants) # number of grain in twin related domain
			numVariants = len(generations) # number of varaints
			# print(numGrains)
			clusterVolume = sum(vol) # total volume of twin related domain
			twinArea = 0
			for i in range(len(variantLines)):
				if twinLine[i]:
					twinArea +=variantAreas[i]
			totalArea = sum(variantAreas)
			# print(twinArea / totalArea)
			print(clusterId, minDist, maxDist, numGenerations)
			outfile.write('%i,%i,%i,%f,%f,%f,%f,%i,%i,%i,%f,%f\n' % (clusterId, numGrains, numVariants, twinArea, totalArea, clusterVolume, max(vol), minDist, maxDist, numGenerations, len(lines), len(variantLines)))
print(clusterCount)
