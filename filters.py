import numpy as np
import crystallography as xtal
import h5py
import copy

def compute_twin_orientations(path):
	symmetry = xtal.Symmetry('Cubic')
	f = h5py.File(path,'r+')['DataContainers/SyntheticVolumeDataContainer/CellFeatureData']
	
	key = 'AvgQuats'
	try:
		quats = xtal.do2qu(np.roll(f['AvgQuats'],-1,1))
	except KeyError:
		key = 'Quats'
		quats = xtal.do2qu(np.roll(f['Quats'],-1,1))
	attrs = f[key].attrs

	parents = np.array(f['ParentIds']).flatten()
	twins = np.arange(len(parents))[np.abs(parents-np.arange(len(parents))) > 0][1:]	
	parents = parents[np.abs(parents-np.arange(len(parents))) > 0][1:]
	sigma_3s = [[1,1,1],[1,1,-1],[1,-1,1],[-1,1,1]]
	
	for twin,parent in zip(twins,parents):
		choices = np.arange(len(sigma_3s))
		sigma_3 = copy.deepcopy(sigma_3s[np.random.choice(choices)])
		sigma_3.append(np.pi)
		sigma_3 = xtal.ax2qu(np.array(sigma_3))[0]	
		quats[twin] = sigma_3*quats[parent]

	quats = np.array(np.roll(xtal.qu2do(quats),1,-1),dtype='float32')
		
	del f[key]
	f.create_dataset(key,data=quats)
	for name,value in attrs.items():
		f[key].attrs[name] = value
	
	print('Done inserting Sigma3 Orientations.')
	
