'''Script to test the "Getting Started with Dipy" section of the Dipy Documentation using the Stanford HARDI dataset'''


#functions to get the home folder and to join text with slashes
from os.path import expanduser, join

#downloading the Stanford HARDI dataset
from dipy.data import fetch_stanford_hardi
fetch_stanford_hardi()

#creating a variable with the path to the folder containing the DTI images
#the Stanford HARDI dataset is downloaded by default to the '~/.dipy/stanford_hardi/' directory
home = expanduser('~')
directory_name = join(home, '.dipy', 'stanford_hardi')

#creating variables for the dwi, b-values and b-vectors
file_dwi = join(directory_name,'HARDI150.nii.gz')
file_bval = join(directory_name, 'HARDI150.bval')
file_bvec = join(directory_name, 'HARDI150.bvec')

#printing the names of the dwi, b-values and b-vectors to the terminal
print(file_dwi)
print(file_bvec)
print(file_bval)

#importing nibabel to load the dwi image
import nibabel as nib
image = nib.load(file_dwi)
data = image.get_data()

print(data.shape)

print(image.get_header().get_zooms()[:3])

import matplotlib.pyplot as plt

axial_middle = data.shape[2] / 2
plt.figure('Showing the datasets')
plt.subplot(1,2,1).set_axis_off()
plt.imshow(data[:,:, axial_middle, 0].T, cmap='gray', origin='lower')

plt.subplot(1,2,2).set_axis_off()
plt.imshow(data[:,:, axial_middle, 10].T, cmap='gray', origin='lower')
plt.show()

from dipy.io import read_bvals_bvecs
bvals, bvecs = read_bvals_bvecs(file_bval, file_bvec)

from dipy.core.gradients import gradient_table
gtab = gradient_table(bvals, bvecs)

print(gtab.info)
print(gtab.bvals)
print(gtab.bvecs[:10,:])

S0s = data[:,:,:, gtab.b0s_mask]

print(S0s.shape)

nib.save(nib.Nifti1Image(S0s, image.get_affine()), 'teste_S0s.nii.gz')

import numpy as np
import dipy.reconst.dti as dti

from dipy.segment.mask import median_otsu
maskdata, mask = median_otsu(data, 3, 1, True, vol_idx=range(10,50), dilate=2)
print('maskdata.shape (%d, %d, %d, %d)' % maskdata.shape)

tensor_model = dti.TensorModel(gtab)

tensor_fit = tensor_model.fit(maskdata)

print('Computing Anisotropy Measures(FA, MD, RGB)')

from dipy.reconst.dti import fractional_anisotropy, color_fa, lower_triangular

FA = fractional_anisotropy(tensor_fit.evals)

FA[np.isnan(FA)] = 0

fa_image = nib.Nifti1Image(FA.astype(np.float32), image.get_affine())
nib.save(fa_image, 'tensor_fa.nii.gz')

eigenvectors_image = nib.Nifti1Image(tensor_fit.evecs.astype(np.float32), image.get_affine())
nib.save(eigenvectors_image, 'tensor_eigenvectors.nii.gz')

MD = dti.mean_diffusivity(tensor_fit.evals)
nib.save(nib.Nifti1Image(MD.astype(np.float32), image.get_affine()), 'tensor_md.nii.gz')

FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, tensor_fit.evecs)
nib.save(nib.Nifti1Image(np.array(255*RGB, 'uint8'), image.get_affine()), 'tensor_rgb.nii.gz')

print('Computing tensor ellipsoids in a part of the splenium of the CC')

from dipy.data import get_sphere
sphere = get_sphere('symmetric724')

from dipy.viz import fvtk
ren = fvtk.ren()

eigenvalues = tensor_fit.evals[13:43, 44:74, 28:29]
eigenvectors = tensor_fit.evecs[13:43, 44:74, 28:29]

color_fa = RGB[13:43, 44:74, 28:29]
cfa /= cfa.max()

fvtk.add(ren, fvtk.tensor(eigenvalues, eigenvectors, color_fa, sphere))

print('Saving illustration as tensor_ellipsoids.png')
fvtk.record(ren, n_frames=1, out_path='tensor_ellipsoids.png')

fvtk.clear(ren)

tensor_odfs = tensor_model.fit(data[20:50, 55:85, 38:39]).odf(sphere)

fvtk.add(ren, fvtk.sphere_funcs(tensor_odfs, sphere, colormap=None))
#fvtk.show(r)
print('Saving illustration as tensor_odfs.png')
fvtk.record(ren, n_frames=1, out_path='tensor_odfs.png', size=(600, 600))
