'''Script to test the "Getting Started with Dipy" section of the Dipy Documentation'''

from os.path import expanduser, join
#functions to get the home folder and to join text with slashes

#creating a variable with the path to the folder containing the DTI images
home = expanduser('~')
directory_name = join(home, 'InsCer', 'visit1', 'DTI')

#creating variables for the dwi, b-values and b-vectors
file_dwi = join(directory_name,'ESCM501.DTI.nii.gz')
file_bval = join(directory_name, 'ESCM501.DTI.bval')
file_bvec = join(directory_name, 'ESCM501.DTI.bvec')

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
plt.savefig('data.png', bbox_inches='tight')

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
