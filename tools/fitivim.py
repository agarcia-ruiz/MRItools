'''
Wrapper of DiPy for fitting of IVIM
Adapted from Francesco Grussu (commented code below) and following the documentation example:
https://dipy.org/documentation/1.5.0/examples_built/reconst_ivim/#example-reconst-ivim 
'''
### Fitting of Diffusion Kurtosis Imaging on DWI data; wrapper of DiPy
#
#
# Author: Francesco Grussu, UCL Queen Square Institute of Neurology
#         <f.grussu@ucl.ac.uk>
#
# Copyright (c) 2019 University College London. 
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project.

# Load 	useful modules
import multiprocessing
import argparse, sys, os
import numpy as np
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dki as dki
from dipy.reconst.ivim import IvimModel

# Run the module as a script
if __name__ == "__main__":

	# Print help and parse arguments
    parser = argparse.ArgumentParser(description='Run DKI fitting using DiPy. Command-line wrapper for the following DiPy tutorial: http://dipy.org/documentation/1.0.0./examples_built/reconst_dki . Author: Francesco Grussu, <f.grussu@ucl.ac.uk>.')
    parser.add_argument('dwi', help='DWI data set (4D Nifti)')
    parser.add_argument('out', help='Output base name. Output files will append to the base name "_s0.nii" (predicted S0 signal), "_fv.nii" (vascular fraction), "_Dv.nii" (vascular perfusion or pseudo-diffusivity, in um^2/ms), "_Dt.nii" (tissue diffusivity, in um^2/ms).')
    parser.add_argument('bvals', help='b-value files in FSL format (1 x no. images), expressed in [s mm^-2]')
    parser.add_argument('bvecs', help='gradient directions in FSL format (3 x no. images)')	
    parser.add_argument('--mask', metavar='<file>', help='mask (Nifti; 1/0 in voxels to include/exclude)')
    parser.add_argument('--bmin', metavar='<list>', help='min values as comma-separaterd list for S0, Dt, fv, Dv (trr); Dt, fv, Dv (VarPro). Default: [0., 0., 0.,0.].')
    parser.add_argument('--bmax', metavar='<list>', help='max values as comma-separaterd list for S0, Dt, fv, Dv (order?); Dt, fv, Dv (VarPro). Default: [np.inf, .3, 1., 1.]. Will convert NaNs to these(?)')
    parser.add_argument('--method', metavar='<type>', default='trr', help='Method for IVIM fitting [trr (default), VarPro]. trr splits the bvalues at thresholds (default: 400 and 200) to initially fit S0 and Dt as a monoexponential. VarPro uses the MIX approach to fit without thresholding')
    parser.add_argument('--finalfit', metavar='<flag>', default='0', help='For trr method, set to 1 to perform a final non-linear fit initialising with previous result. Default: 0')
	
    args = parser.parse_args()

    # Get input parameters
    dwifile = args.dwi
    outbase = args.out
    bvalfile = args.bvals
    bvecfile = args.bvecs
    maskid = args.mask
    bmin = args.bmin
    bmax = args.bmax
    method = args.method
    finalfit = args.finalfit

    if not bmin:
        bmin = [0.01,0.01,0.01]
        if method=='trr':
            bmin.insert(0, 0.01)
    if not bmax:
        bmax = [.32, 1., 1.]
        if method=='trr':
            bmax.insert(0, np.inf)
    bounds = (bmin, bmax)

    # Print
    print('')
    print('*************************************************')
    print('    Fitting of DKI model (wrapper of DiPy)       ')
    print('*************************************************')
    print('')
    print('Called on 4D DWI Nifti file: {}'.format(dwifile))
    print('b-values: {}'.format(bvalfile))
    print('gradient directions: {}'.format(bvecfile))
    # print('kurtosis excess range: [{}; {}]'.format(kmin,kmax))
    print('Output base name: {}'.format(outbase))
    print('')

    # Load DWI
    dwiobj = nib.load(dwifile)    # Input data
    dwiimg = dwiobj.get_fdata()
    imgsize = dwiimg.shape

    # Load b-values and gradient directions
    bvals, bvecs = read_bvals_bvecs(bvalfile, bvecfile)
    gtab = gradient_table(bvals, bvecs, b0_threshold=0)

    # Load mask if necessary
    if isinstance(maskid, str)==1:

        # The user provided a mask: load it
        maskobj = nib.load(maskid)
        maskvals = np.array(maskobj.get_fdata(),dtype=bool)
    else:
        # No mask provided. TODO try segment foreground to avoid error with VarPro method?
        maskvals = np.array(np.ones(imgsize[0:3]),dtype=bool)

    # Starting DKI model fitting
    print('')
    print('... starting model fitting. Please wait...')
    print('')
    
    # maxiter = 3000
    if method=='trr' and finalfit:
        model = IvimModel(gtab, fit_method=method, bounds=bounds, two_stage=finalfit)
    else:
        model = IvimModel(gtab, fit_method=method, bounds=bounds)
    # if isinstance(maskid, str)==1:
    mfit = model.fit(dwiimg, mask=maskvals)

    # Get output metrics
    s0 = np.array(mfit.S0_predicted)
    fv = np.array(mfit.perfusion_fraction)
    Dv = np.array(mfit.D_star)
    Dt = np.array(mfit.D) # 1e3* to convert to um^2/ms?

    # Save output files
    print('')
    print('... saving output files...')
    print('')

    buffer_header = dwiobj.header
    buffer_header.set_data_dtype('float64')   # Make sure we save quantitative maps as float64, even if input header indicates a different data type

    buffer_string=''
    seq_string = (outbase,'_s0.nii')
    fa_outfile = buffer_string.join(seq_string)
        
    buffer_string=''
    seq_string = (outbase,'_fv.nii')
    ad_outfile = buffer_string.join(seq_string)

    buffer_string=''
    seq_string = (outbase,'_Dv.nii')
    rd_outfile = buffer_string.join(seq_string) 

    buffer_string=''
    seq_string = (outbase,'_Dt.nii')
    md_outfile = buffer_string.join(seq_string)

    fa_obj = nib.Nifti1Image(s0,dwiobj.affine,buffer_header)
    nib.save(fa_obj, fa_outfile)

    ad_obj = nib.Nifti1Image(fv,dwiobj.affine,buffer_header)
    nib.save(ad_obj, ad_outfile)

    rd_obj = nib.Nifti1Image(Dv,dwiobj.affine,buffer_header)
    nib.save(rd_obj, rd_outfile)

    md_obj = nib.Nifti1Image(Dt,dwiobj.affine,buffer_header)
    nib.save(md_obj, md_outfile)

    # Done
    print('')
    print('... done.')
    print('')
