# amplitude statistics
import sys
sys.path.insert(0, '/n/home02/toshiyan/Lib/cmblensplus/utils/')

import numpy as np
import analysis as ana

#//// set parameters ////#
#bn = 12
#bn = 15
bn = 10
BK = 'B33y'
#BK = 'BK15'
id = 3

if BK == 'BK15':
    freqs = ['150']
    simn = 499
    data = '/n/holylfs04/LABS/kovac_lab/users/namikawa/BK15/bispec_bbb/'
    #tag   = 'lx30_lcdm_oL1-580_b'+str(bn)
    tag   = 'lx30_lcdm_oL1-300_b'+str(bn)

if BK == 'B33y':
    freqs = ['100']
    simn  = 499
    data  = '/n/holylfs04/LABS/kovac_lab/users/namikawa/B33y/bispec_bbb/'
    tag   = 'lx0_nobl_lcdm_oL1-300_b'+str(bn)


# compute AL for each analysis choice
for freq in freqs:

    if BK == 'BK15':
        #obs = ((np.loadtxt(data+'/b1d_1var_nu'+freq+'_lx30_lcdm_dp1102_oL1-580_b'+str(bn)+'_0.dat')).T[id])[:bn]
        sbl = np.array([(np.loadtxt(data+'/b3d_1var_nu'+freq+'_'+tag+'_'+str(i+1)+'.dat')).T[id,:] for i in range(simn)])
        fbl = np.loadtxt(data+'/mb3d_fnle_1var_nu'+freq+'_'+tag+'.dat',unpack=True)
        sbl = np.array([sbl[i,:][fbl!=0] for i in range(simn)])
        fbl = fbl[fbl!=0]
        print(np.shape(sbl),np.shape(fbl))

    elif BK == 'B33y':
        #obl = ((np.loadtxt(data+'/b3d_1var_nu'+freq+'_'+tag.replace('cdm','cdm_dp1102')+'_0.dat')).T[id,:])
        obl_name = data+'/b3d_1var_nu'+freq+'_'+tag.replace('cdm','cdm_Vdst')+'_1.dat'
        print('Loading obl: ' + obl_name)
        obl = (np.loadtxt(obl_name).T[id,:])
        #obl = ((np.loadtxt(data+'/b3d_1var_nu'+freq+'_'+tag+'_1.dat')).T[id,:])
        
        sbl_name = data+'/b3d_1var_nu'+freq+'_'+tag+'_'
        print('Loading sbl: ' + sbl_name)
        sbl = np.array([(np.loadtxt(sbl_name+str(i+1)+'.dat')).T[id,:] for i in range(simn)])
        print('Loading fbl')
        fbl = np.loadtxt(data+'/mb3d_fnle_1var_nu'+freq+'_'+tag+'.dat',unpack=True)

        sbl = np.array([sbl[i,:][fbl!=0] for i in range(simn)])
        obl = obl[fbl!=0]
        fbl = fbl[fbl!=0]
        print(np.shape(sbl),np.shape(fbl))

    # compute sigma(fNL) for fNL=1
    for diag in [True,False]:
        w = ana.statistics(obl,sbl)
        #ana.statistics.get_amp_split(w,100,fcl=fbl,diag=diag,cor_val=.1)
        ana.statistics.get_amp(w,fcl=fbl,diag=diag)
        print('amp(sim)', np.round(w.mA,3), np.round(w.oA,3), 'sigma(fNL)', np.round(w.sA,3))


