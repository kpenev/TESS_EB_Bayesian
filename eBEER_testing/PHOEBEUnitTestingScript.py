'''Similar to "PHOEBEUnitTesting.ipynb" this notebook is used to "unit" test 
the ellipsoidal and reflection effects against simple PHOEBE systems to ensure 
behavior is correct.

Does the exact same thing as the notebook mentioned above but is best suited 
for verifying hundreds of combinations at once. For shorter tests use 
"PHOEBEUnitTestingScript.py".

When running the script in command line, you are required to define the 
filename (NOT including the file extension) for the output files (--fn). It 
also accepts an optional boolean flag to turn on reflection (--refl) for a 
given run.

The combinations for testing are defined below and must be modified in this 
file if you want to look at different parameters.

The script outputs 2 files with the filename given. The first is a pdf of the 
light curve plots for all the desired combinations. The second file is a 
pickled dictionary with all of the system information required to recreate and 
plot the data. As we return some more information from fitting when reflection 
is on, the values in the dictionary depend on whether the reflection flag is 
enabled. All this means pratically is that you must specify the setup of the 
run when looking at the data in "UnitTestScriptResults.ipynb" (More information 
in the notebook)

NOTE: MPI may not be properly enabled. Check 
https://phoebe-project.org/docs/latest/tutorials/mpi
'''

import phoebe
import sys
sys.path.insert(1, '../scripts')
import phoebe_to_eBEER
import numpy as np
from astropy import units as u
from scipy.optimize import curve_fit

import itertools
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

def main(args):
    start = time.time()
    print(args)

    arrays_dict = {}

    # Define combinations to test
    eccs = (0, 0.3, 0.8)
    incls = (0,30,90)#(0, 30, 60, 90, 120, 150, 180)
    per0s = (0,)#(0, 30, 90, 150, 180, 240)
    qs = (1,)#(1, 0.5, 2,)
    ps = (1,)#(1, 5, 10, 15, 20,)

    combinations = itertools.product(eccs, incls, per0s, ps, qs)
    comb_list = list(combinations)

    arrays_dict['Orbital Parameters'] = (eccs, incls, per0s, ps, qs)
    arrays_dict['Refl ON'] = args.refl
    
    # Initialize Binary with Effects ON
    b_ = phoebe.default_binary()
    b_.add_dataset('lc', times=0, label= 'lc01')
    b_.set_value_all('gravb_bol', 0.32)
    b_.set_value_all('ld_mode*', 'manual')
    b_.set_value_all('ld_func*', 'linear')
    b_.set_value_all('ld_coeffs*', [0.5])
    b_.set_value_all('ntriangles', 20000)
    b_['eclipse_method'] = 'only_horizon'
    b_['passband'] = 'Kepler:mean'
    b_.set_value_all('atm', 'phoenix')
    b_.flip_constraint('mass@primary', solve_for='sma')
    if not args.refl:
        b_.set_value('irrad_method', 'none')
    
    # Initialize Binary with Effects OFF
    b_2 = phoebe.default_binary()
    b_2.add_dataset('lc', times=0, label= 'lc01')
    b_2.set_value_all('gravb_bol', 0.32)
    b_2.set_value_all('ld_mode*', 'manual')
    b_2.set_value_all('ld_func*', 'linear')
    b_2.set_value_all('ld_coeffs*', [0.5])
    b_2.set_value_all('ntriangles', 10000)
    b_2['eclipse_method'] = 'only_horizon'
    b_2['passband'] = 'Kepler:mean'
    b_2.set_value_all('atm', 'phoenix')
    b_2.flip_constraint('mass@primary', solve_for='sma')
    b_2.set_value('irrad_method', 'none')
    b_2.set_value_all('distortion_method', value='sphere')


    def fit_scaling(arefl1, arefl2):
        scaled = np.zeros(2)
        scaled[0] = arefl1 * 7
        scaled[1] = arefl2 * 7
        return scaled

    def fit_eBEER(t, arefl1, arefl2):
        scaled = fit_scaling(arefl1, arefl2)
        flux = phoebe_to_eBEER.get_eBEER_lc(b_, t, 0, 0, scaled[0], scaled[1])
        return flux


    with PdfPages(f'{args.fn}.pdf') as pdf:
        for i, combination in enumerate(comb_list):
            start_sys = time.time()
            ecc, incl, per0, p, q = combination
            comb_str = f'lc_{p}_{int(ecc*10)}_{incl}_{per0}_{int(q*10)}'
            print('System:', comb_str)

            # Configure Binary with Effects ON
            times = np.linspace(0,p,101)
            b_['lc01@dataset@times']= times
            b_['orbit@period'].set_value(p * u.day)
            b_['orbit@incl'].set_value(incl * u.deg)
            b_['orbit@ecc'].set_value(ecc)
            b_['orbit@per0'].set_value(per0 * u.deg)
            b_['mass@primary@component'].set_value(1 * u.M_sun)
            b_['orbit@q'].set_value(q)
            b_.run_checks_compute()
            R1_max = b_['primary@component@requiv_max'].get_value(u.R_sun)
            R2_max = b_['secondary@component@requiv_max'].get_value(u.R_sun)
            b_['primary']['requiv'].set_value(R1_max * u.R_sun)
            b_['secondary']['requiv'].set_value(R2_max * u.R_sun)
            # b_['primary']['requiv'].set_value(1 * u.R_sun)
            # b_['secondary']['requiv'].set_value(q**0.8 * u.R_sun)
            setup_done = time.time()
            print('\tSystem Setup Duration:', setup_done - start_sys)
            
            try: # Will hide any combinations that are failing due to falling 
                 # outside of valid atmosphere tables or similar issues
                b_.run_compute(model= comb_str, overwrite= True)
                lc_done = time.time()
                print('\tLC Dataset Duration:', lc_done - setup_done)
            except:
                print('\tFAILED COMPUTE')
                continue
            
            # Configure Binary with Effects OFF
            b_2['orbit@period'].set_value(p * u.day)
            b_2['orbit@incl'].set_value(incl * u.deg)
            b_2['orbit@ecc'].set_value(ecc)
            b_2['orbit@per0'].set_value(per0 * u.deg)
            b_2['mass@primary@component'].set_value(1 * u.M_sun)
            b_2['orbit@q'].set_value(q)
            b_2['primary']['requiv'].set_value(R1_max * u.R_sun)
            b_2['secondary']['requiv'].set_value(R2_max * u.R_sun)
            # b_2['primary']['requiv'].set_value(1 * u.R_sun)
            # b_2['secondary']['requiv'].set_value(q**0.8 * u.R_sun)
            b_2.run_compute(overwrite= True)

            flux = b_[f'fluxes@lc01@{comb_str}@model'].value
            ref_flux = b_2['fluxes@lc01@latest@model'].value
            relative_flux_ = (flux - ref_flux)/ref_flux * 1e6


            t0 = b_['orbit@component@t0_perpass'].get_value(u.day)
            if not args.refl:
                eBEER_flux = phoebe_to_eBEER.get_eBEER_lc(b_, times, 0, 0, 0, 0)
                sys_values = (times, t0, relative_flux_, eBEER_flux)

            else:
                popt, pcov = curve_fit(fit_eBEER, times, relative_flux_, 
                                    p0= (0.5, 0.5), bounds= ([0, 0], [1, 1]))
                rescaled_ = fit_scaling(popt[0], popt[1])
                fit_flux_ = fit_eBEER(times, popt[0], popt[1])
                print('\tFit Duration:', time.time() - lc_done)
                print(f'\t\tFit Params (rescaled): {rescaled_}')
                sys_values = (times, t0, relative_flux_, fit_flux_, rescaled_)


            fig = plt.figure(figsize=(8, 5))
            plt.plot(times, relative_flux_, 'b.', markersize=1, 
                     label= 'PHOEBE LC')
            if not args.refl:
                plt.plot(times, eBEER_flux, 'r', markersize=1, 
                         label= 'eBEER Flux')
            else:
                plt.plot(times, fit_flux_, 'r', markersize=1, 
                         label= 'eBEER Fit')
            plt.title(fr'{i} - e:{ecc}, incl:{incl}, $\omega$:{per0}, Porb:{p}, q:{q}')
            plt.xlabel('Time (days)')
            plt.ylabel('dFlux (ppm)')
            plt.legend()
            pdf.savefig(fig)
            plt.close(fig)

            arrays_dict[comb_str] = sys_values
            print('\tTotal Duration:', time.time() - start_sys)
    
    outfile = f'{args.fn}.pkl'
    with open(outfile, 'wb') as f:
        pickle.dump(arrays_dict, f)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--refl', action= 'store_true', 
                        help= "Run with reflection")
    parser.add_argument('--fn', type= str, required= True, 
                        help= "Filname for outputs WITHOUT FILE EXTENSION")
    args = parser.parse_args()

    main(args)