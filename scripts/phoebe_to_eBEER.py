"""Configure eBEER modulation to match a given PHOEBE binary."""

import numpy as np
from astropy import units
import phoebe
from poliastro.core.angles import E_to_nu, M_to_E

class eBEERParams:
    """Wrapper class to minimize clunky action of pulling from phoebe bundle.

    Attributes:
        Porb(float): Orbital Period of the binary | Days
        ecc(float): Orbital Eccentricity | Dimensionless
        incl(float): Inclination of Orbital Plane | Radians
        per0(float): Argument    of Periastron | radians
        t0(float): Zeropoint Date of Periastron Passage | Days
        M1(float): Primary Mass | Msun
        M2(float): Secondary Mass | Msun
        R1(float): Primary Radius | Rsun
        R2(float): Secondary Radius | Rsun
        Prot1(float): Rotation Period of Primary | Days
        Prot2(float): Rotation Period of Secondary | Days
        u1(float): Linear-Limb Darkening Coefficient of Primary. Assumes   
                     ld_func='linear' for primary
        u2(float): Linear-Limb Darkening Coefficient of Secondary. Assumes   
                     ld_func='linear' for secondary
        tau1(float): Gravity Darkening Coefficient of Primary
        tau2(float): Gravity Darkening Coefficient of Primary
        v(float or numpy array): true anomaly of system over time
    """
    
    def __init__(self, phoebe_binary, times):
        """Initializes instance with params from phoebe_binary for given times

        phoebe_binary must have ld_mode_bol set to 'manual' and ld_func_bol 
        set to 'linear'. Assumes linear limb darkening (u) and gravity 
        darkening (tau) coeffs are input into PHOEBE binary before eBEER model 
        is computed.

        Args:
            phoebe_binary(PHOEBE bundle): phoebe system to generate lightcurve 
                                            for
            times(float or numpy array): time(s) in DAYS which the model will 
                                           be computed for 
        """
        orbit = phoebe_binary['component@binary']
        self.Porb = orbit['period'].get_value(units.day)
        self.ecc = orbit['ecc'].get_value('')
        self.incl = orbit['incl'].get_value(units.rad)
        self.per0 = orbit['per0'].get_value(units.rad)
        self.t0 = orbit['t0_perpass'].get_value(units.day)

        star1 = phoebe_binary['primary']
        self.M1 = star1['component@mass'].get_value(units.Msun)
        self.R1 = star1['requiv'].get_value(units.R_sun)
        self.Prot1 = star1['component@period'].get_value(units.day)
        self.u1 = star1['ld_coeffs_bol'].get_value('')
        self.tau1 = star1['gravb_bol'].get_value('')

        star2 = phoebe_binary['secondary']
        self.M2 = star2['component@mass'].get_value(units.Msun)
        self.R2 = star2['requiv'].get_value(units.R_sun)
        self.Prot2 = star2['component@period'].get_value(units.day)
        self.u2 = star2['ld_coeffs_bol'].get_value('')
        self.tau2 = star2['gravb_bol'].get_value('')

        # True anomaly using Poliastro
        M = 2*np.pi * (times - self.t0) / self.Porb # mean anomaly
        M = (M + np.pi) % (2*np.pi) - np.pi
        if isinstance(M, (int, float)):
            self.v = E_to_nu(M_to_E(M, self.ecc), self.ecc)
        elif isinstance(M, np.ndarray):
            self.v = np.zeros(len(M))
            for i, m in enumerate(M):
                temp = E_to_nu(M_to_E(m, self.ecc), self.ecc)
                self.v[i] = temp


def Mbeam(eBp: eBEERParams, alpha_beam: float, calc_for_star_1: bool=True):
    """Beaming modulation due to rel. doppler effect from radial velocities.

    Using equation 2 from Engel et al. 2020 (MNRAS, 497, 4884):
    `https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4884E/abstract`

    Args:
        eBp: eBEERParams data structure holding system parameters
        alpha_beam: scaling factor coefficient (to be tuned using MCMC)
        calc_for_star_1: whether to calculate effect from primary or secondary

    Returns:
        A float or numpy array of the flux modulation due to the beaming effect
    """

    if calc_for_star_1:
        M = eBp.M1
        q = eBp.M2 / eBp.M1
        per0 = eBp.per0
    else:
        M = eBp.M2
        q = eBp.M1 / eBp.M2
        per0 = eBp.per0 + np.pi
        
    # Full Calculation (Eq.2 from Engel et al. 2020)
    return (-2830 * alpha_beam * (q / (1+q)**(2/3)) * M**(1/3) 
            * eBp.Porb**(-1/3) * np.sin(eBp.incl) 
            * (np.cos(per0 + eBp.v) / np.sqrt(1 - eBp.ecc**2)))


def Mellip(eBp: eBEERParams, calc_for_star1: bool=True):
    """Ellipsoidal modulation from tidal distortion of stellar shape.

    Using equations 3-5 from Engel et al. 2020 (MNRAS, 497, 4884):
    `https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4884E/abstract`

    Assume linear limb darkening (u) and gravity darkening (tau) coeffs are 
    input into PHOEBE binary before eBEER model is computed

    Args:
        eBp: eBEERParams data structure holding system parameters
        calc_for_star_1: whether to calculate effect from primary or secondary

    Returns:
        A float or numpy array of the flux modulation due to the ellipsoidal 
        effect
    """

    if (calc_for_star1):
        M = eBp.M1
        q = eBp.M2 / eBp.M1
        R = eBp.R1
        Prot = eBp.Prot1
        u = eBp.u1
        tau = eBp.tau1
        per0 = eBp.per0
    else:
        M = eBp.M2
        q = eBp.M1 / eBp.M2
        R = eBp.R2
        Prot = eBp.Prot2
        u = eBp.u2
        tau = eBp.tau2
        per0 = eBp.per0 + np.pi

    # Eqs 4&5 from Engel et al. 2020
    beta = (1 + eBp.ecc * np.cos(eBp.v)) / (1 - eBp.ecc**2)
    alpha_e1 = (15 * u * (2 + tau)) / (32 * (3 - u))
    alpha_e2 = (3 * (15 + u) * (1 + tau)) / (20 * (3 - u))
    alpha_e2b = (15 * (1 - u) * (3 + tau)) / (64 * (3 - u))
    alpha_e0 = alpha_e2/9 
    alpha_e0b = 3*alpha_e2b/20
    alpha_e3 = 5*alpha_e1/3
    alpha_e4 = 7*alpha_e2b/4

    # Full Calculation (Eq.3 from Engel et al. 2020)
    line1 = (13435 * 2*alpha_e0 
             * (2 - 3 * np.sin(eBp.incl)**2) 
             * M**(-1) * Prot**(-2) * R**3)

    line2 = (13435 * 3*alpha_e0 
             * (2 - 3 * np.sin(eBp.incl)**2) 
             * M**(-1) * (q / (1+q)) * eBp.Porb**(-2) * (beta*R)**(3))
    
    line3 = (759 * alpha_e0b 
             * (8 - 40 * np.sin(eBp.incl)**2 + 35 * np.sin(eBp.incl)**4) 
             * M**(-5/3) * (q / (1+q)**(5/3)) * eBp.Porb**(-10/3) 
             * (beta*R)**5)
    
    line4 = (3194 * alpha_e1 
             * (4 * np.sin(eBp.incl) - 5 * np.sin(eBp.incl)**3) 
             * M**(-4/3) * (q / (1+q)**(4/3)) * eBp.Porb**(-8/3) 
             * (beta*R)**4 * np.sin(per0 + eBp.v))
    
    line5 = (13435 * alpha_e2
              * np.sin(eBp.incl)**2 
              * M**(-1) * (q / (1+q)) * eBp.Porb**(-2) 
              * (beta*R)**3 * np.cos(2*(per0 + eBp.v)))
    
    line6 = (759 * alpha_e2b 
             * (6 * np.sin(eBp.incl)**2 - 7 * np.sin(eBp.incl)**4) 
             * M**(-5/3) * (q / (1+q)**(5/3)) * eBp.Porb**(-10/3) 
             * (beta*R)**5 * np.cos(2*(per0 + eBp.v)))
    
    line7 = (3194 * alpha_e3 
             * np.sin(eBp.incl)**3 
             * M**(-4/3) * (q / (1+q)**(4/3)) * eBp.Porb**(-8/3) 
             * (beta*R)**4 * np.sin(3*(per0 + eBp.v)))
    
    line8 = (759 * alpha_e4 
             * np.sin(eBp.incl)**4 
             * M**(-5/3) * (q / (1+q)**(5/3)) * eBp.Porb**(-10/3) 
             * (beta*R)**5 * np.cos(4*(per0 + eBp.v)))

    return line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8


def Mrefl(eBp: eBEERParams, alpha_refl: float, calc_for_star1: bool=True):
    """Modulation from stellar radiation reflected by companion.

    Using equations 4 & 6 from Engel et al. 2020 (MNRAS, 497, 4884):
    `https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4884E/abstract`

    Args:
        eBp: eBEERParams data structure holding system parameters
        alpha_refl: scaling factor coefficient (to be tuned using MCMC)
        calc_for_star_1: whether to calculate effect from primary or secondary

    Returns:
        A float or numpy array of the flux modulation due to the reflection 
        effect
    """

    if (calc_for_star1):
        M = eBp.M1
        q = eBp.M2 / eBp.M1
        R2 = eBp.R2
        per0 = eBp.per0
    else:
        M = eBp.M2
        q = eBp.M1 / eBp.M2
        R2 = eBp.R1
        per0 = eBp.per0 + np.pi
    
    # Eq.4 from Engel et al. 2020
    beta = (1 + eBp.ecc * np.cos(eBp.v)) / (1 - eBp.ecc**2)

    # Full Calculation (Eq.6 from Engel et al. 2020)
    return (56514 * alpha_refl * (1+q)**(-2/3) * M**(-2/3) 
            * eBp.Porb**(-4/3) * (beta * R2)**2 
            * (0.64 - np.sin(eBp.incl) * np.sin(per0 + eBp.v) 
               + 0.18 * np.sin(eBp.incl)**2 * (1 - np.cos(2*(per0 + eBp.v)))))


def get_eBEER_lc(phoebe_binary, times, alpha_beam1: float, alpha_beam2: float,
                  alpha_refl1: float, alpha_refl2: float, 
                  secondary_flux_fraction=None):
    """Total flux modulation from eBEER effects.

    Using equations 1 & 7 from Engel et al. 2020 (MNRAS, 497, 4884):
    `https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4884E/abstract`

    phoebe_binary must have ld_mode_bol set to 'manual' and ld_func_bol set 
    to 'linear'. Assumes linear limb darkening (u) and gravity darkening 
    (tau) coeffs are input into PHOEBE binary before eBEER model is computed.

    Args:
        phoebe_binary(PHOEBE bundle): phoebe system to generate lightcurve for
        times(float or numpy array): time(s) in DAYS which the model will 
                                       be computed for 
        alpha_beam1: beaming scaling factor coefficient of primary (to be tuned 
                       using MCMC)
        alpha_beam2: beaming scaling factor coefficient of secondary (to be 
                       tuned using MCMC)
        alpha_refl1: reflection scaling factor coefficient of primary (to be 
                       tuned using MCMC)
        alpha_refl2: reflection scaling factor coefficient of secondary (to be 
                       tuned using MCMC)
        secondary_flux_fraction(str or float or None):
            relative flux fraction of secondary compared to primary
            Options:    
                        None (Default): Calculated based on component 
                                          temperatures and masses
                        float: relative flux fraction entered manually
                        'split': returns primary and secondary       
                                       components separately

    Returns:
        Total eBEER flux modulation. If secondary_flux_fraction is 'split' 
        function return is a tuple (MeBEER1, MeBEER2) which are the calculated 
        eBEER modulations (float or numpy array) from the primary and secondary 
        respectively. Otherwise return is a single float or numpy array which 
        is the combined eBEER flux modulation from both stars   
    """

    eBp = eBEERParams(phoebe_binary, times)

    # Eq.1 from Engel et al. 2020
    Mbeam1 = Mbeam(eBp, alpha_beam1)
    Mbeam2 = Mbeam(eBp, alpha_beam2, False)
    Mellip1 = Mellip(eBp)
    Mellip2 = Mellip(eBp, False)
    Mrefl1 = Mrefl(eBp, alpha_refl1)
    Mrefl2 = Mrefl(eBp, alpha_refl2, False)
    MeBEER1 = Mbeam1 + Mellip1 + Mrefl1
    MeBEER2 = Mbeam2 + Mellip2 + Mrefl2

    if secondary_flux_fraction is None:
        secondary_flux_fraction = (
            phoebe_binary['secondary@teff'].get_value('K')
            /
            phoebe_binary['primary@teff'].get_value('K')
        )**4 * (eBp.R2 / eBp.R1)**2
    elif secondary_flux_fraction == 'split':
        return (MeBEER1, MeBEER2)

    # Eq.7 from Engel et al. 2020
    return (MeBEER1 + secondary_flux_fraction * MeBEER2) / (1 + secondary_flux_fraction)