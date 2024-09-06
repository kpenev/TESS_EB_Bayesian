import phoebe
from astropy import units as u

def get_phoebe_binary(Porb= 1, t0= -0.25, M1= 0.9988131358058301, 
                      M2= 0.9988131358058301, R1= 1, R2= 1, T1= 6000, T2= 6000,
                      incl= 90, per0= 0, ecc= 0, u1= 0.5, u2= 0.5, tau1= 0.32,
                      tau2= 0.32, metallicity= None):
    """Configure a PHOEBE binary with input parameters.
    
    Default parameters above are the PHOEBE defaults
    Assumes Prot of stars = Porb of system (default PHOEBE behavior)

    Args: System Parameters
        t0: time of periastron passage in days
        per0: argument of periastron in degrees
        u1, u2: primary and secondary linear limb darkening coefficients
        tau1, tau2: primary and secondary gravity darkening coefficients

    Returns:
        A PHOEBE bundle configured with the above parameters
    """

    binary = phoebe.default_binary()

    binary['primary']['teff'].set_value(T1 * u.K)
    binary['secondary']['teff'].set_value(T2 * u.K)

    orbit = binary['orbit']
    orbit['period'].set_value(Porb * u.day)

    orbit['incl'].set_value(incl * u.deg)

    orbit['ecc'].set_value(ecc)
    orbit['per0'].set_value(per0 * u.deg)

    binary.flip_constraint('mass@primary', solve_for='sma')
    binary.flip_constraint('mass@secondary', solve_for='q')
    binary['primary']['mass'].set_value(M1 * u.M_sun)
    binary['secondary']['mass'].set_value(M2 * u.M_sun)

    binary['primary']['requiv'].set_value(R1 * u.R_sun)
    binary['secondary']['requiv'].set_value(R2 * u.R_sun)

    binary.flip_constraint('t0_perpass', solve_for= 't0_supconj')
    orbit['t0_perpass'].set_value(t0 * u.day)
    
    for component in ['primary', 'secondary']:
        binary[component]['ld_mode_bol'].set_value('manual')
        binary[component]['ld_func_bol'].set_value('linear')
    
    binary['primary']['ld_coeffs_bol'].set_value(u1)
    binary['secondary']['ld_coeffs_bol'].set_value(u2)
    binary['primary']['gravb_bol'].set_value(tau1)
    binary['secondary']['gravb_bol'].set_value(tau2)

    if (metallicity is not None):
        binary['primary@abun'] = metallicity
        binary['secondary@abun'] = metallicity

    return binary

def print_pb_params(b):
    orbit = b['component@binary']
    Porb = orbit['period'].get_value(u.day)
    ecc = orbit['ecc'].get_value('')
    incl = orbit['incl'].get_value(u.deg)
    per0 = orbit['per0'].get_value(u.deg)
    q = orbit['q'].get_value('')
    t0 = orbit['t0_perpass'].get_value(u.day)

    star1 = b['primary']
    M1 = star1['component@mass'].get_value(u.Msun)
    R1 = star1['requiv'].get_value(u.R_sun)
    T1 = star1['teff'].get_value(u.K)
    Prot1 = star1['period@component'].get_value(u.day)
    try:
        u1 = star1['ld_coeffs_bol'].get_value('')
    except:
        u1 = "Limb Darkening Coeffs not exposed"
    tau1 = star1['gravb_bol'].get_value('')
    
    star2 = b['secondary']
    M2 = star2['component@mass'].get_value(u.Msun)
    R2 = star2['requiv'].get_value(u.R_sun)
    T2 = star2['teff'].get_value(u.K)
    Prot2 = star2['period@component'].get_value(u.day)
    try:
        u2 = star2['ld_coeffs_bol'].get_value('')
    except:
        u2 = "Limb Darkening Coeffs not exposed"
    tau2 = star2['gravb_bol'].get_value('')
    

    print(fr'Orbit - e: {ecc},  i: {incl},  $\omega$: {per0},  Porb: {Porb}, ',
          f't0_periastron: {t0},  q: {q}')
    print(f'Star 1 - M1: {M1},  R1: {R1},  T1: {T1},  Prot1: {Prot1}, ',
          f'Limb Dark. Coeff: {u1},  Grav Dark. Coeff: {tau1}')
    print(f'Star 2 - M2: {M2},  R2: {R2},  T2: {T2},  Prot2: {Prot2}, ',
          f'Limb Dark. Coeff: {u2},  Grav Dark. Coeff: {tau2}')