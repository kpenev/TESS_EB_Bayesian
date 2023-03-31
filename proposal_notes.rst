Goals
=====

* Fully characterized distributions of EB physical parameters from TESS
  photometry and SED information (GAIA + 2MASS + ... catalogues)

* Spin of EB primary stars

* Detection of ellipsoidal variations, reflected light, doppler beaming

* Measure eclipsing timing variability (ETV)? 
  
  * apsidal precession 

  * other causes?

Applications:
=============

* Measure Qstar:

  * control with long period systems to show no signature => no
    constraints

  * Comppare spin vs eccentricity constraints (same object different
    dominant freuencies)

  * Many more systems than W19 => able to explore more dependencies, not
    just frequency.

* can serve as control for measuring planet Q:

  * Probe very similar population of stars to exoplanet hosts (many
    selection effects shared: brightness limit, resolution, sky coverage)

  * Stellar dissipation has big impact on required planet dissipation (can
    drive partcircularization itself, does not conserve angular momentum
    => bigger orbital decay for same circularization).

* constrain population and properties of exoplanet false positives

* Direct test of stellar evolution:

  * https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.1953D/abstract, 

  * https://ui.adsabs.harvard.edu/abs/2012MNRAS.427..127B/abstract

  * https://ui.adsabs.harvard.edu/abs/2016ApJ...832..121G/abstract


* ETV can constrain k2, hence internal structure

* Limb darkening calibration and stellar atmospheres modeling

* Spectral synthesis for galaxy science:
  https://ui.adsabs.harvard.edu/abs/2017PASA...34...58E/abstract

* Study tides: interesting article:  Rotation Period Evolution in Low-mass
  Binary Stars: The Impact of Tidal Torques and Magnetic Braking 

* Dark Matter studies:
  https://ui.adsabs.harvard.edu/abs/2022ApJ...928L..17C/abstract

* Read through
  https://ui.adsabs.harvard.edu/abs/2012ocpd.conf...51S/abstract for further
  applications

* Look through citations to `Lurie et. al. (2017)
  <https://ui.adsabs.harvard.edu/abs/2017AJ....154..250L/abstract>`_ for
  application of spin measurements, ellipsoidal variations, etc.

Expected outcomes discussion should include:
============================================

* Prediction for:

  * how many binaries will have well characterized physical parameters

  * how many will have spin detections

  * what will be the overlap between the two above

  * how many EBs will have apsidal precession measurements with good
    precision

* Selection effects:

  * Stamp selection effects are hard to characterize but high cadence and
    Prsa et al. (2022) compiled comprehensive list to search

  * FFI: no selection effects (everything downloaded). We will use very
    straightforward algorithm for generating candidate list for manual
    review. Only select well detached high signal to noise EBs. We can do
    injection-recovery simulations to characterize selection.

  * What additional selection effects we impose and how will we
    characterize them

    * physical parameter sample

    * spin sample

    * differences of selection effects between short and long cadence
      LCs

Methodology:
============

* LC modeling:

  * For well separated binaries (vast majority): use BATMAN + orbit based
    phase curvefor LC modeling.

    * Works to few ppt regardless of binary parameters or orientation.

    * How does precision compare to typical model uncertainties?

      * stellar evolution? Seems ~1% differences between models are
        not atypical better near the sun (since Solar calibrated) but
        worse the further away from sun we go. E.g. 
        `Bressan et. al. (2012) <https://ui.adsabs.harvard.edu/abs/2012MNRAS.427..127B/abstract>`_

      * atmospheres?

    * Precision better than spot variability?

    * For building orbits for phase curve terms: 
      
        * will not sample uniform times, but interpolate time like the other
          parameters. 
          
        * Use table of prescribed eccentric anomaly values at grid of
          eccentricities to ensure interpolation to desired precision.

        * Simulate orbit in 2D apply rotation matrix to account for
          inclination and periapsis (use GPUs?)

    * For the very few very close (and high precision?) binaries can use
      PHOEBE if no other option. Investigate when we need to switch. Perhaps
      OK to drop.

    * Analysis will begin with fast model to find max likelihood values, but
      will not run to convergence. Manual inspection of max likelihood LC vs
      instrumental LC will catch few binaries requiring full PHOEBE.

* SED modeling options (choose one or perhps need mixture):

  * **CDS isochrones, combining two isolated stars**: 
    
    * provides TESS magnitudes for LC models

    * Have interpolation already working
  
  * ~~PHOEBE?~~

  * Exctiction and reddening options: 
    
    * what do Justesen & Albrecht 
      
    * W19?

    * CMD: "Using extinction coefficients computed star-by-star (except
      for the OBC case, which uses constant coefficients)". Read paper
      and see how that works.

* Spin measurements:

  * Follow Lurie to combine autocorrelation and Lomb-Scargle periodograms.

  * Most directly relevant paper: `Martins et. al. (2020)
    <https://ui.adsabs.harvard.edu/abs/2020ApJS..250...20C/abstract>`_

  * Wavelets would also be useful. A good starting point may be `Bravo et. al.
    (2014) <https://ui.adsabs.harvard.edu/abs/2014A%26A...568A..34B/abstract>`_.
    It suggest using 6-th order Morlett transform. This could serve as a check
    and perhaps additional way to flag candidates and improve analysis for stars
    with intermittent rotational variability.

  * Find other spin papers to see if other approaches are used?

  * What quality control measures can be used?

    * What does Lurie do?

    * What do other papers do?

  * In addition to spin, we will detect 


* Physical parameter validation:
    
  * W19 

    * analyze small sample of Kepler LCs to compare algorithms

    * Find overlapping detections to compare to independent analysis applied on
      independent data

  * J&A

    * Very similar analysis. Run on J&A short LCs to compare algorithm.
      
    * Compare to their full catalog 

  * `Prsa et. al. (2022)
    <https://ui.adsabs.harvard.edu/abs/2022ApJS..258...16P/abstract>`_

  * What did above validate against

* For spin validation compare to:

  * Martins et. al. (2020) above and the associated `livig database
    <https://filtergraph.com/tess_rotation_tois>`_ (287 TOIs with "unambiguous
    rotation" flag)
    
  * KELT rotations: `Oelkers et al. (2018)
    <https://ui.adsabs.harvard.edu/abs/2018AJ....155...39O/abstract>`_ which
    contains 62x10^3 likely rotation periods based on a KELT photometry + TFA
    
  * Overlap with Kepler binarise from `Lurie et. al. (2017)
    <https://ui.adsabs.harvard.edu/abs/2017AJ....154..250L/abstract>`_?

  * others?

Need to argue
-------------

1. There will be enough systems that we can analyze to generate a big
   catalog, hopefully many thousands 

   * Filter from Prsa et all (2022) to estimate yield from 2x10^5 stamps:
     1,200 EBs with morph < 0.5 & primary eclipse depth > 1% & T < 12 (up to
     1,341 for T < 13, 1,389 for T < 13.5 per QLP)

   * Use some scaling to estimate yield from FFI: 10^7 QLP lightcurves for
     T<13.5. Justesen & Albrecht (2019) autodetected ~350 high quality EBs
     from southern hemisphere alone => ~700 from both hemispheres from
     stamps => ~35,000 from naive scaling. This is an overestimate
     because TESS targets were selected to:

       - avoid blending

       - be brighter on average

       - probably be biased to continuous viewing zone or zones with more
         coverage

     but not selected to target EBs deliberately. Probably safe to say FFIs
     will yield many thousands of well characterized EBs.

2. The precision of the data will be sufficient to get useful constraints:

       this is ensured from the selection criteria above (deep transits,
       well detached, bright => high photometric precision). Signal to noise
       ratio will be >> 10 for most EBs and always bigger than ~20.

3. We have the computing resources to do the analysis: 

   * With fast LC modeling TACC should be plenty

4. We will have the time to do the coding and article writing etc within the
   time frame of the project

   * LC models and already implemented

   * MCMC relies on existing package just have to define likelihood
     function

   * Calculations will run while articles are being written 


Work Plan
---------

* EBs from stamps

  * LC detrending unless using PDCSAP
  
  * Phase curve + BATMAN model: 1 month

  * Automatic candidate EB selection from stamps: 1 month

  * MCMC likelihood: 1 month
    
      * Extinction model

      * SED likelihood

      * LC likelihood

  * Manual selection of EBs: 1 month

  * Sampling: 3 months in parallel to EB selection

  * Analyzing results, validation against W19 and J&A other: 2 months
    parallel to sampling

  * Article: 2 months

* EBs from FFIs

  * LC detrending?

  * Adapt automatic candidate selection: 3 weeks

  * Adapt MCMC likelihood: 2 weeks (extinction and SED models are same)

  * Manual selection of EBs: 5 months

  * Sampling: 6 months in parallel to selection

  * Analyzing results, validation against W19 and J&A other: 2 months
    parallel to sampling

  * Article: 2 months

* Spin from stamp LCs

  * Custom LC detrending for stamps (what did Lurie do?) ?

  * Implement periodogram & autocorrelation searches like Lurie et. al.

  * Automated search of stamp LCs

  * Manual review of candidates from stamps: partially in parallel to
    automated search

  * Custom LC detrending from FFIs?

  * Automated search of FFI LCs: partially in parallel to stamp manual
    review 

  * Article: 2 months 
