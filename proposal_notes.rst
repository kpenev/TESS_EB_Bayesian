Goals
=====

* Fully characterized distributions of EB physical parameters from TESS
  photometry and SED information (GAIA + 2MASS + ... catalogues)

* Spin and differential rotation of TESS EB primary stars

* Detection of ellipsoidal variations, reflected light, doppler beaming

* Measure eclipsing timing variability (ETV). Possible causes:
  
  * Light travel time effect (LITE): a third body perturbing the center of mass
    of the binary system creates a light-time delay along the line of sight
    which can cause eclipses to appear earlier or later than expected

  * Non-hierarchical third body: the presence of a third body actually changes
    the period of the binary over time

  * Mass transfer: mass transfer between the com- ponents in the binary changes
    the period

  * Gravitational quadrupole coupling (Applegate effect): spin–orbit transfer of
    angular momen- tum in a close binary due to one of the stars being active
    produces period changes up to 10–5 times the binary period (Applegate 1992)

  * Apsidal motion: the rotation of the line of apsides causes a change in the
    time between primary and secondary eclipses even though the period remains
    unchanged (requires an eccentric orbit; see Apsidal Angle) 6.  Spurious
    signals: due to spots and other effects that distort the eclipsing binary
    light curve

Applications:
=============

* Measure Qstar:

  * control with long period systems to show no signature => no
    constraints

  * Comppare spin vs eccentricity constraints (same object different
    dominant freuencies)

  * Many more systems than `W19`_ => able to explore more dependencies, not just
    frequency.

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

* Study magnetic breaking from distribution of binaries at short orbital
  periods: `Kareem et. al. (2022)`_ 

* ETV can constrain k2, hence internal structure

* Limb darkening calibration and stellar atmospheres modeling

* Spectral synthesis for galaxy science:
  https://ui.adsabs.harvard.edu/abs/2017PASA...34...58E/abstract

* Study tides: interesting article:  Rotation Period Evolution in Low-mass
  Binary Stars: The Impact of Tidal Torques and Magnetic Braking 

* Dark Matter studies:
  https://ui.adsabs.harvard.edu/abs/2022ApJ...928L..17C/abstract

* `Reinhold et. al. (2013)
  <https://ui.adsabs.harvard.edu/abs/2013A%26A...560A...4R/abstract>`_,
  `Reinhold & Gizon (2015)
  <https://ui.adsabs.harvard.edu/abs/2015A%26A...583A..65R/abstract>`_, and
  `Lurie et. al. (2017)`_ all interpret additional peak in LS periodogram as
  differential rotation.

* Differential rotation is important to binary evolution in its own right, as it
  influences magnetic braking through surface activity and the magnetic dynamo
  (`Schatzman 1962
  <https://ui.adsabs.harvard.edu/abs/1962AnAp...25...18S/abstract>`_, `Adam et.
  al. 2020a <https://ui.adsabs.harvard.edu/abs/2020MNRAS.498.3782J/abstract>`_,
  `Adam et. al. 2020b
  <https://ui.adsabs.harvard.edu/abs/2020MNRAS.491..690J/abstract>`_)

* <++>HOW IS THIS CATALOG UNIQUE<++>

  * First catalog for TESS with detailed characterization of uncertainty
    distributions.

  * Will contain an order of magnitude more EBs than Kepler catalog and those
    will be brighter allowing much cheaper followup observations if necessary. 

  * 

Expected outcomes:
==================

* The precision of the data will be sufficient to get useful constraints: this
  is ensured from the selection criteria above (deep transits, well detached,
  bright => high photometric precision). Signal to noise ratio will be >> 10 for
  most EBs and always bigger than ~20.

* Number of binaries with well characterized physical parameters

  * Filter from `Prsa et. al. (2022)`_ to estimate yield from 2x10^5 stamps: 1,200
    EBs with morph < 0.5 & primary eclipse depth > 1% & T < 12 (up to 1,341
    for T < 13, 1,389 for T < 13.5 per `QLP`_)

  * Use some scaling to estimate yield from FFI: 10^7 `QLP`_ lightcurves for
    T<13.5. Justesen & Albrecht (2019) autodetected ~350 high quality EBs from
    southern hemisphere alone => ~700 from both hemispheres from stamps =>
    ~35,000 from naive scaling. This is an overestimate because TESS targets
    were selected to:

    - avoid blending

    - probably be biased to continuous viewing zone or zones with more
      coverage

    but not selected to target EBs deliberately. Probably safe to say FFIs
    will yield many thousands of well characterized EBs.

* Number of binaries with measured spins 

  * At least 10%-20% of stars will show detectable amplitude rotational
    variability.

    * `Claytor et. al. (2022)
      <https://ui.adsabs.harvard.edu/abs/2022ApJ...927..219C/abstract>`_ and
      references therein may be useful.

      * `Canto Martins et. al. (2020)`_ found 163 rotation signatures out of
        1000 KOI, with 113 having unambiguous rotation period measurements from
        just 1 sector of TESS.

    * `Oelkers et. al. (2018)`_ used
      KELT observatinos found 62,229 objects identified with likely stellar
      rotation periods with rms-amplitude from ∼3 mmag to ∼2.3 mag out of a
      subset of $4\times10^6$ sources (selected to be "likely TESS targets")
      with 47,174 having spin peridos below 13 days (likely upper limit for
      single sector based detections. TESS photometry is better than KELT, with
      less systematics. So we should detect a significantly bigger fraction than
      11%. 

  * At least 10%-20% of binaries with physical parameters will also have spin
    period detections: ~1000 systems or at least many hundreds. 

* Expected precision

  * masses: 3% or better

  * radii: 2% or better

  * ecosw: 1e-4, esinw: 1e-3

  * Connect to applications

    * eccentricity precision is crucial in tidal studies as it allows studying
      close to circular orbits where a single tidal wave dominates.

    * <++>CAN WE CONNECT TO ANY OF THE OTHERS?<++>

* Selection effects:

  * Physical parameters:

    * Pericenter separation should be large enough to be well separated
      (models break down below that). Can characterize by comparing PHOEBE
      with our modeling.
  
    * Period should be short enough to detect multiple consecutive eclipses to
      uniquely determine orbital period.
  
    * Stamps offer high cadence and select higher quality sources thus likely
      higher fractional yield. `Prsa et. al. (2022)`_ compiled comprehensive list
      to search. However, fully characterizing the selection effects for stamps is
      impossible, because targets were deliberately selected through the TESS
      guest investigator program during multiple cycles to observe known binaries
      using a complicated weighting scheme involving things like membership in
      othe catalogs, and scientific importance among more easy to account for
      factors, such as brightness and sky position.
  
    * FFI: everything is downloaded, and we will analyze all candidates flagged by
      other projects leading to hard to characterize selection effects. However,
      we will also use a straightforward algorithm for generating candidate list
      for manual review and a catalog of the automatically selected candidates
      will be provided. 
  
      * We will do injection-recovery simulations to characterize the selection
        effects of the automated search by injecting PHOEBE generated and `W19`_
        (Kepler has much highe S/N than TESS) transits in quiet TESS stars with
        the range of observing patterns as our targets and testing recovery.
  
      * The above procedure does not fully characterize the selection effects. The
        `QLP`_ post-processing may also suppress or otherwise modify astrophysical
        signals along with instrumental effects. Accounting for such effects would
        require injecting signals into raw light curves, or even at the pixel
        level and re-running `QLP`_. This is beyond the scope of this proposal.
        However, should anyone need to carry out such analysis, our automated
        selection algorithm will be publicly available as a pip installable python
        package and well documented. Reproducing the cantdidate selection of this
        effort will simply require running with `QLP`_ formatted lightcurves as
        input.

  * Spins:

    * We will search for spin signatures in:
     
      * all binaries selected for physical parameter characterization

      * all binaries from `Prsa et. al. (2022)`_

    * Injection recovery simulations will be performed on all binaries without
      spin detection:

      * Inject signals from `Lurie et. al. (2017)`_ with random phase (Kepler
        has much higher signal to noise and much longer time coverage than TESS)
        and repeat search to characterize fraction recovered and likelihood to
        mis-identify period.

      * Just like the physical parameter sample, this is not a complete
        characterization of the selection effects which need to be combined with
        PDC or `QLP`_ post-processing. However, that is outside the scope of
        this poposal. We will provide our spin analysis and automated flagging
        procedure as python package to enable other authors to carry out that
        task should it become necessary.
    
Methodology:
============

**EB physical parameters:**

* LC preparation: use the bolded options below and use others to validate max
  likelihood parameters:

  * Starting LCs:

    * PDCSAP: Used by `J&A`_
    
      **pros:** minimize instrumental effects, 

      **cons:** may modify phase curve.

    * **SAP: Used by** `W19`_ **and** `Prsa et. al. (2022)`_

      * `Prsa et. al. (2022)`_ referred to `Twicken et. al. (2010)`_, `Stumpe
        et. al.  (2012)`_, and the `Kepler Data Processing Handbook`_

   
    * For FFI use `QLP`_ lightcurves 
      
      * with detrending
  
      * **without detrending**

  * Include co-trending basis vectors from SPOC in LC model

  * Apply low-pass filter to LC and subtract the result from the original,
    preserving all frequency >= orbital frequency:

    **pros:** preserves all astrophysical variability with frequency >= to the
    orbital. Will also remove non-orbit related astrophysical variability on
    long timescales.

    **cons:** may leave instrumental effects.

  * Follow W19 to use only LC near eclipses together with polynomial out of
    eclipse model:

        Instead, we clipped the LC around each eclipse with a window 1.5–2.0
        times eclipse durations, which were initially taken from VKEBC, and then
        iteratively refined during the optimization process.

  * Candidate selection:

    * Use `Prsa et. al. (2022)`_ for stamps

    * Follow `J&A`_ automatic selection for FFI + a short dynesty run to find
      maximum likelihood parameters and reject badly fittings ones.

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

      * Simulate orbit in 2D apply rotation matrix to account for inclination
        and periapsis (use GPUs?)

    * For the very few very close (and high precision?) binaries can use
      PHOEBE if no other option. Investigate when we need to switch. Perhaps
      OK to drop.

    * Analysis will begin with fast model to find max likelihood values, but
      will not run to convergence. Manual inspection of max likelihood LC vs
      instrumental LC will catch few binaries requiring full PHOEBE.

* SED modeling options (choose one or perhps need mixture):

  * CDS isochrones, combining two isolated stars: 
    
    * provides TESS magnitudes for LC models

    * Have interpolation already working
  
  * Exctiction and reddening options: 
    
    * Per `J&A`_:

      We construct the SED from the apparent magnitudes, distance (adopted from
      Bailer-Jones et al. (2018) using Gaia DR2 parallaxes) and extinction
      listed in the TIC. We include Johnson–Cousins B and V, Two Micron All Sky
      Survey J, H, and K, and Wide-field Infrared Survey Explorer W1 and W2
      magnitudes where available. If no extinction is listed in the TIC, we
      query the 3D galactic dust map by `Green et al. (2019)`_ or the dust map by
      `Schlegel et al. (1998)`_ as the last priority. We convert apparent
      magnitudes to absolute magnitudes via the distance modulus. To obtain
      synthetic absolute magnitudes, we use the bolometric correction (BC)
      tables developed for the Bag of Stellar Tracks and Isochrones (BaSTI)
      isochrones (Hidalgo et al. 2018) and compute the combined absolute
      magnitudes of the two binary components using their effective temperatures
      and radii. We adopt the BC tables assuming solar metallicity and a surface
      gravity of logg=4.5. We note that the BC is mostly insensitive to
      metallicity and surface gravity.
      
    * Per `W19`_:

      To fit the archival data, we sum the predicted SEDs of both stellar
      components, compute the distance modulus, and correct for dust extinction
      along the line of sight assuming an exponential dust distribution with
      scale height h0 = 119 pc (Kruse & Agol 2014):
      
      $$\\mathrm{mag}_{\\lambda,\\mathrm{binary}} + 5\\log_{10}\\left(\\frac{d}{10pc}\\right) + A_\\lambda E(B-V) \\left[1-\\exp\\left(\\frac{-d\\sin b_G}{h_0}\\right)\\right]$$

      where b_G is a target’s Galactic latitude, d is distance in pc,
      and E(B − V) and A_\lambda are reddening and band specific extinction
      computed from a Milky Way extinction law with R = 3.1 (Fitzpatrick 1999).
      The integrated absolute magnitude of the binary in a given bandpass is

      <Equation combining flux from both stars>

      We use cross-matched Gaia distances derived from Bailer-Jones et al.
      (2018) and Schlafly & Finkbeiner (2011) dust maps results, when available,
      to place Gaussian priors on d and E(B − V) in our model. Accurate
      distances from parallax may be used to place better constraints on EB
      masses since mass correlates tightly with luminosity on the main sequence
      (MS). Well-calibrated EBs can be used as standard candles to calibrate
      parallaxes (Southworth, Maxted & Smalley 2005), and the converse should
      also be true for systems with well-constrained geometries (Stassun &
      Torres 2016). In particular, accurate distances may better constrain
      masses for binaries with non-total or non-annular eclipses by precisely
      determining total system luminosities. However, rather than using reported
      uncertainties as fixed σ d , σ E(B−V) , we allow the widths of these
      Gaussian priors to float to tolerate inaccuracies in the dust map or Gaia
      data due to source confusion or presence of tertiary companions, which has
      an occurrence rate of ∼15–20 per cent in the Kepler field (Gies et al.
      2012; Rappaport et al. 2013; Conroy et al.  2014; Orosz 2015). Frequent
      eclipses or nearby long-period binaries may also deleteriously affect the
      accuracy of Gaia astrometry.

    * CMD: "Using extinction coefficients computed star-by-star (except for the
      OBC case, which uses constant coefficients)": extinction of Av=1.0, with
      coefficients derived star-by-star, for Cardelli et al 89 + O'Donnell 94
      Rv=3.1 extinction curve <++>**READ THE PAPERS AND SEE HOW THAT WORKS.**

* Physical parameter validation and quality control:

  * From `Prsa et. al. (2022)`_ the following validation tests will be
    performed:

    * Centroid motion. The centroid test makes use of the open-source python
      package CONTAMINANTE , which executes a pixel-level modeling of the TESS
      target pixel files to determine the most likely location of the source of
      the eclipses. The score for this test is scaled inversely with the
      distance between the location-calculated source of signal and the location
      of the target star.

    * Contamination. The amount of contamination to the TESS aperture from
      nearby stars. This provides an indication of how crowded the field is and
      the likelihood of the signal originating from a nearby companion star.
      The score for this test is scaled inversely with the contamination of
      nearby source.

    * Out-of-transit variability. EBs with orbital periods shorter than around 3
      days are expected to show out-of-transit variability. As such, we search
      for this variability for the short-period candidates. The significance of
      such a detection is proportional to the score given for this test.
      Candidates with periods greater than 3 days are given a score of 0.5 for
      this test by default.

    * Archival classification. Candidates that have previously been listed as an
      EB on Simbad are awarded a score of 1 while all other candidates are
      awarded a score of 0.5.  Figure 7. Histogram showing the distribution of
      the individual test scores (dashed outlines), which combined, give the
      overall likelihood of the candidate being a real EB (solid black
      outlines).

    * TCE/TOI. The TESS automated search pipeline flags light curves containing
      a periodic signals, including both planetary and stellar, as
      threshold-crossing events (TCEs) and TCEs that pass a large number of
      rigorous planet- vetting tests are promoted to TOI status. As such,
      candidates that are TOIs are given a score of 0; candidates that are TCEs
      but not TOIs are given a score of 0.75, and candidates that are neither
      TOIs nor TCEs receive a score of 0.5.

  * Visual inspection to flag poorly fitting LCs or SEDs

  * Run PHOEBE 
    
    * with maximum likelihood parameters from MCMC to check if LC is consistent
      and see if BEER parameters are of the right order of magnitude

    * Use gradient descend from max likelihood parameters to check for bias in
      parameters due to simplified modeling
    
  * Compare overlaps to `W19`_

    * analyze small sample of Kepler LCs to compare algorithms

    * Find overlapping detections to compare to independent analysis applied on
      independent data

  * Compare to `J&A`_

    * Very similar analysis. Run on `J&A`_ short LCs to compare algorithm.
      
    * Compare to their full catalog 

  * Compare to `Prsa et. al. (2022)`_

  * Validate maximum likelihood parameters against other options of preparing
    the LC

  * <++>**WHAT DID ABOVE VALIDATE AGAINST**

**Spin measurements:**

* LC preparation:

  * Collective wisdom suggests starting with PDCSAP is the best option. Lurie
    used PDCSAP masking eclipses.

    **pro:** Minimizes the number of false positive detections

    **con:** Lose slow rotators

  * Options for dealing with binary effects:

    * Mask out eclipses: still leaves phase curve effectsat orbital frequency

    * Subtract full orbital model

    * Do both and compare results from the three period search methods on each
      version of the LC

  * As validation, start with SAP LC, remove binary effects with the two
    algorithms above and apply reconstructive TFA to build Lomb-Scargle.
    
    * Could be too computationally expensive.
    
    * Need to implement template star selection: identify LCs from single
      CCD-campaign combo, use HAT algorithm to select stars.

* Period search:

  * Follow Lurie to combine autocorrelation and Lomb-Scargle periodograms.

  * Most directly relevant paper: `Martins et. al. (2020)
    <https://ui.adsabs.harvard.edu/abs/2020ApJS..250...20C/abstract>`_

  * Add wavelets. A good starting point may be `Bravo et. al.  (2014)
    <https://ui.adsabs.harvard.edu/abs/2014A%26A...568A..34B/abstract>`_.  It
    suggest using 6-th order Morlett transform. This could serve as a check and
    perhaps additional way to flag candidates and improve analysis for stars
    with intermittent rotational variability.

  * In addition to spin, we will detect:

    * periodic or quasi-periodic (with wavelet analysis) pulsations

    * <++>**WHAT ELSE?**

* For spin validation:

  * ACF: Follow Lurie:

    compare the ACF peak heights of EBs with starspot modulations to the EBs
    without periodic out-of-eclipse variability. Following McQuil- lan et al.
    (2013), we define the peak height as the height of the ACF peak relative to
    the adjacent minima. Unlike the absolute height, the relative height is less
    susceptible to systematic effects in the light curve, such as long-term
    trends. The ACF has values between −1 and 1, so the relative peak height has
    values between 0 and 2.

  * LS: use bootstrap to define false alarm probability

  * Compare all three methods

  * Compare to Martins et. al. (2020) above and the associated `livig database
    <https://filtergraph.com/tess_rotation_tois>`_ (287 TOIs with "unambiguous
    rotation" flag)
    
  * Compare to KELT rotations: `Oelkers et al. (2018)
    <https://ui.adsabs.harvard.edu/abs/2018AJ....155...39O/abstract>`_ which
    contains 62x10^3 likely rotation periods based on a KELT photometry + TFA
    
  * Compare to overlap with Kepler binarise from `Lurie et. al. (2017)
    <https://ui.adsabs.harvard.edu/abs/2017AJ....154..250L/abstract>`_?

  * <++>**OTHERS?**

* Rotational or other stellar variability can affect physical parameter
  determination.

  * For timescales longer than the orbit our filtering will remove most of the
    signal, unless very close to the orbital frequency. It will not
    significantlny affect the determination of physical parameters.

  * For synchronized systems, rotational variability can masquarade as phase
    curve effects, especially for circular orbits where they become
    indistinguishable.

    * Rotational variability is usually more complicated than BEER (i.e. not
      just $\Omega_{orb}$ and $2\Omega_{orb}$ components (e.g. Lure et. al. 2017
      Fig. 1 top). 
      
    * Will not remain at exactly the same phase and will change shape over time,
      especially between repeat observations of later sectors.

    * Amplitude of rotational variability if reliably detected should be much
      larger than BEER effects. As a result, if erroneously interpreted as BEER
      during physical parameter fitting, the best fit phase curve amplitudes
      will be orders of magnitude off from PHOEBE calculations (see PHOEBE
      validation of LC models above).

    * EB parameter analysis based on only near eclipse LC should not be
      sensitive to rotational modulations (removed as trends). As a result,
      validating against that approarch will flag problematic cases and protect
      us against erroneous conclusions.

    * <++>**OTHER IDEAS HOW TO DEAL WITH THIS?**

  * As spin gets faster than the orbit the effect on physical parameters
    decreases quickly and rotational signal becomes detectable by our period
    search. For such systems <++>**PICK ONE OPTION**: 
    
    * rotational signal can be subtracted and physical parameters re-analyzed

    * do simultaneous analysis for physical parameters and spin


Catalog contents:
=================

  * TIC

  * RA

  * Dec

  * T (tess magnitude)

  * GR, GB, GG (gaia magnitudes)

  * Primary/secondary eclipse depths

  * LC residuals from maximum likelihood model

  * <physical parameters>

  * <spin periods>

  * manual vetting flag (passed, failed, not vetted)

Need to argue
=============

1. We have the computing resources to do the analysis: 

   * With fast LC modeling TACC should be plenty

2. We will have the time to do the coding and article writing etc within the
   time frame of the project

   * LC models and already implemented

   * MCMC relies on existing package just have to define likelihood
     function

   * Calculations will run while articles are being written 


Work Plan
=========

* EBs from stamps

  * LC preparation

  * Collect SED information. 
    
    * Crossmatch with GAIA already available. 

    * Is GAIA already crossmathed to other catalogs?
  
  * Phase curve + BATMAN model: 1 month

  * Automatic candidate EB selection from stamps: 1 month

  * MCMC likelihood: 1 month
    
    * Extinction model

    * SED likelihood

    * LC likelihood

  * Manual selection of EBs: 1 month

  * Sampling: 3 months in parallel to EB selection

  * Analyzing results, validation against `W19`_ and `J&A`_
    other: 2 months parallel to sampling

  * Article: 2 months

* EBs from FFIs

  * LC preparation

  * Collect SED information: 
    
    * Did `QLP`_ use GAIA positions for photometry?

    * Crossmatch to other catalogs from GAIA already done?

  * Adapt automatic candidate selection: 3 weeks

  * Adapt MCMC likelihood: 2 weeks (extinction and SED models are same)

  * Manual selection of EBs: 5 months

  * Sampling: 6 months in parallel to selection

  * Analyzing results, validation against `W19`_ and `J&A`_
    other: 2 months parallel to sampling

  * Article: 2 months

* Spin

  * Stamp LC preparation

  * Implement periodogram, wavelet, and autocorrelation

  * Automated selection of candidates signals from stamp LCs

  * Manual review of candidates from stamps: partially in parallel to
    automated search

  * FFI LC preparation

  * Automated search of FFI LCs: partially in parallel to stamp manual
    review 

  * Manual review of candidates

  * Investigate cases where rotational or pulsational variability may have
    affected physical parameter determination

  * Article: 2 months 

.. _`Bailer-Jones et al. (2018)`: https://ui.adsabs.harvard.edu/abs/2018AJ....156...58B

.. _`BATMAN`: https://ui.adsabs.harvard.edu/abs/2015PASP..127.1161K/abstract 

.. _`Canto Martins et. al. (2020)`: https://ui.adsabs.harvard.edu/abs/2020ApJS..250...20C/abstract

.. _`Green et al. (2019)`: https://ui.adsabs.harvard.edu/abs/2019ApJ...887...93G

.. _`J&A`: https://ui.adsabs.harvard.edu/abs/2021ApJ...912..123J/abstract

.. _`Kareem et. al. (2022)`: https://ui.adsabs.harvard.edu/abs/2022MNRAS.517.4916E/abstract

.. _`Kepler Data Processing Handbook`: https://ui.adsabs.harvard.edu/abs/2020ksci.rept....9J/abstract 

.. _`Lurie et. al. (2017)`: https://ui.adsabs.harvard.edu/abs/2017AJ....154..250L/abstract

.. _`Oelkers et. al. (2018)`: https://ui.adsabs.harvard.edu/abs/2018AJ....155...39O/abstract

.. _`Prsa et. al. (2022)`: https://ui.adsabs.harvard.edu/abs/2022ApJS..258...16P/abstract

.. _QLP: https://archive.stsci.edu/hlsp/qlp

.. _`Schlegel et al. (1998)`: https://ui.adsabs.harvard.edu/abs/1998ApJ...500..525S

.. _`Stumpe et. al. (2012)`: https://ui.adsabs.harvard.edu/abs/2012PASP..124..985S/abstract

.. _`Twicken et. al. (2010)`: https://ui.adsabs.harvard.edu/abs/2010SPIE.7740E..23T/abstract 

.. _W19: https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.1644W/abstract
