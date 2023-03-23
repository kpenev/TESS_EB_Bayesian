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
          review. Only select well detached high signal to noise EBs. Can we do
          injection-recovery simulations to characterize selection?

        * What additional selection effects we impose and how will we
          characterize them

            * physical parameter sample

            * spin sample

            * differences of selection effects between short and long cadence
              LCs


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
        https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.1953D/abstract, 
        https://ui.adsabs.harvard.edu/abs/2012MNRAS.427..127B/abstract
        https://ui.adsabs.harvard.edu/abs/2016ApJ...832..121G/abstract


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


Methodology:
============

    * LC modeling:

        * For well separated binaries (vast majority): use BATMAN + BEER for LC
          modeling (test speed or ask Simon).

        * For the very few very close (and high precision?) binaries can use
          PHOEBE if no other option.  Investigate when we need to switch.

    * SED modeling options (choose one or perhps need mixture):
        
        * PHOEBE
          
        * CDS isochrones, combining two isolated stars

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

    3. We have the computing resources to do the analysis

    4. We will have the time to do the coding and article writing etc within the
       time frame of the project
