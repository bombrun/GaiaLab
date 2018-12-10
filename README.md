# GaiaLab

ESAC is responsible for developing and running AGIS, the software that computes the global astrometric parameters for the Gaia mission.
The design and validation of Gaia global astrometric mission requires to be able to run simulations that include complex calibration issues.
The current state of the art is AgisLab. This code is proprietary of DPAC, the scientific consortium processing the Gaia data
and responsible for the publication of the final star catalogue.

GaiaLab project is open source, developed by students and going some steps further in the simplification of the problem in order to expose some of the global astrometric issues to a larger community.

The first version will be based on a very simple model :
* single source
* one ccd
* circular satellite orbit
* Newtonian physic (no relativity)

The project will make used of the technical notes written by Lennart Lindegren http://www.astro.lu.se/~lennart/Astrometry/TN.html

In particular the following notes and papers:
- (Lindegren, SAG-LL-14)
- (Lindegren, SAG-LL-30)
- (Lindegren, SAG-LL-35)
- The astrometric core solution for the gaia mission, overview of models, algorithms,
and software implementation, L.Lindegren et al.

The codestyle tries to follow PEP8 guidelines, for example using linter 2.2 codestyle package. (see https://atom.io/packages/linter as of 20.09.2018)

The documentation can be created setting up the **doc** folder. See http://www.sphinx-doc.org/en/master/usage/extensions/viewcode.html for setting up the :source: link!
