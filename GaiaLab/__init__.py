"""
ESAC is responsible for developing and running AGIS, the software that
computes the global astrometric parameters for the Gaia mission. The
design and validation of Gaia global astrometric mission requires to be
able to run simulations that include complex calibration issues.
The current state of the art is AgisLab. This code is proprietary of
DPAC, the scientific consortium processing the Gaia data and
responsible for the publication of the final star catalogue.

GaiaLab project is open source, developed by students and going some
steps further in order to expose some of the global astrometric issues
to a larger community.

The first version will be based on a very simple model :

    single source
    one ccd
    circular satellite orbit
    Newtonian physics (no relativity)

The two packages included in GaiaLab are hits and scan. scan is an
implimentation of the global astrometric solution and nominal scanning
law for Gaia. hits is a micrometeroid impact detection and simulation
package.
"""
from . import hits, scan
