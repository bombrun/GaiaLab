/*
 * File containing code for the queries that are made to gaia_archives
 * :Author: LucaZampieri
 */

/* First query about gatting subsample of sources */
SELECT
	TOP 10000 source_id,random_index,ra,dec,parallax,pmra,pmdec,radial_velocity
	FROM gaiadr2.gaia_source
	WHERE parallax_over_error>10
		AND phot_g_mean_flux_over_error>50
		AND phot_bp_mean_flux_over_error>20
		AND phot_rp_mean_flux_over_error>20
		AND phot_bp_rp_excess_factor<(1.3+0.06*power(phot_bp_mean_mag-phot_rp_mean_mag,2))
		AND phot_bp_rp_excess_factor>(1+0.015*power(phot_bp_mean_mag-phot_rp_mean_mag,2))
		AND visibility_periods_used>8
		AND astrometric_excess_noise<1
	ORDER BY random_index
