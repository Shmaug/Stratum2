#ifndef MICROFACET_H
#define MICROFACET_H

/// A microfacet model assumes that the surface is composed of infinitely many little mirrors/glasses.
/// The orientation of the mirrors determines the amount of lights reflected.
/// The distribution of the orientation is determined empirically.
/// The distribution that fits the best to the data we have so far (which is not a lot of data)
/// is from Trowbridge and Reitz's 1975 paper "Average irregularity representation of a rough ray reflection",
/// wildly known as "GGX" (seems to stand for "Ground Glass X" https://twitter.com/CasualEffects/status/783018211130441728).
///
/// We will use a generalized version of GGX called Generalized Trowbridge and Reitz (GTR),
/// proposed by Brent Burley and folks at Disney (https://www.disneyanimation.com/publications/physically-based-shading-at-disney/)
/// as our normal distribution function. GTR2 is equivalent to GGX.

/// Schlick's Fresnel equation approximation
/// from "An Inexpensive BRDF Model for Physically-based Rendering", Schlick
/// https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.2297&rep=rep1&type=pdf
/// See "Memo on Fresnel equations" from Sebastien Lagarde
/// for a really nice introduction.
/// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
inline Real schlick_fresnel1(const Real F0, const Real cos_theta) {
    return F0 + (1 - F0) * pow(max(Real(1) - cos_theta, Real(0)), Real(5));
}
inline Spectrum schlick_fresnel3(const Spectrum F0, const Real cos_theta) {
    return F0 + (1 - F0) * pow(max(Real(1) - cos_theta, Real(0)), Real(5));
}

/// Fresnel equation of a dielectric interface.
/// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
/// n_dot_i: abs(cos(incident angle))
/// n_dot_t: abs(cos(transmission angle))
/// eta: eta_transmission / eta_incident
inline Real fresnel_dielectric(const Real n_dot_i, const Real n_dot_t, const Real eta) {
    const Real rs = (n_dot_i - eta * n_dot_t) / (n_dot_i + eta * n_dot_t);
    const Real rp = (eta * n_dot_i - n_dot_t) / (eta * n_dot_i + n_dot_t);
    const Real F = (rs * rs + rp * rp) / 2;
    return F;
}

/// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
/// This is a specialized version for the code above, only using the incident angle.
/// The transmission angle is derived from
/// n_dot_i: cos(incident angle) (can be negative)
/// eta: eta_transmission / eta_incident
inline Real fresnel_dielectric(const Real n_dot_i, const Real eta) {
    const Real n_dot_t_sq = 1 - (1 - n_dot_i * n_dot_i) / (eta * eta);
    if (n_dot_t_sq < 0) {
        // total internal reflection
        return 1;
    }
    const Real n_dot_t = sqrt(n_dot_t_sq);
    return fresnel_dielectric(abs(n_dot_i), n_dot_t, eta);
}

inline Real GTR2(const Real n_dot_h, const Real alpha) {
    const Real a2 = alpha * alpha;
    const Real t = 1 + (a2 - 1) * n_dot_h * n_dot_h;
    return a2 / (M_PI * t*t);
}

/// The masking term models the occlusion between the small mirrors of the microfacet models.
/// See Eric Heitz's paper "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
/// for a great explanation.
/// https://jcgt.org/published/0003/02/03/paper.pdf
/// The derivation is based on Smith's paper "Geometrical shadowing of a random rough surface".
/// Note that different microfacet distributions have different masking terms.
inline Real smith_masking_gtr2(const Vector3 w, const Real alpha) {
    const Real a2 = alpha * alpha;
    const Vector3 v2 = w * w;
    const Real Lambda = (-1 + sqrt(1 + (v2[0] * a2 + v2[1] * a2) / v2[2])) / 2;
    return 1 / (1 + Lambda);
}

/// See "Sampling the GGX Distribution of Visible Normals", Heitz, 2018.
/// https://jcgt.org/published/0007/04/01/
inline Vector3 sample_visible_normals(Vector3 local_dir_in, const Real alpha_x, const Real alpha_y, const float2 rnd_param) {
    // The incoming direction is in the "ellipsodial configuration" in Heitz's paper
	const bool inside = local_dir_in[2] < 0;
	// Ensure the input is on top of the surface.
    if (inside) local_dir_in = -local_dir_in;

    // Transform the incoming direction to the "hemisphere configuration".
    const Vector3 hemi_dir_in = normalize(Vector3(alpha_x * local_dir_in[0], alpha_y * local_dir_in[1], local_dir_in[2]));

    // Parameterization of the projected area of a hemisphere.
    // First, sample a disk.
    const Real r = sqrt(rnd_param[0]);
    const Real phi = 2 * M_PI * rnd_param[1];
    const Real t1 = r * cos(phi);
    Real t2 = r * sin(phi);
    // Vertically scale the position of a sample to account for the projection.
    const Real s = (1 + hemi_dir_in[2]) / 2;
    t2 = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
    // Point in the disk space
    const Vector3 disk_N = Vector3(t1, t2, sqrt(max(Real(0), 1 - t1*t1 - t2*t2)));

    // Reprojection onto hemisphere -- we get our sampled normal in hemisphere space.
    Vector3 T1,T2;
    make_orthonormal(hemi_dir_in, T1, T2);
    const Vector3 hemi_N = disk_N[0]*T1 + disk_N[1]*T2 + disk_N[2]*hemi_dir_in;

    // Transforming the normal back to the ellipsoid configuration
    Vector3 N = normalize(Vector3(alpha_x * hemi_N[0], alpha_y * hemi_N[1], max(0.f, hemi_N[2])));
    if (inside) N = -N;
    return N;
}

#endif