#pragma once

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
inline float schlick_fresnel1(const float F0, const float cosTheta) {
    return F0 + (1 - F0) * pow(max(1 - cosTheta, 0), 5);
}
inline float3 schlick_fresnel3(const float3 F0, const float cosTheta) {
    return F0 + (1 - F0) * pow(max(1 - cosTheta, 0), 5);
}

/// Fresnel equation of a dielectric interface.
/// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
/// n_dot_i: abs(cos(incident angle))
/// n_dot_t: abs(cos(transmission angle))
/// eta: eta_transmission / eta_incident
inline float fresnel_dielectric(const float n_dot_i, const float n_dot_t, const float eta) {
    const float rs = (n_dot_i - eta * n_dot_t) / (n_dot_i + eta * n_dot_t);
    const float rp = (eta * n_dot_i - n_dot_t) / (eta * n_dot_i + n_dot_t);
    const float F = (rs * rs + rp * rp) / 2;
    return F;
}

/// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
/// This is a specialized version for the code above, only using the incident angle.
/// The transmission angle is derived from
/// n_dot_i: cos(incident angle) (can be negative)
/// eta: eta_transmission / eta_incident
inline float fresnel_dielectric(const float n_dot_i, const float eta) {
    const float n_dot_t_sq = 1 - (1 - n_dot_i * n_dot_i) / (eta * eta);
    if (n_dot_t_sq < 0) {
        // total internal reflection
        return 1;
    }
    const float n_dot_t = sqrt(n_dot_t_sq);
    return fresnel_dielectric(abs(n_dot_i), n_dot_t, eta);
}

inline float GTR2(const float n_dot_h, const float alpha) {
    const float a2 = alpha * alpha;
    const float t = 1 + (a2 - 1) * n_dot_h * n_dot_h;
    return a2 / (M_PI * t*t);
}

/// The masking term models the occlusion between the small mirrors of the microfacet models.
/// See Eric Heitz's paper "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
/// for a great explanation.
/// https://jcgt.org/published/0003/02/03/paper.pdf
/// The derivation is based on Smith's paper "Geometrical shadowing of a random rough surface".
/// Note that different microfacet distributions have different masking terms.
inline float smith_masking_gtr2(const float3 w, const float alpha) {
    const float a2 = alpha * alpha;
    const float3 v2 = w * w;
    const float Lambda = (-1 + sqrt(1 + (v2[0] * a2 + v2[1] * a2) / v2[2])) / 2;
    return 1 / (1 + Lambda);
}

/// See "Sampling the GGX Distribution of Visible Normals", Heitz, 2018.
/// https://jcgt.org/published/0007/04/01/
inline float3 sampleVisibleNormals(float3 localDirIn, const float alphaX, const float alphaY, const float2 rnd) {
    // The incoming direction is in the "ellipsodial configuration" in Heitz's paper
	const bool inside = localDirIn[2] < 0;
	// Ensure the input is on top of the surface.
    if (inside) localDirIn = -localDirIn;

    // Transform the incoming direction to the "hemisphere configuration".
    const float3 hemiDirIn = normalize(float3(alphaX * localDirIn[0], alphaY * localDirIn[1], localDirIn[2]));

    // Parameterization of the projected area of a hemisphere.
    // First, sample a disk.
    const float r = sqrt(rnd[0]);
    const float phi = 2 * M_PI * rnd[1];
    const float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    // Vertically scale the position of a sample to account for the projection.
    const float s = (1 + hemiDirIn[2]) / 2;
    t2 = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
    // Point in the disk space
    const float3 diskN = float3(t1, t2, sqrt(max(0, 1 - t1*t1 - t2*t2)));

    // Reprojection onto hemisphere -- we get our sampled normal in hemisphere space.
    const float3x3 frame = makeOrthonormal(hemiDirIn);
	const float3 hemiN = frame[0] * diskN.x + frame[1] * diskN.y + frame[2] * diskN.z;

    // Transforming the normal back to the ellipsoid configuration
    float3 N = normalize(float3(alphaX * hemiN[0], alphaY * hemiN[1], max(0, hemiN[2])));
    if (inside) N = -N;
    return N;
}