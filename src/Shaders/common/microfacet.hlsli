#pragma once

#include "compat/common.h"

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
/// https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.2297rep=rep1type=pdf
/// See "Memo on Fresnel equations" from Sebastien Lagarde
/// for a really nice introduction.
/// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
float3 schlick_fresnel(const float3 F0, const float cos_theta) {
    return F0 + (1 - F0) * pow(max(1 - cos_theta, float(0)), float(5));
}
float schlick_fresnel(const float F0, const float cos_theta) {
    return F0 + (1 - F0) * pow(max(1 - cos_theta, float(0)), float(5));
}
float3 schlick_fresnel(const float3 F0, const float cos_theta, const float eta) {
    float h_dot_out_sq = 1 - (1 / (eta * eta)) * (1 - cos_theta * cos_theta);
    float3 F = 1;
    if (h_dot_out_sq > 0) {
        F = schlick_fresnel(F0, eta > 1 ? cos_theta : sqrt(h_dot_out_sq));
    }
    return F;
}

/// Fresnel equation of a dielectric interface.
/// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
/// n_dot_i: abs(cos(incident angle))
/// n_dot_t: abs(cos(transmission angle))
/// eta: eta_transmission / eta_incident
float fresnel_dielectric(float n_dot_i, float n_dot_t, float eta) {
    float rs = (n_dot_i - eta * n_dot_t) / (n_dot_i + eta * n_dot_t);
    float rp = (eta * n_dot_i - n_dot_t) / (eta * n_dot_i + n_dot_t);
    float F = (rs * rs + rp * rp) / 2;
    return F;
}

/// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
/// This is a specialized version for the code above, only using the incident angle.
/// The transmission angle is derived from
/// n_dot_i: cos(incident angle) (can be negative)
/// eta: eta_transmission / eta_incident
float fresnel_dielectric(float n_dot_i, float eta) {
    float n_dot_t_sq = 1 - (1 - n_dot_i * n_dot_i) / (eta * eta);
    if (n_dot_t_sq < 0) {
        // total internal reflection
        return 1;
    }
    float n_dot_t = sqrt(n_dot_t_sq);
    return fresnel_dielectric(abs(n_dot_i), n_dot_t, eta);
}

float GTR1(float n_dot_h, float alpha) {
    float a2 = alpha * alpha;
    float t = 1 + (a2 - 1) * n_dot_h * n_dot_h;
    return (a2 - 1) / (M_PI * log(a2) * t);
}

float GTR2(float n_dot_h, float roughness) {
    float alpha = roughness * roughness;
    float a2 = alpha * alpha;
    float t = 1 + (a2 - 1) * n_dot_h * n_dot_h;
    return a2 / (M_PI * t * t);
}

float GTR2(const float3 h_local, float alpha_x, float alpha_y) {
    float3 h_local_scaled = h_local / float3(alpha_x, alpha_y, 1);
    float h_local_scaled_len_sq = dot(h_local_scaled, h_local_scaled);
    return 1 / (M_PI * alpha_x * alpha_y * h_local_scaled_len_sq * h_local_scaled_len_sq);
}

float GGX(float n_dot_h, float roughness) {
    return GTR2(n_dot_h, roughness);
}

/// The masking term models the occlusion between the small mirrors of the microfacet models.
/// See Eric Heitz's paper "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
/// for a great explanation.
/// https://jcgt.org/published/0003/02/03/paper.pdf
/// The derivation is based on Smith's paper "Geometrical shadowing of a random rough surface".
/// Note that different microfacet distributions have different masking terms.
float smith_masking_gtr2(const float3 v_local, float roughness) {
    float alpha = roughness * roughness;
    float a2 = alpha * alpha;
    float3 v2 = v_local * v_local;
    float Lambda = (-1 + sqrt(1 + (v2.x * a2 + v2.y * a2) / v2.z)) / 2;
    return 1 / (1 + Lambda);
}

float smith_masking_gtr2(const float3 v_local, float alpha_x, float alpha_y) {
    float ax2 = alpha_x * alpha_x;
    float ay2 = alpha_y * alpha_y;
    float3 v2 = v_local * v_local;
    float Lambda = (-1 + sqrt(1 + (v2.x * ax2 + v2.y * ay2) / v2.z)) / 2;
    return 1 / (1 + Lambda);
}

float smith_masking_gtr1(const float3 v_local) {
    return smith_masking_gtr2(v_local, float(0.25), float(0.25));
}

/// See "Sampling the GGX Distribution of Visible Normals", Heitz, 2018.
/// https://jcgt.org/published/0007/04/01/
float3 sample_visible_normals(float3 local_dir_in, const float alpha, const float2 rnd_param) {
    bool flip = local_dir_in.z < 0;
    // The incoming direction is in the "ellipsodial configuration" in Heitz's paper
    if (flip) {
        // Ensure the input is on top of the surface.
        local_dir_in = -local_dir_in;
    }

    // Transform the incoming direction to the "hemisphere configuration".
    float3 hemi_dir_in = normalize(float3(alpha * local_dir_in.x, alpha * local_dir_in.y, local_dir_in.z));

    // Parameterization of the projected area of a hemisphere.
    // First, sample a disk.
    float r = sqrt(rnd_param.x);
    float phi = 2 * M_PI * rnd_param.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    // Vertically scale the position of a sample to account for the projection.
    float s = (1 + hemi_dir_in.z) / 2;
    t2 = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
    // Point in the disk space
    float3 disk_N = float3(t1, t2, sqrt(max(float(0), 1 - t1 * t1 - t2 * t2)));

    // Reprojection onto hemisphere -- we get our sampled normal in hemisphere space.
    float3x3 hemi_frame = makeOrthonormal(hemi_dir_in);
    float3 hemi_N = hemi_frame[0] * disk_N[0] + hemi_frame[1] * disk_N[1] + hemi_dir_in * disk_N[2];

    // Transforming the normal back to the ellipsoid configuration
    float3 dir_out = normalize(float3(alpha * hemi_N.x, alpha * hemi_N.y, max(float(0), hemi_N.z)));
    if (flip) dir_out = -dir_out;
    return dir_out;
}

float3 sample_visible_normals(float3 local_dir_in, const float alpha_x, const float alpha_y, const float2 rnd_param) {
    // The incoming direction is in the "ellipsodial configuration" in Heitz's paper
    bool flip = local_dir_in.z < 0;
    if (flip) {
        // Ensure the input is on top of the surface.
        local_dir_in = -local_dir_in;
    }

    // Transform the incoming direction to the "hemisphere configuration".
    float3 hemi_dir_in = normalize(float3(alpha_x * local_dir_in.x, alpha_y * local_dir_in.y, local_dir_in.z));

    // Parameterization of the projected area of a hemisphere.
    // First, sample a disk.
    float r = sqrt(rnd_param.x);
    float phi = 2 * M_PI * rnd_param.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    // Vertically scale the position of a sample to account for the projection.
    float s = (1 + hemi_dir_in.z) / 2;
    t2 = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
    // Point in the disk space
    float3 disk_N = float3(t1, t2, sqrt(max(float(0), 1 - t1 * t1 - t2 * t2)));

    // Reprojection onto hemisphere -- we get our sampled normal in hemisphere space.
    float3x3 hemi_frame = makeOrthonormal(hemi_dir_in);
    float3 hemi_N = hemi_frame[0] * disk_N[0] + hemi_frame[1] * disk_N[1] + hemi_dir_in * disk_N[2];

    // Transforming the normal back to the ellipsoid configuration
    float3 dir_out = normalize(float3(alpha_x * hemi_N.x, alpha_y * hemi_N.y, max(float(0), hemi_N.z)));
    if (flip) dir_out = -dir_out;
    return dir_out;
}
