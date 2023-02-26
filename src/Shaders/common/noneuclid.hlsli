#include "compat/transform.h"
#include "compat/scene.h"

float dotProduct(float4 u, float4 v, const float lorentzSign) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2] + lorentzSign * u[3] * v[3];
}

float4 direction(float4 to, float4 from, const float lorentzSign) {
    if (lorentzSign > 0) {
        float cosd = dotProduct(from, to, lorentzSign);
        float sind = sqrt(1 - cosd * cosd);
        return (to - from * cosd) / sind;
    }
    if (lorentzSign < 0) {
        float coshd = -dotProduct(from, to, lorentzSign);
        float sinhd = sqrt(coshd * coshd - 1);
        return (to - from * coshd) / sinhd;
    }
    return normalize(to - from);
}

float4 euclidianToCurved(float4 eucPoint, const float lorentzSign) {
    const float3 P = eucPoint.xyz;
    const float distance = length(P);
    if (distance < 1e-5f) return eucPoint;
    if (lorentzSign > 0) return float4(P / distance * sin(distance), cos(distance));
    if (lorentzSign < 0) return float4(P / distance * sinh(distance), cosh(distance));
    return eucPoint;
}

float4 euclidianToCurved(float3 eucPoint, const float lorentzSign) {
    return euclidianToCurved(float4(eucPoint[0], eucPoint[1], eucPoint[2], 1), lorentzSign);
}

float4x4 TranslateMatrix(float4 to, const float lorentzSign) {
    if (lorentzSign != 0) {
        const float denom = 1 / (1 + to[3]);
        return transpose(float4x4(1 - lorentzSign * to[0] * to[0] * denom,    -lorentzSign * to[0] * to[1] * denom,    -lorentzSign * to[0] * to[2] * denom, -lorentzSign * to[0],
                                     -lorentzSign * to[1] * to[0] * denom, 1 - lorentzSign * to[1] * to[1] * denom,    -lorentzSign * to[1] * to[2] * denom, -lorentzSign * to[1],
                                     -lorentzSign * to[2] * to[0] * denom,    -lorentzSign * to[2] * to[1] * denom, 1 - lorentzSign * to[2] * to[2] * denom, -lorentzSign * to[2],
                                  to[0], to[1], to[2], to[3]));
    }
    return transpose(float4x4(1, 0, 0, 0,
                              0, 1, 0, 0,
                              0, 0, 1, 0,
                              to[0], to[1], to[2], 1));
}

float4x4 ViewMat(const float3 cameraPosition, const TransformData worldToCamera, const float lorentzSign) {
    const float4 ic = float4(worldToCamera.m[0].xyz, 0);
    const float4 jc = float4(worldToCamera.m[1].xyz, 0);
    const float4 kc = float4(worldToCamera.m[2].xyz, 0);

    const float4 geomEye = euclidianToCurved(cameraPosition, lorentzSign);

    const float4x4 eyeTranslate = TranslateMatrix(geomEye, lorentzSign);
    const float4 icp = mul(eyeTranslate, ic);
    const float4 jcp = mul(eyeTranslate, jc);
    const float4 kcp = mul(eyeTranslate, kc);

    if (abs(lorentzSign) < 1e-4) {
        return transpose(float4x4(
            icp[0], jcp[0], kcp[0], 0,
            icp[1], jcp[1], kcp[1], 0,
            icp[2], jcp[2], kcp[2], 0,
            -dotProduct(icp, geomEye, lorentzSign), -dotProduct(jcp, geomEye, lorentzSign), -dotProduct(kcp, geomEye, lorentzSign), 1 ));
    }
    return transpose(float4x4(
        icp[0], jcp[0], kcp[0], lorentzSign * geomEye[0],
        icp[1], jcp[1], kcp[1], lorentzSign * geomEye[1],
        icp[2], jcp[2], kcp[2], lorentzSign * geomEye[2],
        lorentzSign * icp[3], lorentzSign * jcp[3], lorentzSign * kcp[3], geomEye[3] ));
}