#ifndef DISNEYDATA_H
#define DISNEYDATA_H

#ifdef __cplusplus
#include <Utils/hlslmath.hpp>
#endif

STM_NAMESPACE_BEGIN

struct DisneyMaterialData {
	static const uint gDataCount = 3;
	float4 data[gDataCount];

#ifdef __cplusplus
	float3 baseColor()           { return data[0].head<3>(); }
#else
	float3 baseColor()           { return data[0].rgb; }
#endif
	float emission()              { return data[0][3]; }
	float metallic()              { return data[1][0]; }
	float roughness()             { return data[1][1]; }
	float anisotropic()           { return data[1][2]; }
	float subsurface()            { return data[1][3]; }
	float clearcoat()             { return data[2][0]; }
	float clearcoatGloss()       { return data[2][1]; }
	float transmission()          { return data[2][2]; }
	float eta()                   { return data[2][3]; }
	float alpha()                 { return roughness()*roughness(); }

#ifdef __cplusplus
	void baseColor(const float3 v)     { data[0].head<3>() = v; }
#else
	SLANG_MUTATING
	void baseColor(const float3 v)     { data[0].rgb = v; }
#endif
	SLANG_MUTATING
	void emission(const float v)        { data[0][3] = v; }
	SLANG_MUTATING
	void metallic(const float v)        { data[1][0] = v; }
	SLANG_MUTATING
	void roughness(const float v)       { data[1][1] = v; }
	SLANG_MUTATING
	void anisotropic(const float v)     { data[1][2] = v; }
	SLANG_MUTATING
	void subsurface(const float v)      { data[1][3] = v; }
	SLANG_MUTATING
	void clearcoat(const float v)       { data[2][0] = v; }
	SLANG_MUTATING
	void clearcoatGloss(const float v)  { data[2][1] = v; }
	SLANG_MUTATING
	void transmission(const float v)    { data[2][2] = v; }
	SLANG_MUTATING
	void eta(const float v)             { data[2][3] = v; }
};

STM_NAMESPACE_END

#endif