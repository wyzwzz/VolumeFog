#version 460 core

#define PI 3.14159265

layout(location = 0) in vec2 iUV;

layout(location = 0) out vec4 oFragColor;

layout(std140, binding = 0) uniform AtmosphereProperties{
    vec3 RayleighScattering;
    float RayleighDensityH;

    float MieScattering;
    float MieAsymmetryG;
    float MieAbsorption;
    float MieDensityH;

    vec3 OzoneAbsorption;
    float OzoneCenterH;
    float OzoneWidth;

    float GroundRadius;
    float TopAtmosphereRadius;
    float pad;
};

layout(binding = 0) uniform sampler2D TransmittanceLUT;
layout(binding = 1) uniform sampler3D FroxelLUT;// rgba : in-scattering + tranmittance
layout(binding = 2) uniform sampler2D GBuffer0;
layout(binding = 3) uniform sampler2D GBuffer1;
layout(binding = 4) uniform sampler2D ShadowMap;
layout(binding = 5) uniform sampler2D BlueNoiseMap;


layout(binding = 1, std140) uniform TerrianParams{
    vec3 SunDir; float SunTheta;
    vec3 SunRadiance; float MaxAerialDist;
    vec3 ViewPos; float WorldScale;
    vec2 BlueNoiseUVFactor; vec2 JitterFactor;
    mat4 ShadowProjView;
    mat4 CameraProjView;
    int FrameIndex;
};

vec3 decodeNormal(vec2 v)
{
    vec3 nor = vec3(v, 1.0 - abs(v.x) - abs(v.y)); // Rune Stubbe's version,
    float t = max(-nor.z, 0.0);                     // much faster than original
    nor.x += (nor.x > 0.0) ? -t : t;                     // implementation of this
    nor.y += (nor.y > 0.0) ? -t : t;                     // technique

    return normalize( nor );
}

vec3 PostProcessColor(vec3 color)
{
    vec3 white_point = vec3(1.08241, 0.96756, 0.95003);
    float exposure = 10.0;
    return pow(vec3(1.0) - exp(-color / white_point * exposure), vec3(1.0 / 2.2));
}

float SafeSqrt(float a) {
    return sqrt(max(a, 0.0));
}

float ClampDistance(float d) {
    return max(d, 0.0);
}

float DistanceToTopAtmosphereBoundary(float r, float mu){
    float discriminant = r * r * (mu * mu - 1.0) + TopAtmosphereRadius * TopAtmosphereRadius;
    return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

// mapping form [0, 1] to [0.5 / n, 1 - 0.5 / n]
float GetTextureCoordFromUnitRange(float x, int texture_size) {
    return 0.5 / float(texture_size) + x * (1.0 - 1.0 / float(texture_size));
}

vec2 GetTransmittanceTextureUVFromRMu(float r, float mu, in ivec2 res){
    float H =  sqrt(TopAtmosphereRadius * TopAtmosphereRadius - GroundRadius * GroundRadius);
    float rho = SafeSqrt(r * r - GroundRadius * GroundRadius);
    float d = DistanceToTopAtmosphereBoundary(r, mu);
    float d_min = (TopAtmosphereRadius - r);
    float d_max = rho + H;
    float x_mu = (d - d_min) / (d_max - d_min);
    float x_r = rho / H;
    return vec2(GetTextureCoordFromUnitRange(x_mu, res.x), GetTextureCoordFromUnitRange(x_r, res.y));
}

// theta is view with horizon
vec3 GetTransmittance(float h, float theta){
    float r = h * 0.99  + GroundRadius;
    float mu = cos(PI / 2 - theta);
    vec2 uv = GetTransmittanceTextureUVFromRMu(r, mu, textureSize(TransmittanceLUT, 0));
    return texture(TransmittanceLUT, uv).rgb;
}

void main(){
    vec4 p0 = texture(GBuffer0, iUV);
    vec4 p1 = texture(GBuffer1, iUV);
    if(all(equal(p0,vec4(0))) && all(equal(p1, vec4(0)))){
        discard;
    }
    vec3 world_pos = p0.xyz;
    vec2 oct_normal = vec2(p0.w, p1.w);
    vec3 normal = decodeNormal(oct_normal);
    vec2 color1 = unpackHalf2x16(floatBitsToUint(p1.x));
    vec3 albedo = vec3(color1.x, p1.g, color1.y);

    vec4 ndc = CameraProjView * vec4(world_pos, 1.0);
    ndc.xyz /= ndc.w;
    vec3 clip_pos = ndc.xyz * vec3(0.5, -0.5, 0.5) + 0.5;
    float z = WorldScale * distance(ViewPos, world_pos) / MaxAerialDist;
    vec2 bn_uv = clip_pos.xy * BlueNoiseUVFactor;
    vec2 bn = texture(BlueNoiseMap, bn_uv).xy;
    vec2 offset = JitterFactor * bn.x * vec2(cos(2 * PI * bn.y), sin(2 * PI * bn.y));
    vec4 aerial_res = texture(FroxelLUT, vec3(clip_pos.xy + offset, z));

    vec3 in_scattering = aerial_res.rgb;
    float view_transmiitance = aerial_res.w;

    vec3 sun_transmittance = GetTransmittance(world_pos.y * WorldScale, SunTheta);

    vec4 shadow_ndc_pos = ShadowProjView * vec4(world_pos + 0.03 * normal, 1.0);
    vec3 shadow_clip_pos = shadow_ndc_pos.xyz / shadow_ndc_pos.w;
    vec2 shadow_uv = 0.5 + 0.5 * shadow_clip_pos.xy;
    float shadow_factor = 1.0;
    if(all(equal(clamp(shadow_uv, vec2(0), vec2(1)), shadow_uv))){
        float test_z = shadow_clip_pos.z * 0.5 + 0.5;
        float shadow_z = texture(ShadowMap, shadow_uv).r;
        // compress lost, need to minus bias
        shadow_factor = float(test_z - 0.01  <= shadow_z);
    }

    vec3 color = SunRadiance * (
        in_scattering +
        shadow_factor * sun_transmittance * albedo * max(0, dot(normal, -SunDir) * view_transmiitance)
    );

    color = PostProcessColor(color);

    gl_FragDepth = clip_pos.z;
    oFragColor = vec4(color, 1.0);
}