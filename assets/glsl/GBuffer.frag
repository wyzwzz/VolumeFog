#version 460 core
layout(location = 0) in vec3 iFragPos;
layout(location = 1) in vec3 iFragNormal;
layout(location = 2) in vec3 iFragTangent;
layout(location = 3) in vec2 iUV;
layout(location = 4) in float iViewDepth;

layout(binding = 0 ) uniform sampler2D AlbedoMap;
layout(binding = 1) uniform sampler2D NormalMap;

layout(location = 0) out vec4 output0;
layout(location = 1) out vec4 output1;


vec2 msign( vec2 v )
{
    return vec2( (v.x>=0.0) ? 1.0 : -1.0,
    (v.y>=0.0) ? 1.0 : -1.0 );
}
// https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
// https://www.shadertoy.com/view/llfcRl
vec2 encodeNormal(vec3 nor)
{
    nor.xy /= ( abs( nor.x ) + abs( nor.y ) + abs( nor.z ) );
    nor.xy  = (nor.z >= 0.0) ? nor.xy : (1.0-abs(nor.yx))*msign(nor.xy);
    return nor.xy;
}

layout(binding = 0) uniform Transform{
    mat4 Model;
    mat4 ViewModel;
    mat4 ProjViewModel;
};

void main() {
    vec3 world_normal  = normalize(iFragNormal);
    vec3 world_tangent = normalize(iFragTangent);
    vec3 world_bitangent = normalize(cross(world_normal, world_tangent));

    //gamma space to linear space
    vec3 albedo = pow(texture(AlbedoMap, iUV).rgb, vec3(2.2));

    vec3 local_normal = normalize(2 * texture(NormalMap, iUV).xyz - vec3(1));
    vec3 normal = local_normal.x * world_tangent + local_normal.y * world_bitangent + local_normal.z * world_normal;
    vec2 oct_normal = encodeNormal(normalize(normal));

    //1/PI for cos weight hemisphere sampling
    vec3 color  = albedo / 3.14159265;
    float color1 = uintBitsToFloat(packHalf2x16(color.rb));
    float color2 = color.g;

    //ok, just need two rgba32f texture
    output0 = vec4(iFragPos, normal.x);
    output1 = vec4(color1, color2, iViewDepth, normal.y);

}
