#version 460 core

#define PI 3.14159265

layout(location = 0) in vec2 iFragTexCoord;
layout(location = 1) in vec2 iScreenCoord;

layout(location = 0) out vec4 oFragColor;

layout(std140, binding = 0) uniform SkyViewParams{
    float Exposure;
    float WOverH;
};

layout(std140, binding = 1) uniform SkyViewPerFrameParams{
    vec3 ViewDir;
    float Scale; // tan(fov/2)
    vec3 ViewRight;
    int pad;
};

layout(binding = 0) uniform sampler2D SkyLUT;

vec3 WhitePointColorMapping(float exposure, in out vec3 color){
    const vec3 white_point = vec3(1.08241, 0.96756, 0.95003);
    return pow(vec3(1.0) - exp(-color / white_point * exposure), vec3(1.0 / 2.2));
}

void main() {
    vec3 view_up = normalize(cross(ViewRight, ViewDir));
    vec3 view_dir = normalize(ViewDir + WOverH * Scale * ViewRight * iScreenCoord.x + Scale * view_up * iScreenCoord.y);

    float phi = atan(view_dir.z, view_dir.x);
    float u = (phi < 0 ? phi + 2 * PI : phi) / (2 * PI);

    float theta = asin(view_dir.y);
    float v = 0.5 + 0.5 * sign(theta) * sqrt(abs(theta) / (PI / 2));

    vec3 sky_color = texture(SkyLUT, vec2(u, v)).rgb;

    sky_color = WhitePointColorMapping(Exposure, sky_color);

    oFragColor = vec4(sky_color, 1.0);
}
