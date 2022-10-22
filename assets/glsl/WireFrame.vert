#version 460 core
layout(location = 0) in vec3 iVertexPos;


layout(binding = 0) uniform Params{
    mat4 MVP;
    vec4 LineColor;
};

void main() {
    gl_Position = MVP * vec4(iVertexPos, 1.0);

}
