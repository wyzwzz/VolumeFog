#version 460 core

layout(binding = 0) uniform Params{
    mat4 MVP;
    vec4 LineColor;
};

layout(location = 0) out vec4 oFragColor;

void main(){
    oFragColor = LineColor;
}