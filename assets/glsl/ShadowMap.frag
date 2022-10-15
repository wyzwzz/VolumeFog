#version 460 core

layout(location = 0) out float oFragDepth;

void main() {
    oFragDepth = gl_FragCoord.z;
}
