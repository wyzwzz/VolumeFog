#version 460 core
layout(location = 0) in vec3 iVertexPos;


layout(binding = 0) uniform Transform{
    mat4 Model;
    mat4 MVP;
};

void main() {
    gl_Position = MVP * vec4(iVertexPos, 1.0);

}
