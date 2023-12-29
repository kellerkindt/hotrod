#version 450

layout(location = 0) in vec2 in_uv;
layout(location = 1) in float shading;

layout(location = 0) out vec4 out_color;

layout(binding = 0, set = 0) uniform sampler2D bound_texture;

void main() {
    out_color = texture(bound_texture, in_uv) * (1.0 - shading);
}