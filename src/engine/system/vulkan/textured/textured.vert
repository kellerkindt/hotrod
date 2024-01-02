#version 450

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;

layout(binding = 101) uniform WindowProperties { vec2 screen_size; } window;

layout(location = 0) out vec2 out_uv;

void main() {
    gl_Position = vec4(
    2.0 * pos.x / window.screen_size.x - 1.0,
    2.0 * pos.y / window.screen_size.y - 1.0,
    0.0,
    1.0
    );


    out_uv = uv;
}