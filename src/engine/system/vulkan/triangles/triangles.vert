#version 450

layout(location = 0) in vec2 pos;

layout(location = 0) out vec4 out_color;

layout(binding = 101) uniform WindowProperties { vec2 screen_size; } window;
layout(push_constant) uniform PushConstants { vec4 color; } push_constants;

void main() {
    gl_Position = vec4(
    2.0 * pos.x / window.screen_size.x - 1.0,
    2.0 * pos.y / window.screen_size.y - 1.0,
    0.0,
    1.0
    );

    out_color = push_constants.color;
}