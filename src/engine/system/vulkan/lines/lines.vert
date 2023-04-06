#version 450

layout(location = 0) in vec2 pos;
layout(location = 1) in vec4 color;

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants { vec2 screen_size; } push_constants;

void main() {
    out_color = color;

    gl_Position = vec4(
    2.0 * pos.x / push_constants.screen_size.x - 1.0,
    2.0 * pos.y / push_constants.screen_size.y - 1.0,
    0.0,
    1.0
    );

}