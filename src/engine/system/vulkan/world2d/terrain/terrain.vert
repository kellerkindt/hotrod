#version 450

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in float shading;

layout(binding = 101) uniform WindowProperties { vec2 screen_size; } window;
layout(binding = 201) uniform WorldView2d { vec2 position; } view;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out float out_shading;

void main() {
    gl_Position = vec4(
    2.0 * (view.position.x + pos.x) / window.screen_size.x - 1.0,
    2.0 * (view.position.y + pos.y) / window.screen_size.y - 1.0,
    0.0,
    1.0
    );


    out_uv = uv;
    out_shading = shading;
}