#version 450

// per vertex data
layout(location = 0) in vec2 pos;

// per instance data
layout(location = 1) in vec2 entity_pos;
layout(location = 2) in vec2 uv0;
layout(location = 3) in vec2 uv1;
layout(location = 4) in float size;
layout(location = 5) in vec3 outline_color;
layout(location = 6) in float outline_size;

layout(binding = 101) uniform WindowProperties { vec2 screen_size; } window;
layout(binding = 201) uniform WorldView2d { vec2 position; float zoom; } view;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_outline_color;
layout(location = 2) out float out_outline_size;

void main() {
    gl_Position = vec4(
    2.0 * (((view.zoom * ((pos.x * size) + entity_pos.x - view.position.x))) / window.screen_size.x),
    2.0 * (((view.zoom * ((pos.y * size) + entity_pos.y - view.position.y))) / window.screen_size.y),
    0.0,
    1.0
    );


    out_uv = mix(uv0, uv1, pos + 0.5);
    out_outline_color = outline_color;
    out_outline_size = outline_size;
}