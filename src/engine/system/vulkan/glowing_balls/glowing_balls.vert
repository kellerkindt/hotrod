#version 450
#extension GL_ARB_separate_shader_objects : enable

// per vertex data
layout(location = 0) in vec2 pos;

// per instance data
layout(location = 1) in vec2  instance_pos;
layout(location = 2) in vec4  instance_color;
layout(location = 3) in float instance_radius;
layout(location = 5) in float instance_corona;
layout(location = 6) in float instance_lateAlpha;

layout(binding = 101) uniform WindowProperties { vec2 screen_size; } window;
layout(binding = 201) uniform WorldView2d { vec2 position; float zoom; } view;

layout(location = 0)    out vec2    pass_Position;
layout(location = 1)    out vec4    pass_Color;
layout(location = 2)    out vec2    pass_Center;
layout(location = 3)    out float   pass_radius2;
layout(location = 4)    out float   pass_corona2;
layout(location = 5)    out float   pass_lateAlpha;


void main(void) {
    float size = instance_radius + instance_corona;

    gl_Position = vec4(
    2.0 * (((view.zoom * ((size * pos.x) + instance_pos.x - view.position.x))) / window.screen_size.x),
    2.0 * (((view.zoom * ((size * pos.y) + instance_pos.y - view.position.y))) / window.screen_size.y),
    0.0,
    1.0
    );

    pass_Position = vec2(
    (size * pos.x) + instance_pos.x,
    (size * pos.y) + instance_pos.y
    );

    pass_Center = instance_pos;

    pass_Color      = instance_color;
    pass_radius2   = instance_radius * 0.5;
    pass_corona2   = instance_corona * 0.5;
    pass_lateAlpha = instance_lateAlpha;
}