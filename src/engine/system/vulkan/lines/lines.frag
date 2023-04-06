#version 450

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 in_xy;
layout(location = 2) in float width;

layout(pixel_center_integer) gl_FragCoord;

layout(location = 0) out vec4 out_color;

const float RATIO = 0.65;

void main() {
    vec4 color = in_color;


    float dist = distance(in_xy, gl_FragCoord.xy) / (width / 4.0);

    if (dist > RATIO) {
        // color.a *= smoothstep(0, 1, 1 - ((dist - RATIO) / (1.0 - RATIO)));
        // color.a *= 1 - smoothstep(0, 1- RATIO, (dist - RATIO));
        // color.a *= 1.0 - (dist - RATIO) / (1.0 - RATIO);
        color.a *= 1.0 - dist;

    }

    // color.a = 1 - (distance(in_xy, gl_FragCoord.xy) / (width / 2.0));
    out_color = color;
}