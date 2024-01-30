#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0)    in  vec2    pass_Position;
layout(location = 1)    in  vec4    pass_Color;
layout(location = 2)    in  vec2    pass_Center;
layout(location = 3)    in  float   pass_radius2;
layout(location = 4)    in  float   pass_corona2;
layout(location = 5)    in  float   pass_lateAlpha;

layout(location = 0)    out vec4    out_Color;


const float WIDTH_EDGE_LINE = 1.2f;


void main(void) {

    vec2  rel_Position  = pass_Position - pass_Center;

    float distance2    = (rel_Position.x * rel_Position.x) + (rel_Position.y * rel_Position.y);
    float edge         = WIDTH_EDGE_LINE;//max(pass_radius2, pass_corona2) * WIDTH_EDGE_LINE;

    distance2 = sqrt(distance2);

    if (distance2 < pass_radius2-edge) {
        out_Color = pass_Color;

        float val = 0.25f * (1.0f - smoothstep(0, pass_radius2-edge, distance2*0.8f));

        out_Color.rgb += val;
    }

    else if (distance2 < pass_radius2) {
        out_Color    = pass_Color;
        out_Color.r *= .9f;
        out_Color.g *= .9f;
        out_Color.b *= .9f;
    }

    else if (distance2 < pass_corona2-edge) {
        out_Color   = pass_Color;
        out_Color.a = (1.1f - smoothstep(
        pass_radius2,
        pass_corona2,
        distance2
        )) * .5f;
    }

    else if (distance2 <= pass_corona2) {
        out_Color   = pass_Color;
        out_Color.a = .125f;
    }

    else {
        // out_Color.a = 0.0f;
        discard;
    }


    out_Color.a = out_Color.a*.3f + out_Color.a*.7f*pass_lateAlpha;


    // http://en.wikibooks.org/wiki/GLSL_Programming/GLUT/Transparent_Textures
    if (out_Color.a <= 0.001f) {
        discard;
    }
}