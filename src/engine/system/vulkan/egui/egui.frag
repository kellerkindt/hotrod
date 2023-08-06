#version 450

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

layout(binding = 0, set = 0) uniform sampler2D font_texture;

// 0-255 sRGB  from  0-1 linear
vec3 srgb_from_linear(vec3 rgb) {
    bvec3 cutoff = lessThan(rgb, vec3(0.0031308));
    vec3 lower = rgb * vec3(3294.6);
    vec3 higher = vec3(269.025) * pow(rgb, vec3(1.0 / 2.4)) - vec3(14.025);
    return mix(higher, lower, vec3(cutoff));
}

// 0-255 sRGBA  from  0-1 linear
vec4 srgba_from_linear(vec4 rgba) {
    return vec4(srgb_from_linear(rgba.rgb), 255.0 * rgba.a);
}

// 0-1 gamma  from  0-1 linear
vec4 gamma_from_linear_rgba(vec4 linear_rgba) {
    return vec4(srgb_from_linear(linear_rgba.rgb) / 255.0, linear_rgba.a);
}

void main() {
    // The texture is set up with `SRGB8_ALPHA8`
    out_color = in_color * gamma_from_linear_rgba(texture(font_texture, in_uv));
}