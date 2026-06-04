#version 450

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 outline_color;
layout(location = 2) in float outline_size;

layout(location = 0) out vec4 out_color;

layout(binding = 0, set = 0) uniform sampler2D bound_texture;

void main() {
    vec4 the_out_color = out_color = texture(bound_texture, in_uv);

    if (the_out_color.a < 1.0 && outline_size > 0.0) {
        vec2 texel_size = 1.0 / vec2(textureSize(bound_texture, 0));
        int radius = int(outline_size + 0.9);

        float max_alpha = 0.0;
        float min_distance = 9999.0;

        for (int x = -radius; x <= radius; ++x) {
            for (int y = -radius; y <= radius; ++y) {
                vec2 offset = vec2(x, y) * texel_size;
                float neighbor_alpha = texture(bound_texture, in_uv + offset).a;

                if (neighbor_alpha > 0.01) {
                    max_alpha = max(max_alpha, neighbor_alpha);
                    min_distance = min(min_distance, float(x*x + y*y));
                }
            }
        }

        float min_distance_sqrt = sqrt(min_distance);
        if (max_alpha > 0.0) {
            float alpha = 1.0 - smoothstep(0.0, outline_size, min_distance_sqrt);

            vec3 final_color = mix(outline_color, the_out_color.rgb, the_out_color.a);
            float final_alpha = the_out_color.a + alpha * (1.0 - the_out_color.a);

            out_color = vec4(final_color, final_alpha);
        }
    }
}