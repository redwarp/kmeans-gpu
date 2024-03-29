// Let's do some order dithering.
// https://en.wikipedia.org/wiki/Ordered_dithering

struct Centroids {
    count: u32,
    data: array<vec4<f32>>,
};

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture : texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var color_indices: texture_2d<u32>;
@group(0) @binding(3) var<storage, read> centroids: Centroids;

const index_matrix: array<u32, 16> = array<u32, 16>(0u,  8u,  2u,  10u,
                                                    12u, 4u,  14u, 6u,
                                                    3u,  11u, 1u,  9u,
                                                    15u, 7u,  13u, 5u);

// #include functions/delta_e.wgsl

fn index_value(coords: vec2<u32>) -> f32 {
    let x = coords.x % 4u;
    let y = coords.y % 4u;
    let index = x + y * 4u;
    var mine = index_matrix;
    return f32(mine[index]) / 16.0;
}

fn two_closest_colors(color: vec4<f32>) -> array<vec4<f32>, 2> {
    var values: array<vec4<f32>, 2>;
    var closest = vec4<f32>(10000.0);
    var second_closest = vec4<f32>(10000.0);

    for (var i: u32 = 0u; i < centroids.count; i = i + 1u) {
        let temp = centroids.data[i];
        let temp_distance = distance_cie94(color.rgb, temp.rgb);
        if (temp_distance < distance_cie94(color.rgb, closest.rgb)){
            second_closest = closest;
            closest = temp;
        } else if (temp_distance < distance_cie94(color.rgb, second_closest.rgb)){
            second_closest = temp;
        }
    }
    values[0] = closest;
    values[1] = second_closest;

    return values;
}

fn dither(color: vec4<f32>, coords: vec2<u32>) -> vec4<f32> {
    // Based on https://en.wikipedia.org/wiki/Ordered_dithering
    // Maybe this threshold should be computed by a different shader first?
    var color_a: vec3<f32> = centroids.data[0].rgb;
    var color_b: vec3<f32> = centroids.data[1].rgb;
    var distance_a_b = distance_cie94(color_a, color_b);
    for (var i: u32 = 2u; i < centroids.count; i = i + 1u) {
        let distance_a = distance_cie94(centroids.data[i].rgb, color_a);
        let distance_b = distance_cie94(centroids.data[i].rgb, color_b);

        if(distance_a > distance_b && distance_a > distance_a_b) {
            distance_a_b = distance_a;
            color_b = centroids.data[i].rgb;
        } else if (distance_b > distance_a_b) {
            distance_a_b = distance_b;
            color_a = centroids.data[i].rgb;
        }
    }
    let threshold = vec3<f32>(distance_a_b / sqrt(f32(centroids.count)));

    let index_value = index_value(coords) - 0.5;

    let adjusted = color.rgb + threshold * index_value;
    
    var closest = vec3<f32>(10000.0);
    for (var i: u32 = 0u; i < centroids.count; i = i + 1u) {
        let temp = centroids.data[i].rgb;
        let temp_distance = distance_cie94(adjusted, temp);
        if (temp_distance < distance_cie94(adjusted, closest)){
            closest = temp;
        }
    }
    return vec4<f32>(closest, 1.0);
}

fn meld(color: vec4<f32>, coords: vec2<u32>) -> vec4<f32> {
    let closest_colors = two_closest_colors(color);
    let factor = distance_cie94(color.rgb, closest_colors[1].rgb) / distance_cie94(closest_colors[0].rgb, closest_colors[1].rgb);

    return factor * closest_colors[0] + (1.0 - factor) * closest_colors[1];
}

@compute
@workgroup_size(16, 16)
fn main_dither(
    @builtin(global_invocation_id) global_id : vec3<u32>,
) {
    let dimensions = textureDimensions(output_texture);
    let coords = global_id.xy;

    if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }
    
    if (centroids.count == 1u) {
        // Only one color, so nothing to meld.
        textureStore(output_texture, coords, centroids.data[0]);
        return;
    }

    let color = textureLoad(input_texture, coords, 0);

    textureStore(output_texture, coords, dither(color, coords));
}

@compute
@workgroup_size(16, 16)
fn main_meld(
    @builtin(global_invocation_id) global_id : vec3<u32>,
) {
    let dimensions = textureDimensions(output_texture);
    let coords = global_id.xy;

    if (coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    if (centroids.count == 1u) {
        // Only one color, so nothing to meld.
        textureStore(output_texture, coords, centroids.data[0]);
        return;
    }

    let color = textureLoad(input_texture, coords, 0);
    
    textureStore(output_texture, coords, meld(color, coords));
}