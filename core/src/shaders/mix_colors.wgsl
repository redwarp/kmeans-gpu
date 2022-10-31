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

let index_matrix: array<i32, 16> = array<i32, 16>(0,  8,  2,  10,
                                                  12, 4,  14, 6,
                                                  3,  11, 1,  9,
                                                  15, 7,  13, 5);

fn index_value(coords: vec2<i32>) -> f32 {
    let x = coords.x % 4;
    let y = coords.y % 4;
    let index = x + y * 4;
    var mine = index_matrix;
    return f32(mine[index]) / 16.0;
}

fn distance_cie94(one: vec3<f32>, second: vec3<f32>) -> f32{
    let kL = 1.0;
    let K1 = 0.045;
    let K2 = 0.015;
    let dL = one.r - second.r;
    let da = one.g - second.g;
    let db = one.b - second.b;

    let C1 = sqrt(one.g * one.g + one.b * one.b);
    let C2 = sqrt(second.g * second.g + second.b * second.b);
    let dCab = C1 - C2;

    var dHab = (da * da) + (db * db) - (dCab * dCab);
    // var dHab = pow(distance(one, second), 2.0) - dL * dL - dCab * dCab;
    if (dHab < 0.0) {
        dHab = 0.0;
    } else {
        dHab = sqrt(dHab);
    }

    let kC = 1.0;
    let kH = 1.0;

    let SL = 1.0;
    let SC = 1.0 + K1 * C1;
    let SH = 1.0 + K2 * C1;

    let i = pow(dL/(kL * SL), 2.0) + pow(dCab/(kC * SC), 2.0) + pow(dHab/(kH * SH), 2.0);
    if (i < 0.0) {
        return 0.0;
    } else {
        return sqrt(i);
    }
}

fn two_closest_colors(color: vec4<f32>) -> array<vec4<f32>, 2> {
    var values: array<vec4<f32>, 2>;
    var closest = vec4<f32>(10000.0);
    var second_closest = vec4<f32>(10000.0);

    for (var i: u32 = 0u; i < centroids.count; i = i + 1u) {
        let temp = centroids.data[i];
        let temp_distance = distance(color.rgb, temp.rgb);
        if (temp_distance < distance(color.rgb, closest.rgb)){
            second_closest = closest;
            closest = temp;
        } else if (temp_distance < distance(color.rgb, second_closest.rgb)){
            second_closest = temp;
        }
    }
    values[0] = closest;
    values[1] = second_closest;

    return values;
}

fn dither(color: vec4<f32>, coords: vec2<i32>) -> vec4<f32> {
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

fn meld(color: vec4<f32>, coords: vec2<i32>) -> vec4<f32> {
    let closest_colors = two_closest_colors(color);
    let factor = distance(color.rgb, closest_colors[1].rgb) / distance(closest_colors[0].rgb, closest_colors[1].rgb);

    return factor * closest_colors[0] + (1.0 - factor) * closest_colors[1];
}

@compute
@workgroup_size(16, 16)
fn main_dither(
    @builtin(global_invocation_id) global_id : vec3<u32>,
) {
    let dimensions = textureDimensions(output_texture);
    let coords = vec2<i32>(global_id.xy);

    if(coords.x >= dimensions.x || coords.y >= dimensions.y) {
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
    let coords = vec2<i32>(global_id.xy);

    if(coords.x >= dimensions.x || coords.y >= dimensions.y) {
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