// Let's do some order dithering.
// https://en.wikipedia.org/wiki/Ordered_dithering

struct Centroids {
    count: u32;
    data: array<vec4<f32>>;
};

// struct Indices {
//     data: array<u32, 16>;
// };

[[group(0), binding(0)]] var input_texture: texture_2d<f32>;
[[group(0), binding(1)]] var output_texture : texture_storage_2d<rgba16float, write>;
[[group(0), binding(2)]] var color_indices: texture_2d<u32>;
[[group(0), binding(3)]] var<storage, read> centroids: Centroids;
// [[group(0), binding(4)]] var<uniform> index_matrix: Indices;

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

[[stage(compute), workgroup_size(16, 16)]]
fn main(
    [[builtin(global_invocation_id)]] global_id : vec3<u32>,
) {
    let dimensions = textureDimensions(output_texture);
    let coords = vec2<i32>(global_id.xy);

    if(coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let color = textureLoad(input_texture, coords, 0);
    let closest_colors = two_closest_colors(color);
    let index_value = index_value(coords);
    let factor = distance(color, closest_colors[0]) / distance(closest_colors[0], closest_colors[1]);

    let final_color: vec4<f32> = select(closest_colors[1], closest_colors[0], factor < index_value);

    textureStore(output_texture, coords, final_color);
}