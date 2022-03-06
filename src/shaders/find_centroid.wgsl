struct Centroids {
    count: u32;
    data: array<f32>;
};

struct Indices {
    data: array<u32>;
};

[[group(0), binding(0)]] var pixels: texture_2d<f32>;
[[group(0), binding(1)]] var<storage, read> centroids: Centroids;
[[group(0), binding(2)]] var<storage, write> calculated: Indices;

let max_int : u32 = 4294967295u;
let max_f32: f32 = 1000.0;

fn distance_not_sqrt(one: vec4<f32>, other: vec4<f32>) -> f32 {
    var length: vec4<f32> = one - other;

    return length.r * length.r + length.g * length.g + length.b * length.b;
}

[[stage(compute), workgroup_size(16, 16)]]
fn main(
    [[builtin(global_invocation_id)]] global_id : vec3<u32>,
) {
    let dimensions = textureDimensions(pixels);
    let coords = vec2<i32>(global_id.xy);

    if(coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let pixel : vec4<f32> = textureLoad(pixels, coords.xy, 0);

    var min_distance: f32 = max_f32;
    var found_index: u32 = 0u;

    for(var index: u32 = 0u; index < centroids.count; index = index + 1u){
        let centroid_components : vec4<f32> = vec4<f32>(
            centroids.data[index * 4u],
            centroids.data[index * 4u + 1u],
            centroids.data[index * 4u + 2u],
            centroids.data[index * 4u + 3u],
        );

        let distance: f32 = distance_not_sqrt(pixel, centroid_components);
        let smaller = bool(distance < min_distance);
        min_distance = distance * f32(smaller) + min_distance * f32(!smaller);
        found_index = index * u32(smaller) + found_index * u32(!smaller);
    }

    let index: u32 = global_id.y * u32(dimensions.x) + global_id.x;
    calculated.data[index] = found_index;
}
