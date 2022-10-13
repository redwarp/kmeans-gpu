struct Centroids {
    count: u32,
    // Aligned 16. See https://www.w3.org/TR/WGSL/#address-space-layout-constraints
    data: array<vec4<f32>>,
};

struct KIndex {
    k: u32,
};

@group(0) @binding(0) var<storage, read> centroids: Centroids;
@group(0) @binding(1) var pixels: texture_2d<f32>;
@group(0) @binding(2) var distance_map: texture_storage_2d<r32float, write>;
@group(1) @binding(0) var<uniform> k_index: KIndex;

@compute
@workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id : vec3<u32>,
) {
    let dimensions = textureDimensions(pixels);
    let coords = vec2<i32>(global_id.xy);

    if(coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let color = textureLoad(pixels, coords, 0).rgb;
    var min_distance: f32 = 1000000.0;
    for (var k: u32 = 0u; k < k_index.k; k = k + 1u) {
        let distance_to_centroid = distance(color, centroids.data[k].rgb);
        min_distance = min(min_distance, distance_to_centroid);
    }

    textureStore(distance_map, coords, vec4<f32>(min_distance, 0.0, 0.0, 0.0));
}