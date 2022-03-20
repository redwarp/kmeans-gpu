struct Centroids {
    count: u32;
    data: array<vec4<f32>>;
};

struct Indices {
    data: array<u32>;
};

[[group(0), binding(0)]] var<storage, read> centroids: Centroids;
[[group(0), binding(1)]] var color_indices: texture_2d<u32>;
[[group(0), binding(2)]] var output_texture : texture_storage_2d<rgba32float, write>;

[[stage(compute), workgroup_size(16, 16)]]
fn main(
    [[builtin(global_invocation_id)]] global_id : vec3<u32>,
) {
    let dimensions = textureDimensions(output_texture);
    let coords = vec2<i32>(global_id.xy);

    if(coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let index = textureLoad(color_indices, coords, 0).r;
    
    textureStore(output_texture, coords.xy, centroids.data[index]);
}
