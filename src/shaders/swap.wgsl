struct Centroids {
    count: u32;
    data: array<u32>;
};

struct Indices {
    data: array<u32>;
};

[[group(0), binding(0)]] var<storage, read> centroids: Centroids;
[[group(0), binding(1)]] var<storage, read> calculated: Indices;
[[group(0), binding(2)]] var output_texture : texture_storage_2d<rgba8uint, write>;

[[stage(compute), workgroup_size(16, 16)]]
fn main(
    [[builtin(global_invocation_id)]] global_id : vec3<u32>,
) {
    let dimensions = textureDimensions(output_texture);
    let coords = vec2<i32>(global_id.xy);

    if(coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let index = calculated.data[global_id.y * u32(dimensions.x) + global_id.x];
    let color = vec4<u32>(
        centroids.data[index * 4u + 0u],
        centroids.data[index * 4u + 1u],
        centroids.data[index * 4u + 2u],
        centroids.data[index * 4u + 3u],
    );
    
    textureStore(output_texture, coords.xy, color);
}
