struct Centroids {
    count: u32;
    data: array<vec4<f32>>;
};

struct Indices {
    data: array<u32>;
};

[[group(0), binding(0)]] var pixels: texture_2d<f32>;
[[group(0), binding(1)]] var<storage, read> centroids: Centroids;
[[group(0), binding(2)]] var color_indices: texture_storage_2d<r32uint, write>;

let max_int : u32 = 4294967295u;
let max_f32: f32 = 1000.0;

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
    var found_index: u32 = centroids.count;

    for(var index: u32 = 0u; index < centroids.count; index = index + 1u){
        let centroid_components : vec4<f32> = centroids.data[index];

        let distance: f32 = distance(pixel.rgb, centroid_components.rgb);

        if (distance < min_distance) {            
            min_distance = distance;
            found_index = index;
        }
    }

    let index: u32 = global_id.y * u32(dimensions.x) + global_id.x;
    textureStore(color_indices, coords, vec4<u32>(found_index));
    storageBarrier();
}
