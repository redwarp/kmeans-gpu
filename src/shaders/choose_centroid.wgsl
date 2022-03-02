struct Pixels {
    count: u32;
    data: array<u32>;
};

[[group(0), binding(0)]] var<storage, read_write> centroids: Pixels;
[[group(0), binding(1)]] var<storage, read> pixels: Pixels;
[[group(0), binding(2)]] var<storage, read> calculated: Pixels;

let max_int : u32 = 4294967295u;

[[stage(compute), workgroup_size(16)]]
fn main(
  [[builtin(global_invocation_id)]] global_id : vec3<u32>,
) {
    if(global_id.x >= centroids.count) {
        return;
    }
    
    var centroid = centroids.data[global_id.x];
    var new_centroid = vec3<u32>(0u, 0u, 0u);
    var count: u32 = 0u;

    for(var index: u32 = 0u; index < pixels.count; index = index + 1u){
        let same: bool = centroid == calculated.data[index];
        let color: u32 = pixels.data[index];
        let pixel : vec3<u32> = vec3<u32>(
            color >> 16u & 0xffu,
            color >> 8u & 0xffu,
            color & 0xffu
        );
        new_centroid = new_centroid + pixel * u32(same);
        count = count + u32(same);
    }

    new_centroid = new_centroid / max(count, 1u);
    
    centroid = new_centroid.r << 16u | new_centroid.g << 8u | new_centroid.b;
    centroids.data[global_id.x] = centroid;
}