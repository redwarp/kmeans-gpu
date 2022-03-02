struct Pixels {
    count: u32;
    data: array<u32>;
};

[[group(0), binding(0)]] var<storage, read> centroids: Pixels;
[[group(0), binding(1)]] var<storage, read> pixels: Pixels;
[[group(0), binding(2)]] var<storage, write> calculated: Pixels;

let max_int : u32 = 4294967295u;

fn distance_not_sqrt(one: vec3<u32>, other: vec3<u32>) -> u32 {
  var length: vec3<u32> = one - other;

  return length.r * length.r + length.g * length.g + length.b * length.b;
}

[[stage(compute), workgroup_size(256)]]
fn main(
  [[builtin(global_invocation_id)]] global_id : vec3<u32>,
) {
    if (global_id.x >= pixels.count) {
      return;
    }

    let pixel : vec3<u32> = vec3<u32>(
      pixels.data[global_id.x] >> 16u & 0xffu,
      pixels.data[global_id.x] >> 8u & 0xffu,
      pixels.data[global_id.x] & 0xffu
    );

    var min_distance: u32 = max_int;
    var found: u32 = 0u;

    for(var index: u32 = 0u; index < centroids.count; index = index + 1u){
      let centroid: u32 = centroids.data[index];
      let centroid_components : vec3<u32> = vec3<u32>(
        centroid >> 16u & 0xffu,
        centroid >> 8u & 0xffu,
        centroid & 0xffu
      );

      let distance: u32 = distance_not_sqrt(pixel, centroid_components);
      let smaller = bool(distance < min_distance);
      min_distance = distance * u32(smaller) + min_distance * u32(!smaller);
      found = centroid * u32(smaller) + found * u32(!smaller);
    }

    calculated.data[global_id.x] = found;
}
