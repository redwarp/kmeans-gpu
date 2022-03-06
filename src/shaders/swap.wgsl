struct Pixels {
    count: u32;
    data: array<u32>;
};

[[group(0), binding(0)]] var<storage, read> centroids: Pixels;
[[group(0), binding(1)]] var<storage, read_write> calculated: Pixels;

[[stage(compute), workgroup_size(256)]]
fn main(
  [[builtin(global_invocation_id)]] global_id : vec3<u32>,
) {
    if (global_id.x >= calculated.count) {
      return;
    }

    let centroid_id = calculated.data[global_id.x];
    calculated.data[global_id.x] = centroids.data[centroid_id];
}
