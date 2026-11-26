struct Centroids {
    count: u32,
    // Aligned 16. See https://www.w3.org/TR/WGSL/#address-space-layout-constraints
    data: array<vec4<f32>>,
};

struct Candidate {
    index: u32,
    distance: f32,
};

const N_SEQ = 16u;
const workgroup_size: u32 = 256u;

@group(0) @binding(0) var<storage, read_write> centroids: Centroids;
@group(0) @binding(1) var pixels: texture_2d<f32>;
// One candidate per workgroup: the pixel it covers that is the furthest from the existing centroids.
@group(0) @binding(2) var<storage, read_write> candidates: array<Candidate>;
@group(0) @binding(3) var distance_map: texture_2d<f32>;
@group(1) @binding(0) var<uniform> k_index: u32;

var<workgroup> scratch: array<Candidate, workgroup_size>;

fn coords(pixel_index: u32, dimensions: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(pixel_index % dimensions.x, pixel_index / dimensions.x);
}

fn in_bounds(pixel_index: u32, dimensions: vec2<u32>) -> bool {
    return pixel_index < dimensions.x * dimensions.y;
}

fn rand(seed: f32) -> f32 {
    return fract(sin(dot(vec2<f32>(seed), vec2<f32>(12.9898,78.233))) * 43758.5453);
}

fn selectCandidate(a: Candidate, b: Candidate) -> Candidate {
    if (a.distance < b.distance) {
        return b;
    } else {
        return a;
    }
}

// Finds the pixel that is the furthest from the existing centroids, one candidate per workgroup.
@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id : vec3<u32>,
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
) {
    let dimensions = textureDimensions(pixels);
    let global_x = workgroup_id.x * workgroup_size + local_id.x;

    var local = Candidate(0u, 0.0);

    for (var i: u32 = 0u; i < N_SEQ; i = i + 1u) {
        let pixel_index = global_x * N_SEQ + i;

        if (in_bounds(pixel_index, dimensions)) {
            let min_distance = textureLoad(distance_map, coords(pixel_index, dimensions), 0).r;
            local = selectCandidate(local, Candidate(pixel_index, min_distance));
        }
    }

    scratch[local_id.x] = local;

    // Prefix scan inside the workgroup: the last invocation ends up with the workgroup best.
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        workgroupBarrier();
        if (local_id.x >= (1u << i)) {
            let value = scratch[local_id.x - (1u << i)];
            local = selectCandidate(local, value);
        }
        workgroupBarrier();
        scratch[local_id.x] = local;
    }

    if (local_id.x == workgroup_size - 1u) {
        candidates[workgroup_id.x] = local;
    }
}

@compute
@workgroup_size(1)
fn initial() {
    let dimensions = textureDimensions(pixels);
    let x = i32(f32(dimensions.x) * rand(42.0));
    let y = i32(f32(dimensions.y) * rand(12.0));

    let new_centroid = textureLoad(pixels, vec2<i32>(x, y), 0);
    centroids.data[0] = new_centroid;
}

// Compares the candidates of every workgroup, and uses the furthest pixel as centroid k.
// Dispatched as a single invocation, after `main` completed.
@compute
@workgroup_size(1)
fn pick() {
    let dimensions = textureDimensions(pixels);

    var best = Candidate(0u, 0.0);
    let candidate_count = arrayLength(&candidates);
    for (var i = 0u; i < candidate_count; i = i + 1u) {
        best = selectCandidate(best, candidates[i]);
    }

    let centroid_coords = coords(best.index, dimensions);
    let new_centroid = vec4<f32>(textureLoad(pixels, centroid_coords, 0).rgb, 1.0);

    centroids.data[k_index] = new_centroid;
}
