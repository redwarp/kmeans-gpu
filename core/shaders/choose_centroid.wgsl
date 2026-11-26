struct Centroids {
    count: u32,
    // Aligned 16. See https://www.w3.org/TR/WGSL/#address-space-layout-constraints
    data: array<vec4<f32>>,
};

struct Settings {
    n_seq: u32,
    convergence: f32,
};

struct ColorAggregator {
    color: vec3<f32>,
    count: u32,
};

@group(0) @binding(0) var<storage, read_write> centroids: Centroids;
@group(0) @binding(1) var color_indices: texture_2d<u32>;
@group(0) @binding(2) var pixels: texture_2d<f32>;
// One aggregate per workgroup: the sum and count of the pixels it covers that belong to centroid k.
@group(1) @binding(0) var<storage, read_write> aggregates: array<ColorAggregator>;
@group(1) @binding(1) var<storage, read_write> convergence: array<u32>;
@group(1) @binding(2) var<uniform> settings: Settings;
@group(2) @binding(0) var<uniform> k_index: u32;

const workgroup_size: u32 = 256u;

var<workgroup> scratch: array<ColorAggregator, workgroup_size>;

fn coords(global_x: u32, dimensions: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(global_x % dimensions.x, global_x / dimensions.x);
}

fn in_bounds(global_x: u32, dimensions: vec2<u32>) -> bool {
    return global_x < dimensions.x * dimensions.y;
}

fn matches_centroid(k: u32, coords: vec2<u32>) -> bool {
    return k == textureLoad(color_indices, coords, 0).r;
}

// #include functions/delta_e.wgsl

// Sums the pixels assigned to centroid k, one partial sum per workgroup.
@compute
@workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id : vec3<u32>,
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
) {
    let k = k_index;
    let N_SEQ = settings.n_seq;

    let dimensions = textureDimensions(pixels);
    let global_x = workgroup_id.x * workgroup_size + local_id.x;

    var local: ColorAggregator = ColorAggregator(vec3<f32>(0.0), 0u);
    for (var i: u32 = 0u; i < N_SEQ; i = i + 1u) {
        let index = global_x * N_SEQ + i;
        let coords = coords(index, dimensions);
        if (in_bounds(index, dimensions) && matches_centroid(k, coords)) {
            local.color = local.color + textureLoad(pixels, coords, 0).rgb;
            local.count = local.count + 1u;
        }
    }

    scratch[local_id.x] = local;

    // Prefix sum inside the workgroup: the last invocation ends up with the workgroup total.
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        workgroupBarrier();
        if (local_id.x >= (1u << i)) {
            local.color = local.color + scratch[local_id.x - (1u << i)].color;
            local.count = local.count + scratch[local_id.x - (1u << i)].count;
        }
        workgroupBarrier();
        scratch[local_id.x] = local;
    }

    if (local_id.x == workgroup_size - 1u) {
        aggregates[workgroup_id.x] = local;
    }
}

// Combines the partial sums of every workgroup, and moves centroid k to the resulting mean.
// Dispatched as a single invocation, after `main` completed.
@compute
@workgroup_size(1)
fn pick() {
    var sum = ColorAggregator(vec3<f32>(0.0), 0u);
    let aggregate_count = arrayLength(&aggregates);
    for (var i = 0u; i < aggregate_count; i = i + 1u) {
        let aggregate = aggregates[i];
        sum.color = sum.color + aggregate.color;
        sum.count = sum.count + aggregate.count;
    }

    let k = k_index;
    if(sum.count > 0u) {
        let new_centroid = vec4<f32>(sum.color / f32(sum.count), 1.0);
        let previous_centroid = centroids.data[k];

        centroids.data[k] = new_centroid;

        convergence[k] = u32(distance_cie94(new_centroid.rgb, previous_centroid.rgb) < settings.convergence);
    } else {
        convergence[k] = 0u;
    }

    if (k == centroids.count - 1u) {
        var converge = 0u;
        for (var i = 0u; i < centroids.count; i = i + 1u) {
            converge = converge + convergence[i];
            convergence[i] = 0u;
        }
        convergence[centroids.count] = converge;
    }
}
