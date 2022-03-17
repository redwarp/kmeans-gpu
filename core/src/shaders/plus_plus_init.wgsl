struct Centroids {
    count: u32;
    // Aligned 16. See https://www.w3.org/TR/WGSL/#address-space-layout-constraints
    data: array<vec4<f32>>;
};

struct KIndex {
    k: u32;
};

struct AtomicBuffer {
    data: array<atomic<u32>>;
};

struct Candidate {
    index: u32;
    distance: f32;
};

let FLAG_NOT_READY = 0u;
let FLAG_AGGREGATE_READY = 1u;
let FLAG_PREFIX_READY = 2u;
let N_SEQ = 24u;
let workgroup_size: u32 = 256u;
let max_f32: f32 = 1000000.0;
let max_int : u32 = 4294967295u;

[[group(0), binding(0)]] var<storage, read_write> centroids: Centroids;
[[group(0), binding(1)]] var pixels: texture_2d<f32>;
[[group(0), binding(2)]] var<storage, read_write> prefix_buffer: AtomicBuffer;
[[group(0), binding(3)]] var<storage, read_write> flag_buffer: AtomicBuffer;
[[group(1), binding(0)]] var<uniform> k_index: KIndex;

var<workgroup> scratch: array<Candidate, workgroup_size>;
var<workgroup> shared_prefix: Candidate;
var<workgroup> shared_flag: u32;

fn coords(global_x: u32, dimensions: vec2<i32>) -> vec2<i32> {
    return vec2<i32>(vec2<u32>(global_x % u32(dimensions.x), global_x / u32(dimensions.x)));
}

fn last_group_idx() -> u32 {
    return arrayLength(&flag_buffer.data) - 1u;
}

fn in_bounds(global_x: u32, dimensions: vec2<i32>) -> bool {
    let x = global_x % u32(dimensions.x);
    let y = global_x / u32(dimensions.x);
    return x < u32(dimensions.x) && y < u32(dimensions.y);
}

fn atomicStoreCandidate(index: u32, value: Candidate) {
    atomicStore(&prefix_buffer.data[index + 0u], value.index);
    atomicStore(&prefix_buffer.data[index + 1u], bitcast<u32>(value.distance));
}

fn atomicLoadCandidate(index: u32) -> Candidate {
    var output: Candidate;
    output.index = atomicLoad(&prefix_buffer.data[index + 0u]);
    output.distance = bitcast<f32>(atomicLoad(&prefix_buffer.data[index + 1u]));
    return output;
}

fn my_dist(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return pow(a.r - b.r, 2.0) + pow(a.g - b.g, 2.0) + pow(a.b - b.b, 2.0);
}

fn rand(seed: f32) -> f32 {
    return fract(sin(dot(vec2<f32>(seed), vec2<f32>(12.9898,78.233))) * 43758.5453);
}

[[stage(compute), workgroup_size(256)]]
fn main(
    [[builtin(local_invocation_id)]] local_id : vec3<u32>,
    [[builtin(workgroup_id)]] workgroup_id : vec3<u32>,
    [[builtin(global_invocation_id)]] global_id : vec3<u32>,
) {    
    if (k_index.k == 0u) {
        if (local_id.x == 0u) {
            let dimensions = textureDimensions(pixels);
            let x = i32(f32(dimensions.x) * rand(42.0));
            let y = i32(f32(dimensions.y) * rand(12.0));

            let new_centroid = textureLoad(pixels, vec2<i32>(x, y), 0);

            centroids.data[k_index.k] = new_centroid;
        }

        return;
    }

    if (local_id.x == workgroup_size - 1u) {
        atomicStore(&flag_buffer.data[workgroup_id.x], FLAG_NOT_READY);
    }
    storageBarrier();

    let dimensions = textureDimensions(pixels);
    let width = u32(dimensions.x);
    let global_x = global_id.x;
   
    scratch[local_id.x] = Candidate(0u, 0.0);

    var local = Candidate(0u, 0.0);

    for (var i: u32 = 0u; i < N_SEQ; i = i + 1u) {
        let pixel_index = global_x * N_SEQ + i;

        let in_bounds = in_bounds(pixel_index, dimensions);
        let color = textureLoad(pixels, coords(pixel_index, dimensions), 0).rgb;
        var min_index: u32 = 0u;
        var min_diff: f32 = max_f32;
        for(var k: u32 = 0u; k < k_index.k; k = k + 1u) {
            let k_diff = my_dist(color, centroids.data[k].rgb);
            let smaller = in_bounds && (k_diff < min_diff);
            min_diff = f32(smaller) * k_diff + f32(!smaller) * min_diff;
            min_index = u32(smaller) * pixel_index + u32(!smaller) * min_index;
        }

        let smaller = in_bounds && local.distance < min_diff;
        local.distance = f32(smaller) * min_diff + f32(!smaller) * local.distance;
        local.index = u32(smaller) * min_index + u32(!smaller) * local.index;
    }

    scratch[local_id.x] = local;

    workgroupBarrier();
    
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        if (local_id.x >= (1u << i)) {
            let value = scratch[local_id.x - (1u << i)];
            let smaller = local.distance < value.distance;
            local.distance = select(local.distance, value.distance, smaller);
            local.index = select(local.index, value.index, smaller);
        }
        workgroupBarrier();
        scratch[local_id.x] = local;
        workgroupBarrier();
    }
    
    var flag = FLAG_AGGREGATE_READY;
    
    if (local_id.x == workgroup_size - 1u) {
        atomicStoreCandidate(workgroup_id.x * 4u + 2u, local);
        if (workgroup_id.x == 0u) {
            // Special case for group 0.
            atomicStoreCandidate(workgroup_id.x * 4u + 0u, local);
            flag = FLAG_PREFIX_READY;
        }
    }
    storageBarrier();

    if (local_id.x == workgroup_size - 1u) {
        atomicStore(&flag_buffer.data[workgroup_id.x], flag);
    }
    storageBarrier();

    if(workgroup_id.x != 0u) {
        // decoupled loop-back
        var loop_back_ix = workgroup_id.x - 1u;
        loop {
            if(local_id.x == workgroup_size - 1u) {
                shared_flag = atomicLoad(&flag_buffer.data[loop_back_ix]);
            }
            workgroupBarrier();
            flag = shared_flag;
            storageBarrier();

            if (flag == FLAG_PREFIX_READY) {
                if (local_id.x == workgroup_size - 1u) {
                    let their_prefix = atomicLoadCandidate(loop_back_ix * 4u + 0u);
                    let smaller = their_prefix.distance > local.distance;
                    local.distance = select(local.distance, their_prefix.distance, smaller);
                    local.index = select(local.index, their_prefix.index, smaller);
                }
                break;
            } else if (flag == FLAG_AGGREGATE_READY) {                
                if (local_id.x == workgroup_size - 1u) {                    
                    let their_aggregate = atomicLoadCandidate(loop_back_ix * 4u + 2u);       
                    let smaller = their_aggregate.distance > local.distance;                    
                    local.distance = select(local.distance, their_aggregate.distance, smaller);
                    local.index = select(local.index, their_aggregate.index, smaller);
                }
                loop_back_ix = loop_back_ix - 1u;
            }
            // else spin
        }

        storageBarrier();
        if (local_id.x == workgroup_size - 1u) {
            shared_prefix = local;
            
            atomicStoreCandidate(workgroup_id.x * 2u + 0u, local);
            atomicStore(&flag_buffer.data[workgroup_id.x], FLAG_PREFIX_READY);
        }
        workgroupBarrier();
        storageBarrier();
    }

    if (workgroup_id.x == last_group_idx() & local_id.x == workgroup_size - 1u) {
        var centroid: Candidate = shared_prefix;

        let centroid_coords = coords(centroid.index, dimensions);
        let new_centroid = textureLoad(pixels, centroid_coords, 0);

        centroids.data[k_index.k] = new_centroid;
    }
    storageBarrier();
}