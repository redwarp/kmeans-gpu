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
let max_f32: f32 = 10000.0;
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

[[stage(compute), workgroup_size(256)]]
fn main(
    [[builtin(local_invocation_id)]] local_id : vec3<u32>,
    [[builtin(workgroup_id)]] workgroup_id : vec3<u32>,
    [[builtin(global_invocation_id)]] global_id : vec3<u32>,
) {    
    if (local_id.x == workgroup_size - 1u) {
        atomicStore(&flag_buffer.data[workgroup_id.x], FLAG_NOT_READY);
    }
    storageBarrier();

    let dimensions = textureDimensions(pixels);
    let width = u32(dimensions.x);
    let global_x = global_id.x;
   
    var blank_candidate: Candidate;
    scratch[local_id.x] = blank_candidate;

    var local: Candidate;
    local.index = 0u;
    local.distance = 0.0;

    for (var i: u32 = 0u; i < N_SEQ; i = i + 1u) {
        if (in_bounds(global_x * N_SEQ + i, dimensions)) {
            let color = vec4<f32>(textureLoad(pixels, coords(global_x * N_SEQ + i, dimensions), 0).rgb, 1.0);
            var min_index: u32 = 0u;
            var min_diff: f32 = max_f32;
            for(var k: u32 = 0u; k < k_index.k; k = k + 1u) {
                let k_diff = distance(color.rgb, centroids.data[k].rgb);
                if (k_diff < min_diff) {
                    min_diff = k_diff;
                    min_index = global_x * N_SEQ + i;
                }
            }

            if(local.distance < min_diff) {
                local.distance = min_diff;
                local.index = min_index;
            }
        }
    }

    scratch[local_id.x] = local;

    workgroupBarrier();
    
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        if (local_id.x >= (1u << i)) {
            if (local.distance < scratch[local_id.x - (1u << i)].distance) {
                local.distance = scratch[local_id.x - (1u << i)].distance;
                local.index = scratch[local_id.x - (1u << i)].index;
            }
        }
        workgroupBarrier();
        scratch[local_id.x].distance = local.distance;
        scratch[local_id.x].index = local.index;
        workgroupBarrier();
    }
    
    var exclusive_prefix: Candidate;
    exclusive_prefix.index = 0u;
    exclusive_prefix.distance = 0.0;
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
                    if (their_prefix.distance > exclusive_prefix.distance) {
                        exclusive_prefix.distance = their_prefix.distance;
                        exclusive_prefix.index = their_prefix.index;
                    }
                }
                break;
            } else if (flag == FLAG_AGGREGATE_READY) {                
                if (local_id.x == workgroup_size - 1u) {                    
                    let their_aggregate = atomicLoadCandidate(loop_back_ix * 4u + 2u);                    
                    if (their_aggregate.distance > exclusive_prefix.distance) {
                        exclusive_prefix.distance = their_aggregate.distance;
                        exclusive_prefix.index = their_aggregate.index;
                    }
                }
                loop_back_ix = loop_back_ix - 1u;
            }
            // else spin
        }

        // compute inclusive prefix
        storageBarrier();
        if (local_id.x == workgroup_size - 1u) {
            var inclusive_prefix: Candidate;
            if (local.distance > exclusive_prefix.distance) {
                inclusive_prefix.distance = local.distance;
                inclusive_prefix.index = local.index;
            } else {                
                inclusive_prefix.distance = exclusive_prefix.distance;
                inclusive_prefix.index = exclusive_prefix.index;
            }

            shared_prefix = exclusive_prefix;
            
            atomicStoreCandidate(workgroup_id.x * 2u + 0u, inclusive_prefix);
            atomicStore(&flag_buffer.data[workgroup_id.x], FLAG_PREFIX_READY);
        }
        workgroupBarrier();
        storageBarrier();
    }

    if (workgroup_id.x == last_group_idx() & local_id.x == workgroup_size - 1u) {
        var centroid: Candidate;
        if (local.distance > shared_prefix.distance) {
            centroid.distance = local.distance;
            centroid.index = local.index;
        } else {                
            centroid.distance = shared_prefix.distance;
            centroid.index = shared_prefix.index;
        }

        let centroid_coords = coords(centroid.index, dimensions);
        let new_centroid = textureLoad(pixels, centroid_coords, 0);

        centroids.data[k_index.k] = new_centroid;
    }
    storageBarrier();
}