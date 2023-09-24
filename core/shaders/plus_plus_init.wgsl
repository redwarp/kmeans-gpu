struct Centroids {
    count: u32,
    // Aligned 16. See https://www.w3.org/TR/WGSL/#address-space-layout-constraints
    data: array<vec4<f32>>,
};

struct Candidate {
    index: u32,
    distance: f32,
};

const FLAG_NOT_READY = 0u;
const FLAG_AGGREGATE_READY = 1u;
const FLAG_PREFIX_READY = 2u;
const N_SEQ = 16u;
const workgroup_size: u32 = 256u;
const max_f32: f32 = 4294967295.0;
const max_int : u32 = 4294967295u;

@group(0) @binding(0) var<storage, read_write> centroids: Centroids;
@group(0) @binding(1) var pixels: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> prefix_buffer: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> flag_buffer: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> part_id_buffer : array<atomic<u32>>;
@group(0) @binding(5) var distance_map: texture_2d<f32>;
@group(1) @binding(0) var<uniform> k_index: u32;

var<workgroup> scratch: array<Candidate, workgroup_size>;
var<workgroup> shared_flag: u32;
var<workgroup> part_id: u32;

fn coords(pixel_index: u32, dimensions: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(pixel_index % dimensions.x, pixel_index / dimensions.x);
}

fn last_group_idx() -> u32 {
    return arrayLength(&flag_buffer) - 1u;
}

fn in_bounds(pixel_index: u32, dimensions: vec2<u32>) -> bool {
    return pixel_index < dimensions.x * dimensions.y;
}

fn atomicStoreCandidate(index: u32, value: Candidate) {
    atomicStore(&prefix_buffer[index], value.index);
}

fn atomicLoadCandidate(index: u32, dimensions: vec2<u32>) -> Candidate {
    var output: Candidate;
    output.index = atomicLoad(&prefix_buffer[index]);

    let coords = coords(output.index, dimensions);

    output.distance = textureLoad(distance_map, coords, 0).r;
    return output;
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

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id : vec3<u32>,
) {  
    if (local_id.x == 0u) {
        part_id = atomicAdd(&part_id_buffer[0], 1u);
    }
    workgroupBarrier();
    let workgroup_x = part_id;

    let dimensions = textureDimensions(pixels);
    let width = dimensions.x;
    let global_x = workgroup_x * workgroup_size + local_id.x;

    var local = Candidate(0u, 0.0);

    for (var i: u32 = 0u; i < N_SEQ; i = i + 1u) {
        let pixel_index = global_x * N_SEQ + i;

        let in_bounds = in_bounds(pixel_index, dimensions);
        if (in_bounds){
            let min_distance = textureLoad(distance_map, coords(pixel_index, dimensions), 0).r;
            local = selectCandidate(local, Candidate(pixel_index, min_distance));
        }
    }

    scratch[local_id.x] = local;
    
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        workgroupBarrier();
        if (local_id.x >= (1u << i)) {
            let value = scratch[local_id.x - (1u << i)];
            local = selectCandidate(local, value);
        }
        workgroupBarrier();
        scratch[local_id.x] = local;
    }
    
    var flag = FLAG_AGGREGATE_READY;
    if (local_id.x == workgroup_size - 1u) {
        atomicStoreCandidate(workgroup_x * 2u + 1u, local);
        if (workgroup_x == 0u) {
            // Special case for group 0.
            atomicStoreCandidate(workgroup_x * 2u + 0u, local);
            flag = FLAG_PREFIX_READY;
        }
    }
    storageBarrier();
    if (local_id.x == workgroup_size - 1u) {
        atomicStore(&flag_buffer[workgroup_x], flag);
    }

    if(workgroup_x != 0u) {
        // decoupled loop-back
        var loop_back_ix = workgroup_x - 1u;
        loop {
            if(local_id.x == workgroup_size - 1u) {
                shared_flag = atomicLoad(&flag_buffer[loop_back_ix]);
            }
            workgroupBarrier();
            flag = shared_flag;
            storageBarrier();

            if (flag == FLAG_PREFIX_READY) {
                if (local_id.x == workgroup_size - 1u) {
                    let their_prefix = atomicLoadCandidate(loop_back_ix * 2u + 0u, dimensions);
                    local = selectCandidate(local, their_prefix);
                }
                break;
            } else if (flag == FLAG_AGGREGATE_READY) {                
                if (local_id.x == workgroup_size - 1u) {                    
                    let their_aggregate = atomicLoadCandidate(loop_back_ix * 2u + 1u, dimensions);       
                    local = selectCandidate(local, their_aggregate);
                }
                loop_back_ix = loop_back_ix - 1u;
            }
            // else spin
        }

        if (local_id.x == workgroup_size - 1u) {
            atomicStoreCandidate(workgroup_x * 2u + 0u, local);
        }
        storageBarrier();
        if (local_id.x == workgroup_size - 1u) {
            atomicStore(&flag_buffer[workgroup_x], FLAG_PREFIX_READY);
        }
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

@compute
@workgroup_size(1)
fn pick() {
    let dimensions = textureDimensions(pixels);
    let centroid = atomicLoadCandidate(last_group_idx() * 2u + 0u, dimensions);

    let centroid_coords = coords(centroid.index, dimensions);
    let new_centroid = vec4<f32>(textureLoad(pixels, centroid_coords, 0).rgb, 1.0);

    centroids.data[k_index] = new_centroid;

    // Reset part ids for next centroid.
    atomicStore(&part_id_buffer[0], 0u);
    // Reset flags.
    for (var i = 0u; i < last_group_idx(); i = i + 1u) {
        atomicStore(&flag_buffer[i], FLAG_NOT_READY);
    }
}