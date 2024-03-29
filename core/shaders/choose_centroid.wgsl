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
@group(0) @binding(3) var<storage, read_write> part_id_buffer : array<atomic<u32>>;
@group(1) @binding(0) var<storage, read_write> prefix_buffer: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> flag_buffer: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> convergence: array<atomic<u32>>;
@group(1) @binding(3) var<uniform> settings: Settings;
@group(2) @binding(0) var<uniform> k_index: u32;

const workgroup_size: u32 = 256u;

var<workgroup> scratch: array<ColorAggregator, workgroup_size>;
var<workgroup> shared_flag: u32;
var<workgroup> part_id: u32;

const FLAG_NOT_READY = 0u;
const FLAG_AGGREGATE_READY = 1u;
const FLAG_PREFIX_READY = 2u;

fn coords(global_x: u32, dimensions: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(global_x % dimensions.x, global_x / dimensions.x);
}

fn last_group_idx() -> u32 {
    return arrayLength(&flag_buffer) - 1u;
}

fn in_bounds(global_x: u32, dimensions: vec2<u32>) -> bool {
    return global_x < dimensions.x * dimensions.y;
}

fn matches_centroid(k: u32, coords: vec2<u32>) -> bool {
    return k == textureLoad(color_indices, coords, 0).r;
}

fn atomicStoreAggregator(index: u32, value: ColorAggregator) {
    atomicStore(&prefix_buffer[index + 0u], bitcast<u32>(value.color.r));
    atomicStore(&prefix_buffer[index + 1u], bitcast<u32>(value.color.g));
    atomicStore(&prefix_buffer[index + 2u], bitcast<u32>(value.color.b));
    atomicStore(&prefix_buffer[index + 3u], value.count);
}

fn atomicLoadAggregator(index: u32) -> ColorAggregator {
    var value: ColorAggregator;
    let r = bitcast<f32>(atomicLoad(&prefix_buffer[index + 0u]));
    let g = bitcast<f32>(atomicLoad(&prefix_buffer[index + 1u]));
    let b = bitcast<f32>(atomicLoad(&prefix_buffer[index + 2u]));
    let count = atomicLoad(&prefix_buffer[index + 3u]);
    value.color = vec3<f32>(r, g, b);
    value.count = count;
    return value;
}

// #include functions/delta_e.wgsl

@compute
@workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id : vec3<u32>,
) {
    if (local_id.x == 0u) {
        part_id = atomicAdd(&part_id_buffer[0], 1u);
    }
    workgroupBarrier();
    let workgroup_x = part_id;
    
    if (local_id.x == 0u) {
        atomicStore(&flag_buffer[workgroup_x], FLAG_NOT_READY);
    }
    storageBarrier();

    let k = k_index;
    let N_SEQ = settings.n_seq;

    let dimensions = textureDimensions(pixels);
    let width = u32(dimensions.x);
    let global_x = workgroup_x * workgroup_size + local_id.x;

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
    
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        workgroupBarrier();
        if (local_id.x >= (1u << i)) {
            local.color = local.color + scratch[local_id.x - (1u << i)].color;
            local.count = local.count + scratch[local_id.x - (1u << i)].count;
        }
        workgroupBarrier();
        scratch[local_id.x] = local;
    }
    
    var exclusive_prefix = ColorAggregator(vec3<f32>(0.0), 0u);
    var flag = FLAG_AGGREGATE_READY;
    
    if (local_id.x == workgroup_size - 1u) {
        atomicStoreAggregator(workgroup_x * 8u + 4u, local);
        if (workgroup_x == 0u) {
            // Special case, group 0 will not need to sum prefix.
            atomicStoreAggregator(workgroup_x * 8u + 0u, local);
            flag = FLAG_PREFIX_READY;
        }
    }
    storageBarrier();

    if (local_id.x == workgroup_size - 1u) {
        atomicStore(&flag_buffer[workgroup_x], flag);
    }    
    storageBarrier();

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
                    let their_prefix = atomicLoadAggregator(loop_back_ix * 8u);
                    exclusive_prefix.color = exclusive_prefix.color + their_prefix.color;
                    exclusive_prefix.count = exclusive_prefix.count + their_prefix.count;
                }
                break;
            } else if (flag == FLAG_AGGREGATE_READY) {                
                if (local_id.x == workgroup_size - 1u) {                    
                    let their_aggregate = atomicLoadAggregator(loop_back_ix * 8u + 4u);
                    exclusive_prefix.color = their_aggregate.color + exclusive_prefix.color;
                    exclusive_prefix.count = their_aggregate.count + exclusive_prefix.count;
                }
                loop_back_ix = loop_back_ix - 1u;
            }
            // else spin
        }

        // compute inclusive prefix
        if (local_id.x == workgroup_size - 1u) {
            var inclusive_prefix: ColorAggregator;
            inclusive_prefix.color = exclusive_prefix.color + local.color;
            inclusive_prefix.count = exclusive_prefix.count + local.count;
            
            atomicStoreAggregator(workgroup_x * 8u + 0u, inclusive_prefix);
        }
        storageBarrier();
        if (local_id.x == workgroup_size - 1u) {
            atomicStore(&flag_buffer[workgroup_x], FLAG_PREFIX_READY);
        }
    }
}

@compute
@workgroup_size(1)
fn pick() {
    let sum = atomicLoadAggregator(last_group_idx() * 8u + 0u);
    let k = k_index;
    if(sum.count > 0u) {
        let new_centroid = vec4<f32>(sum.color / f32(sum.count), 1.0);
        let previous_centroid = centroids.data[k];

        centroids.data[k] = new_centroid;

        atomicStore(&convergence[k], u32(distance_cie94(new_centroid.rgb, previous_centroid.rgb) < settings.convergence));
    } else {
        atomicStore(&convergence[k], 0u);
    }

    if (k == centroids.count - 1u) {
        var converge = atomicExchange(&convergence[0u], 0u);
        for (var i = 1u; i < centroids.count; i = i + 1u) {
            converge = converge + atomicExchange(&convergence[i], 0u);
        }
        atomicStore(&convergence[centroids.count], converge);
    }
    
    // Reset part ids for next centroid.
    atomicStore(&part_id_buffer[0], 0u);
}