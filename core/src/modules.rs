use log::{debug, log_enabled};
use std::num::NonZeroU32;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferAddress, BufferBinding,
    BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePass,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, MapMode,
    PipelineLayoutDescriptor, Queue, ShaderSource, ShaderStages, StorageTextureAccess, Texture,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor,
    TextureViewDimension,
};

use crate::{
    utils::compute_work_group_count, CentroidsBuffer, ColorIndexTexture, ColorSpace, InputTexture,
    MixMode, OutputTexture, WorkTexture,
};

pub(crate) trait Module {
    fn dispatch<'a>(&'a self, compute_pass: &mut ComputePass<'a>);
}

pub(crate) struct ColorConverterModule {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    dispatch_size: (u32, u32),
}

impl ColorConverterModule {
    pub fn new(
        device: &Device,
        color_space: &ColorSpace,
        image_dimensions: (u32, u32),
        input_texture: &InputTexture,
        work_texture: &WorkTexture,
    ) -> Self {
        let convert_color_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Convert color shader"),
            source: ShaderSource::Wgsl(
                match color_space {
                    ColorSpace::Lab => include_str!("shaders/converters/rgb_to_lab.wgsl"),
                    ColorSpace::Rgb => include_str!("shaders/converters/rgb8u_to_rgb32f.wgsl"),
                }
                .into(),
            ),
        });

        let convert_color_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Convert color bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    WorkTexture::texture_storage_layout(1),
                ],
            });

        let convert_color_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Convert color layout"),
                bind_group_layouts: &[&convert_color_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Convert color pipeline"),
            layout: Some(&convert_color_pipeline_layout),
            module: &convert_color_shader,
            entry_point: "main",
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Convert color bind group"),
            layout: &convert_color_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&input_texture.create_view(
                        &TextureViewDescriptor {
                            label: None,
                            format: Some(TextureFormat::Rgba8Unorm),
                            aspect: wgpu::TextureAspect::All,
                            base_mip_level: 0,
                            mip_level_count: NonZeroU32::new(1),
                            dimension: Some(TextureViewDimension::D2),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &work_texture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
            ],
        });

        let dispatch_size = compute_work_group_count(image_dimensions, (16, 16));

        Self {
            pipeline,
            bind_group,
            dispatch_size,
        }
    }
}

impl Module for ColorConverterModule {
    fn dispatch<'a>(&'a self, compute_pass: &mut ComputePass<'a>) {
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.dispatch(self.dispatch_size.0, self.dispatch_size.1, 1);
    }
}

pub(crate) struct ColorReverterModule {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    dispatch_size: (u32, u32),
}

impl ColorReverterModule {
    pub fn new(
        device: &Device,
        color_space: &ColorSpace,
        image_dimensions: (u32, u32),
        work_texture: &WorkTexture,
        output_texture: &OutputTexture,
    ) -> Self {
        let revert_color_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Revert color shader"),
            source: ShaderSource::Wgsl(
                match color_space {
                    ColorSpace::Lab => include_str!("shaders/converters/lab_to_rgb.wgsl"),
                    ColorSpace::Rgb => include_str!("shaders/converters/rgb32f_to_rgb8u.wgsl"),
                }
                .into(),
            ),
        });

        let revert_color_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Revert color bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba8Unorm,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let revert_color_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Revert color layout"),
                bind_group_layouts: &[&revert_color_bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Revert color pipeline"),
            layout: Some(&revert_color_pipeline_layout),
            module: &revert_color_shader,
            entry_point: "main",
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Revert color bind group"),
            layout: &revert_color_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &work_texture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&output_texture.create_view(
                        &TextureViewDescriptor {
                            label: None,
                            format: Some(TextureFormat::Rgba8Unorm),
                            aspect: wgpu::TextureAspect::All,
                            base_mip_level: 0,
                            mip_level_count: NonZeroU32::new(1),
                            dimension: Some(TextureViewDimension::D2),
                            ..Default::default()
                        },
                    )),
                },
            ],
        });

        let dispatch_size = compute_work_group_count(image_dimensions, (16, 16));

        Self {
            pipeline,
            bind_group,
            dispatch_size,
        }
    }
}

impl Module for ColorReverterModule {
    fn dispatch<'a>(&'a self, compute_pass: &mut ComputePass<'a>) {
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.dispatch(self.dispatch_size.0, self.dispatch_size.1, 1);
    }
}

pub(crate) struct SwapModule {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    dispatch_size: (u32, u32),
}

impl SwapModule {
    pub fn new(
        device: &Device,
        image_dimensions: (u32, u32),
        work_texture: &WorkTexture,
        centroid_buffer: &CentroidsBuffer,
        color_index_texture: &ColorIndexTexture,
    ) -> Self {
        let swap_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Swap colors shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/swap.wgsl").into()),
        });

        let swap_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Swap bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Uint,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                WorkTexture::texture_storage_layout(2),
            ],
        });

        let swap_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &swap_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: centroid_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &color_index_texture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &work_texture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
            ],
        });

        let swap_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Swap pipeline layout"),
            bind_group_layouts: &[&swap_bind_group_layout],
            push_constant_ranges: &[],
        });
        let swap_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Swap pipeline"),
            layout: Some(&swap_pipeline_layout),
            module: &swap_shader,
            entry_point: "main",
        });

        let dispatch_size = compute_work_group_count(image_dimensions, (16, 16));

        Self {
            pipeline: swap_pipeline,
            bind_group: swap_bind_group,
            dispatch_size,
        }
    }
}

impl Module for SwapModule {
    fn dispatch<'a>(&'a self, compute_pass: &mut ComputePass<'a>) {
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.dispatch(self.dispatch_size.0, self.dispatch_size.1, 1);
    }
}

pub(crate) struct FindCentroidModule {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    dispatch_size: (u32, u32),
}

impl FindCentroidModule {
    pub fn new(
        device: &Device,
        image_dimensions: (u32, u32),
        work_texture: &WorkTexture,
        centroid_buffer: &CentroidsBuffer,
        color_index_texture: &ColorIndexTexture,
    ) -> Self {
        let find_centroid_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Find centroid shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/find_centroid.wgsl").into()),
        });

        let find_centroid_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Find centroid bind group layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::R32Uint,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let find_centroid_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Pipeline layout"),
                bind_group_layouts: &[&find_centroid_bind_group_layout],
                push_constant_ranges: &[],
            });

        let find_centroid_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Find centroid pipeline"),
            layout: Some(&find_centroid_pipeline_layout),
            module: &find_centroid_shader,
            entry_point: "main",
        });

        let find_centroid_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Find centroid bind group"),
            layout: &find_centroid_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &work_texture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: centroid_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &color_index_texture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
            ],
        });

        let dispatch_size = compute_work_group_count(image_dimensions, (16, 16));

        Self {
            pipeline: find_centroid_pipeline,
            bind_group: find_centroid_bind_group,
            dispatch_size,
        }
    }
}

impl Module for FindCentroidModule {
    fn dispatch<'a>(&'a self, compute_pass: &mut ComputePass<'a>) {
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.dispatch(self.dispatch_size.0, self.dispatch_size.1, 1);
    }
}

pub(crate) struct ConvergenceBuffer {
    gpu_buffer: Buffer,
    mapped_buffer: Buffer,
}

pub(crate) struct ChooseCentroidModule<'a> {
    k: u32,
    pipeline: ComputePipeline,
    pick_pipeline: ComputePipeline,
    bind_group_0: BindGroup,
    bind_group_1: BindGroup,
    bind_groups: Vec<BindGroup>,
    dispatch_size: u32,
    convergence_buffer: ConvergenceBuffer,
    find_centroid_module: &'a FindCentroidModule,
    centroid_buffer: &'a CentroidsBuffer,
}

impl<'a> ChooseCentroidModule<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        color_space: &ColorSpace,
        image_dimensions: (u32, u32),
        k: u32,
        work_texture: &WorkTexture,
        centroid_buffer: &'a CentroidsBuffer,
        color_index_texture: &ColorIndexTexture,
        find_centroid_module: &'a FindCentroidModule,
    ) -> Self {
        const WORKGROUP_SIZE: u32 = 256;
        const N_SEQ: u32 = 20;
        let choose_centroid_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Choose centroid shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/choose_centroid.wgsl").into()),
        });

        let choose_centroid_bind_group_0_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Choose centroid bind group 0 layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Uint,
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let part_id_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let choose_centroid_bind_group_0 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &choose_centroid_bind_group_0_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: centroid_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &color_index_texture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &work_texture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: part_id_buffer.as_entire_binding(),
                },
            ],
        });

        let mut choose_centroid_settings_content: Vec<u8> = Vec::new();
        choose_centroid_settings_content.extend_from_slice(bytemuck::cast_slice(&[N_SEQ]));
        choose_centroid_settings_content
            .extend_from_slice(bytemuck::cast_slice(&[color_space.convergence()]));
        let choose_centroid_settings_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &choose_centroid_settings_content,
            usage: BufferUsages::UNIFORM,
        });

        let (dispatch_size, _) = compute_work_group_count(
            (image_dimensions.0 * image_dimensions.1, 1),
            (WORKGROUP_SIZE * N_SEQ, 1),
        );
        let color_buffer_size = dispatch_size * 8 * 4;
        let color_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: color_buffer_size as BufferAddress,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let flag_buffer_size = dispatch_size * 4;
        let flag_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: flag_buffer_size as BufferAddress,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let convergence_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice::<u32, u8>(&vec![0; k as usize + 1]),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let check_convergence_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: (k + 1) as u64 * 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let choose_centroid_bind_group_1_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Choose centroid bind group 1 layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let choose_centroid_bind_group_1 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &choose_centroid_bind_group_1_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: color_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: flag_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: convergence_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: choose_centroid_settings_buffer.as_entire_binding(),
                },
            ],
        });

        let k_index_buffers: Vec<Buffer> = (0..k)
            .map(|k| {
                device.create_buffer_init(&BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[k]),
                    usage: BufferUsages::UNIFORM,
                })
            })
            .collect();

        let k_index_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Choose centroid bind group 2 layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let bind_groups: Vec<_> = (0..k)
            .map(|k| {
                device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &k_index_bind_group_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &k_index_buffers[k as usize],
                            offset: 0,
                            size: None,
                        }),
                    }],
                })
            })
            .collect();

        let choose_centroid_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Choose centroid pipeline layout"),
                bind_group_layouts: &[
                    &choose_centroid_bind_group_0_layout,
                    &choose_centroid_bind_group_1_layout,
                    &k_index_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Choose centroid pipeline"),
            layout: Some(&choose_centroid_pipeline_layout),
            module: &choose_centroid_shader,
            entry_point: "main",
        });

        let pick_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Choose centroid pipeline"),
            layout: Some(&choose_centroid_pipeline_layout),
            module: &choose_centroid_shader,
            entry_point: "pick",
        });

        Self {
            k,
            pipeline,
            pick_pipeline,
            bind_group_0: choose_centroid_bind_group_0,
            bind_group_1: choose_centroid_bind_group_1,
            bind_groups,
            dispatch_size,
            convergence_buffer: ConvergenceBuffer {
                gpu_buffer: convergence_buffer,
                mapped_buffer: check_convergence_buffer,
            },
            find_centroid_module,
            centroid_buffer,
        }
    }

    pub(crate) async fn compute(&self, device: &Device, queue: &Queue) {
        const MAX_OBS_CHAIN: usize = 64;
        const MAX_ITERATION: u32 = 128;
        const MAX_ITERATION_BEFORE_CONVERGENCE_CHECK: u32 = 8;
        let mut current_iteration = 0;

        println!("Dispatch size {}", self.dispatch_size);

        'iteration: for iteration in 0..MAX_ITERATION {
            current_iteration = iteration;
            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            for k_start in (0..self.k as usize).step_by(MAX_OBS_CHAIN) {
                let max_k = (k_start + MAX_OBS_CHAIN).min(self.k as usize);

                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Choose centroid pass"),
                });
                compute_pass.set_bind_group(0, &self.bind_group_0, &[]);
                compute_pass.set_bind_group(1, &self.bind_group_1, &[]);
                for k in k_start..max_k {
                    compute_pass.set_bind_group(2, &self.bind_groups[k as usize], &[]);
                    compute_pass.set_pipeline(&self.pipeline);
                    compute_pass.dispatch(self.dispatch_size, 1, 1);
                    compute_pass.set_pipeline(&self.pick_pipeline);
                    compute_pass.dispatch(1, 1, 1);
                }
            }

            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);

            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut compute_pass =
                    encoder.begin_compute_pass(&ComputePassDescriptor { label: None });

                self.find_centroid_module.dispatch(&mut compute_pass);
            }

            if iteration > 0 && iteration % MAX_ITERATION_BEFORE_CONVERGENCE_CHECK == 0 {
                encoder.copy_buffer_to_buffer(
                    &self.convergence_buffer.gpu_buffer,
                    0,
                    &self.convergence_buffer.mapped_buffer,
                    0,
                    (self.k + 1) as u64 * 4,
                );

                queue.submit(Some(encoder.finish()));
                let check_convergence_slice = self.convergence_buffer.mapped_buffer.slice(..);
                let check_convergence_future = check_convergence_slice.map_async(MapMode::Read);

                device.poll(wgpu::Maintain::Wait);

                match check_convergence_future.await {
                    Ok(_) => {
                        let convergence_data = bytemuck::cast_slice::<u8, u32>(
                            &check_convergence_slice.get_mapped_range(),
                        )
                        .to_vec();
                        if convergence_data[self.k as usize] >= self.k {
                            // We converged, time to go.
                            debug!("We have convergence, checked at iteration {iteration}");
                            break 'iteration;
                        }
                    }
                    Err(_) => break 'iteration,
                };

                self.convergence_buffer.mapped_buffer.unmap();
            } else {
                queue.submit(Some(encoder.finish()));
            }
        }

        if log_enabled!(log::Level::Debug) {
            debug!("== Final centroids at iteration {current_iteration}: ==");
            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });

            let staging_buffer = self.centroid_buffer.staging_buffer(device, &mut encoder);

            queue.submit(Some(encoder.finish()));
            let cent_buffer_slice = staging_buffer.slice(..);
            let cent_buffer_future = cent_buffer_slice.map_async(MapMode::Read);

            device.poll(wgpu::Maintain::Wait);

            if let Ok(()) = cent_buffer_future.await {
                let data = cent_buffer_slice.get_mapped_range();

                for (index, k) in bytemuck::cast_slice::<u8, f32>(&data[16..])
                    .chunks_exact(4)
                    .enumerate()
                {
                    debug!("Centroid {index} = {k:?}")
                }
            }
            staging_buffer.unmap();
            debug!("========================");
        }
    }
}

struct DistanceMapTexture(Texture);

impl DistanceMapTexture {
    fn new(device: &Device, (width, height): (u32, u32)) -> Self {
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Distance map texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        });

        Self(texture)
    }

    fn texture_2d_layout(binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: false },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }
    }

    fn texture_storage_layout(binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::R32Float,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        }
    }
}

pub(crate) struct PlusPlusInitModule<'a> {
    k: u32,
    image_dimensions: (u32, u32),
    centroid_buffer: &'a CentroidsBuffer,
    work_texture: &'a WorkTexture,
}

impl<'a> PlusPlusInitModule<'a> {
    pub(crate) fn new(
        image_dimensions: (u32, u32),
        k: u32,
        work_texture: &'a WorkTexture,
        centroid_buffer: &'a CentroidsBuffer,
    ) -> Self {
        Self {
            k,
            image_dimensions,
            centroid_buffer,
            work_texture,
        }
    }

    pub(crate) async fn compute(&self, device: &Device, queue: &Queue) {
        const WORKGROUP_SIZE: u32 = 256;
        const N_SEQ: u32 = 16;
        const MAX_OPERATIONS_CHAIN: usize = 32;

        let distance_map_texture = DistanceMapTexture::new(device, self.image_dimensions);
        let choose_centroid_shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Plus plus init shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/plus_plus_init.wgsl").into()),
        });

        let choose_centroid_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Choose centroid bind group 0 layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    DistanceMapTexture::texture_2d_layout(5),
                ],
            });

        let (dispatch_size, _) = compute_work_group_count(
            (self.image_dimensions.0 * self.image_dimensions.1, 1),
            (WORKGROUP_SIZE * N_SEQ, 1),
        );
        let prefix_buffer_size = dispatch_size * 4 * 4;
        let prefix_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: prefix_buffer_size as BufferAddress,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let flag_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice::<u32, u8>(&vec![0; dispatch_size as usize]),
            usage: BufferUsages::STORAGE,
        });
        let part_id_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &choose_centroid_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.centroid_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &self
                            .work_texture
                            .create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: prefix_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: flag_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: part_id_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(
                        &distance_map_texture
                            .0
                            .create_view(&TextureViewDescriptor::default()),
                    ),
                },
            ],
        });

        let k_index_buffers: Vec<Buffer> = (0..self.k)
            .map(|k| {
                device.create_buffer_init(&BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[k]),
                    usage: BufferUsages::UNIFORM,
                })
            })
            .collect();

        let k_index_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Choose centroid bind group 2 layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let bind_groups: Vec<_> = (0..self.k)
            .map(|k| {
                device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &k_index_bind_group_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: &k_index_buffers[k as usize],
                            offset: 0,
                            size: None,
                        }),
                    }],
                })
            })
            .collect();

        let choose_centroid_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Choose centroid pipeline layout"),
                bind_group_layouts: &[
                    &choose_centroid_bind_group_layout,
                    &k_index_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let initial_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Choose centroid pipeline"),
            layout: Some(&choose_centroid_pipeline_layout),
            module: &choose_centroid_shader,
            entry_point: "initial",
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Choose centroid pipeline"),
            layout: Some(&choose_centroid_pipeline_layout),
            module: &choose_centroid_shader,
            entry_point: "main",
        });
        let pick_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Choose centroid pipeline"),
            layout: Some(&choose_centroid_pipeline_layout),
            module: &choose_centroid_shader,
            entry_point: "pick",
        });

        let centroid_size = (self.k as u64 + 1) * 16;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: centroid_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let calc_diff_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    CentroidsBuffer::layout(0, true),
                    WorkTexture::texture_2d_layout(1),
                    DistanceMapTexture::texture_storage_layout(2),
                ],
            });

        let calc_diff_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&calc_diff_bind_group_layout, &k_index_bind_group_layout],
            push_constant_ranges: &[],
        });

        let calc_diff_shader_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Calc diff shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/kmeans++_calc_diff.wgsl").into()),
        });

        let calc_diff_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &calc_diff_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.centroid_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &self
                            .work_texture
                            .create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &distance_map_texture
                            .0
                            .create_view(&TextureViewDescriptor::default()),
                    ),
                },
            ],
        });

        let calc_diff_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&calc_diff_pipeline_layout),
            module: &calc_diff_shader_module,
            entry_point: "main",
        });

        let calc_diff_dispatch_size = compute_work_group_count(self.image_dimensions, (16, 16));

        for k_start in (0..self.k as usize).step_by(MAX_OPERATIONS_CHAIN) {
            let max_k = (k_start + MAX_OPERATIONS_CHAIN).min(self.k as usize);

            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Plus plus init pass"),
                });
                for k in k_start..max_k {
                    compute_pass.set_bind_group(1, &bind_groups[k as usize], &[]);
                    if k == 0 {
                        compute_pass.set_pipeline(&initial_pipeline);
                        compute_pass.set_bind_group(0, &bind_group, &[]);
                        compute_pass.dispatch(1, 1, 1);
                    } else {
                        // Calculate difference
                        compute_pass.set_pipeline(&calc_diff_pipeline);
                        compute_pass.set_bind_group(0, &calc_diff_bind_group, &[]);
                        compute_pass.dispatch(
                            calc_diff_dispatch_size.0,
                            calc_diff_dispatch_size.1,
                            1,
                        );

                        compute_pass.set_pipeline(&pipeline);
                        compute_pass.set_bind_group(0, &bind_group, &[]);
                        compute_pass.dispatch(dispatch_size, 1, 1);
                        compute_pass.set_pipeline(&pick_pipeline);
                        compute_pass.dispatch(1, 1, 1);
                    }
                }
            }

            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
        }

        if log_enabled!(log::Level::Debug) {
            debug!("== Initial centroids: ==");
            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(
                self.centroid_buffer,
                0,
                &staging_buffer,
                0,
                centroid_size,
            );

            queue.submit(Some(encoder.finish()));
            let cent_buffer_slice = staging_buffer.slice(..);
            let cent_buffer_future = cent_buffer_slice.map_async(MapMode::Read);

            device.poll(wgpu::Maintain::Wait);

            if let Ok(()) = cent_buffer_future.await {
                let data = cent_buffer_slice.get_mapped_range();

                for (index, k) in bytemuck::cast_slice::<u8, f32>(&data[16..])
                    .chunks_exact(4)
                    .enumerate()
                {
                    debug!("Centroid {index} = {k:?}")
                }
            }
            debug!("========================");

            staging_buffer.unmap();
        }
    }
}

pub(crate) struct MixColorsModule {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    dispatch_size: (u32, u32),
}

impl MixColorsModule {
    pub fn new(
        device: &Device,
        image_dimensions: (u32, u32),
        input_texture: &WorkTexture,
        output_texture: &WorkTexture,
        color_index_texture: &ColorIndexTexture,
        centroids_buffer: &CentroidsBuffer,
        mix_mode: &MixMode,
    ) -> Self {
        let shader_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Mix colors shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/mix_colors.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Mix colors bind group layout"),
            entries: &[
                WorkTexture::texture_2d_layout(0),
                WorkTexture::texture_storage_layout(1),
                ColorIndexTexture::texture_2d_layout(2),
                CentroidsBuffer::layout(3, true),
            ],
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Mix colors bind group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &input_texture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &output_texture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &color_index_texture.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: centroids_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Mix colors pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Mix colors pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: match mix_mode {
                MixMode::Dither => "main_dither",
                MixMode::Meld => "main_meld",
            },
        });

        let dispatch_size = compute_work_group_count(image_dimensions, (16, 16));

        Self {
            pipeline,
            bind_group,
            dispatch_size,
        }
    }
}

impl Module for MixColorsModule {
    fn dispatch<'a>(&'a self, compute_pass: &mut ComputePass<'a>) {
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.dispatch(self.dispatch_size.0, self.dispatch_size.1, 1);
    }
}
