use std::num::NonZeroU32;

use pollster::FutureExt;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferAddress, BufferBinding,
    BufferBindingType, BufferDescriptor, BufferUsages, ComputePass, ComputePipeline,
    ComputePipelineDescriptor, Device, MapMode, PipelineLayoutDescriptor, Queue, ShaderSource,
    ShaderStages, StorageTextureAccess, Texture, TextureFormat, TextureViewDescriptor,
    TextureViewDimension,
};

use crate::{utils::compute_work_group_count, ColorSpace};

pub(crate) trait Module {
    fn dispatch<'a>(&'a self, compute_pass: &mut ComputePass<'a>);
}

pub(crate) trait ComputeBlock<T> {
    fn compute(&mut self, device: &Device, queue: &Queue) -> T;
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
        input_texture: &Texture,
        work_texture: &Texture,
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
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
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
                            format: TextureFormat::Rgba32Float,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
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
                    resource: BindingResource::TextureView(&work_texture.create_view(
                        &TextureViewDescriptor {
                            label: None,
                            format: Some(TextureFormat::Rgba32Float),
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
        work_texture: &Texture,
        output_texture: &Texture,
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
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
                    resource: BindingResource::TextureView(&work_texture.create_view(
                        &TextureViewDescriptor {
                            label: None,
                            format: Some(TextureFormat::Rgba32Float),
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
        work_texture: &Texture,
        centroid_buffer: &Buffer,
        color_index_buffer: &Buffer,
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
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
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
                    resource: color_index_buffer.as_entire_binding(),
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
        work_texture: &Texture,
        centroid_buffer: &Buffer,
        color_index_buffer: &Buffer,
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
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
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
                    resource: BindingResource::TextureView(&work_texture.create_view(
                        &TextureViewDescriptor {
                            label: None,
                            format: Some(TextureFormat::Rgba32Float),
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
                    resource: centroid_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: color_index_buffer.as_entire_binding(),
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
    bind_group_0: BindGroup,
    bind_group_1: BindGroup,
    bind_groups: Vec<BindGroup>,
    dispatch_size: u32,
    convergence_buffer: ConvergenceBuffer,
    find_centroid_module: &'a FindCentroidModule,
}

impl<'a> ChooseCentroidModule<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        color_space: &ColorSpace,
        image_dimensions: (u32, u32),
        k: u32,
        work_texture: &Texture,
        centroid_buffer: &Buffer,
        color_index_buffer: &Buffer,
        find_centroid_module: &'a FindCentroidModule,
    ) -> Self {
        const WORKGROUP_SIZE: u32 = 256;
        const N_SEQ: u32 = 24;
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
                        ty: BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
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
                    resource: color_index_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &work_texture.create_view(&TextureViewDescriptor::default()),
                    ),
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
        let state_buffer_size = dispatch_size * 4;
        let state_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: state_buffer_size as BufferAddress,
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
                    resource: state_buffer.as_entire_binding(),
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

        Self {
            k,
            pipeline,
            bind_group_0: choose_centroid_bind_group_0,
            bind_group_1: choose_centroid_bind_group_1,
            bind_groups,
            dispatch_size,
            convergence_buffer: ConvergenceBuffer {
                gpu_buffer: convergence_buffer,
                mapped_buffer: check_convergence_buffer,
            },
            find_centroid_module,
        }
    }
}

impl<'a> ComputeBlock<()> for ChooseCentroidModule<'a> {
    fn compute(&mut self, device: &Device, queue: &Queue) {
        let max_obs_chain = 32;
        let max_iteration = 128;
        let max_step_before_convergence_check = 8;
        let mut iteration = 0;
        let mut op_count;
        let mut current_k = 0;
        let mut current_step = 0;

        'iteration: loop {
            op_count = 0;
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Choose centroid pass"),
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group_0, &[]);
            compute_pass.set_bind_group(1, &self.bind_group_1, &[]);

            #[allow(clippy::mut_range_bound)]
            for step in current_step..max_step_before_convergence_check {
                for k in current_k..self.k {
                    compute_pass.set_bind_group(2, &self.bind_groups[k as usize], &[]);
                    compute_pass.dispatch(self.dispatch_size, 1, 1);
                    op_count += 1;

                    #[allow(clippy::mut_range_bound)]
                    if op_count >= max_obs_chain {
                        current_k = k + 1;
                        current_step = step;
                        break;
                    }
                }

                if op_count >= max_obs_chain {
                    if current_k == self.k {
                        // We actually finished this step.
                        iteration += 1;
                        current_step += 1;

                        if iteration >= max_iteration {
                            break 'iteration;
                        }
                    }

                    drop(compute_pass);
                    queue.submit(Some(encoder.finish()));
                    continue 'iteration;
                } else {
                    current_k = 0;
                }

                self.find_centroid_module.dispatch(&mut compute_pass);

                iteration += 1;
                current_step += 1;

                if iteration >= max_iteration {
                    break 'iteration;
                } else {
                    compute_pass.set_pipeline(&self.pipeline);
                    compute_pass.set_bind_group(0, &self.bind_group_0, &[]);
                    compute_pass.set_bind_group(1, &self.bind_group_1, &[]);
                }
            }
            current_step = 0;
            drop(compute_pass);
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

            match check_convergence_future.block_on() {
                Ok(_) => {
                    let convergence_data = bytemuck::cast_slice::<u8, u32>(
                        &check_convergence_slice.get_mapped_range(),
                    )
                    .to_vec();
                    if convergence_data[self.k as usize] >= self.k {
                        // We converged, time to go.
                        println!("We have convergence, checked at iteration {iteration}");
                        break;
                    }
                }
                Err(_) => break 'iteration,
            };

            self.convergence_buffer.mapped_buffer.unmap();
        }
    }
}

struct ConvertCentroidColorsContext {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
}

pub(crate) struct ConvertCentroidColorsModule {
    k: u32,
    context: Option<ConvertCentroidColorsContext>,
}

impl ConvertCentroidColorsModule {
    pub(crate) fn new(
        device: &Device,
        k: u32,
        color_space: &ColorSpace,
        centroid_buffer: &Buffer,
    ) -> Self {
        let context = match color_space {
            ColorSpace::Lab => {
                let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: ShaderSource::Wgsl(
                        include_str!("shaders/converters/centroid_lab_to_rgb.wgsl").into(),
                    ),
                });

                let bind_group_layout =
                    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        }],
                    });

                let bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: centroid_buffer.as_entire_binding(),
                    }],
                });

                let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

                let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: "main",
                });

                Some(ConvertCentroidColorsContext {
                    pipeline,
                    bind_group,
                })
            }

            ColorSpace::Rgb => None,
        };

        Self { k, context }
    }
}

impl Module for ConvertCentroidColorsModule {
    fn dispatch<'a>(&'a self, compute_pass: &mut ComputePass<'a>) {
        if let Some(context) = &self.context {
            compute_pass.set_pipeline(&context.pipeline);
            compute_pass.set_bind_group(0, &context.bind_group, &[]);
            compute_pass.dispatch(1, 1, 1);
        }
    }
}
