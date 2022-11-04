use std::{borrow::Cow, sync::mpsc::channel};

use palette::{IntoColor, Lab, Srgb};
use pollster::FutureExt;
use wgpu::{
    util::{self, DeviceExt},
    Backends, BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePipelineDescriptor, Device, DeviceDescriptor, Instance,
    MaintainBase, MapMode, Queue, ShaderSource,
};

use crate::modules::include_shader;

struct TestingContext {
    device: Device,
    queue: Queue,
}

trait ToLab {
    fn to_lab(self) -> [f32; 3];
}

impl ToLab for [u8; 3] {
    fn to_lab(self) -> [f32; 3] {
        let lab: Lab = Srgb::new(self[0], self[1], self[2])
            .into_format()
            .into_color();
        [lab.l, lab.a, lab.b]
    }
}

fn vec3x2_as_input_f32_as_output(
    shader_string: Cow<str>,
    entry_point: &str,
    a: [f32; 3],
    b: [f32; 3],
) -> f32 {
    run_wgpu_test(|context| {
        let shader_module = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(shader_string),
            });

        let mut input_data: Vec<u8> = vec![];
        input_data.extend_from_slice(bytemuck::cast_slice(&a));
        input_data.extend_from_slice(&[0, 0, 0, 0]);
        input_data.extend_from_slice(bytemuck::cast_slice(&b));
        input_data.extend_from_slice(&[0, 0, 0, 0]);

        let input_buffer = context
            .device
            .create_buffer_init(&util::BufferInitDescriptor {
                label: None,
                contents: &input_data,
                usage: BufferUsages::UNIFORM,
            });
        let output_buffer = context.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stating_buffer = context.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compute_pipeline = context
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader_module,
                entry_point,
            });

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = context.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = context
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("Compute cie94");
            cpass.dispatch_workgroups(1, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &stating_buffer, 0, 4);
        context.queue.submit(Some(encoder.finish()));

        let buffer_slice = stating_buffer.slice(..);
        let (sender, receiver) = channel();
        buffer_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());

        context.device.poll(MaintainBase::Wait);

        if let Ok(_) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let result: f32 = bytemuck::cast_slice(&data)[0];

            drop(data);
            stating_buffer.unmap();

            result
        } else {
            panic!("Failed to compute cie94 on gpu!")
        }
    })
}

fn run_wgpu_test<T>(test_function: impl FnOnce(&TestingContext) -> T) -> T {
    let backend_bits = util::backend_bits_from_env().unwrap_or_else(Backends::all);
    let instance = Instance::new(backend_bits);
    let adapter = pollster::block_on(util::initialize_adapter_from_env_or_default(
        &instance,
        backend_bits,
        None,
    ))
    .expect("could not find suitable adapter on the system");

    let features = adapter.features();
    let limits = adapter.limits();

    let (device, queue) = adapter
        .request_device(
            &DeviceDescriptor {
                label: None,
                features,
                limits,
            },
            None,
        )
        .block_on()
        .expect("Couldn't create device");

    let context = TestingContext { device, queue };

    let result = test_function(&context);

    context.device.poll(MaintainBase::Wait);

    result
}

#[test]
fn test_delta_e_cie94() {
    fn delta_e_cie94(a: [f32; 3], b: [f32; 3]) -> f32 {
        vec3x2_as_input_f32_as_output(
            include_shader!("shaders/tests/test_distance.wgsl").into(),
            "run_distance_cie94",
            a,
            b,
        )
    }

    let delta_e = delta_e_cie94([255, 0, 0].to_lab(), [255, 128, 0].to_lab());

    assert!((delta_e - 19.094658).abs() < 0.01);
}

#[test]
fn test_delta_e_cie2000() {
    fn delta_e_cie2000(a: [f32; 3], b: [f32; 3]) -> f32 {
        vec3x2_as_input_f32_as_output(
            include_shader!("shaders/tests/test_distance.wgsl").into(),
            "run_distance_cie2000",
            a,
            b,
        )
    }

    let lab1 = [50.0000, 2.6772, -79.7751];
    let lab2 = [50.0000, 0.0000, -82.7485];

    let delta_e_0 = delta_e_cie2000(lab1, lab2);

    assert!((delta_e_0 - 2.0424595).abs() < 0.01);

    let delta_e_1 = delta_e_cie2000([255, 0, 0].to_lab(), [255, 128, 0].to_lab());

    assert!((delta_e_1 - 21.164806).abs() < 0.01);
}

#[test]
fn test_dummy_pow() {
    fn run_pow(number: f32, pow: f32) -> f32 {
        vec3x2_as_input_f32_as_output(
            include_shader!("shaders/tests/test_distance.wgsl").into(),
            "run_pow",
            [number, pow, 0.0],
            [0.0, 0.0, 0.0],
        )
    }

    let number = 25.0;
    let pow = 4.0;

    let result = run_pow(number, pow);
    let expected = number.powf(pow);

    assert_eq!(result, expected);
    assert!(
        (result - expected).abs() < 0.5,
        "{number}^{pow} = {expected}, was {result}"
    );
}
