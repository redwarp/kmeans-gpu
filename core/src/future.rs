use std::{
    future::Future,
    sync::{
        mpsc::{channel, Receiver},
        Arc,
    },
    task::Poll,
};

use wgpu::{BufferAsyncError, BufferSlice, BufferView, Device};

pub(crate) struct AsyncData<'a> {
    buffer_slice: BufferSlice<'a>,
    device: Arc<Device>,
    receiver: Receiver<Result<(), BufferAsyncError>>,
}

impl<'a> AsyncData<'a> {
    pub fn new(buffer_slice: BufferSlice<'a>, device: Arc<Device>) -> Self {
        let (sender, receiver) = channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).expect("Couldn't notify mapping")
        });

        AsyncData {
            buffer_slice,
            device,
            receiver,
        }
    }
}

impl<'a> Future for AsyncData<'a> {
    type Output = Result<BufferView<'a>, BufferAsyncError>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        self.device.poll(wgpu::MaintainBase::Poll);
        match self.receiver.try_recv() {
            Ok(received) => match received {
                Ok(_) => Poll::Ready(Ok(self.buffer_slice.get_mapped_range())),
                Err(e) => Poll::Ready(Err(e)),
            },
            Err(_) => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}
