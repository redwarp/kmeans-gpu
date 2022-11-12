use std::{
    future::Future,
    sync::mpsc::{channel, Receiver},
    task::Poll,
};

use wgpu::{BufferAsyncError, BufferSlice, BufferView, Device};

#[must_use]
pub(crate) struct AsyncBufferView<'a> {
    buffer_slice: BufferSlice<'a>,
    device: &'a Device,
    receiver: Receiver<Result<(), BufferAsyncError>>,
}

impl<'a> AsyncBufferView<'a> {
    pub fn new(buffer_slice: BufferSlice<'a>, device: &'a Device) -> Self {
        let (sender, receiver) = channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).expect("Couldn't notify mapping")
        });

        AsyncBufferView {
            buffer_slice,
            device,
            receiver,
        }
    }
}

impl<'a> Future for AsyncBufferView<'a> {
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
