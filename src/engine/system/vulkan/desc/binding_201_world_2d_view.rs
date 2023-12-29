use crate::engine::system::vulkan::desc::WriteDescriptorSetOrigin;
use crate::engine::system::vulkan::system::VulkanSystem;

pub struct World2dView {
    x: f32,
    y: f32,
}

impl From<[f32; 2]> for World2dView {
    #[inline]
    fn from([x, y]: [f32; 2]) -> Self {
        Self { x, y }
    }
}

impl From<&VulkanSystem> for World2dView {
    #[inline]
    fn from(_vs: &VulkanSystem) -> Self {
        Self { x: 0.0, y: 0.0 }
    }
}

impl WriteDescriptorSetOrigin for World2dView {
    type BufferContents = f32;
    type Data = <[f32; 2] as IntoIterator>::IntoIter;

    #[inline]
    fn binding(&self) -> u32 {
        201
    }

    #[inline]
    fn data(&self) -> Self::Data {
        [self.x, self.y].into_iter()
    }
}
