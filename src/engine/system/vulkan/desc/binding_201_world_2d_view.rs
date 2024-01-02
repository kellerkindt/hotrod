use crate::engine::system::vulkan::desc::WriteDescriptorSetOrigin;
use crate::engine::system::vulkan::system::VulkanSystem;

#[derive(Debug, Copy, Clone)]
pub struct World2dView {
    pub x: f32,
    pub y: f32,
    pub zoom: f32,
}

impl From<[f32; 3]> for World2dView {
    #[inline]
    fn from([x, y, zoom]: [f32; 3]) -> Self {
        Self { x, y, zoom }
    }
}

impl From<&VulkanSystem> for World2dView {
    #[inline]
    fn from(_vs: &VulkanSystem) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            zoom: 1.0,
        }
    }
}

impl WriteDescriptorSetOrigin for World2dView {
    type BufferContents = f32;
    type Data = <[f32; 3] as IntoIterator>::IntoIter;

    #[inline]
    fn binding(&self) -> u32 {
        201
    }

    #[inline]
    fn data(&self) -> Self::Data {
        [self.x, self.y, self.zoom].into_iter()
    }
}
