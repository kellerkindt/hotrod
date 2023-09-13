use crate::engine::system::vulkan::desc::WriteDescriptorSetOrigin;
use crate::engine::system::vulkan::system::VulkanSystem;

pub struct WindowSize {
    width: f32,
    height: f32,
}

impl From<&VulkanSystem> for WindowSize {
    fn from(vs: &VulkanSystem) -> Self {
        let [width, height] = vs.swapchain().image_extent();
        Self {
            width: width as f32,
            height: height as f32,
        }
    }
}

impl WriteDescriptorSetOrigin for WindowSize {
    type BufferContents = f32;
    type Data = <[f32; 2] as IntoIterator>::IntoIter;

    #[inline]
    fn binding(&self) -> u32 {
        101
    }

    #[inline]
    fn data(&self) -> Self::Data {
        [self.width, self.height].into_iter()
    }
}
