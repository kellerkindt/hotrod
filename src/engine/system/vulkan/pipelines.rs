use crate::engine::system::vulkan::beautiful_lines::BeautifulLinePipeline;
use crate::engine::system::vulkan::lines::LinePipeline;
use crate::engine::system::vulkan::system::VulkanSystem;
use crate::engine::system::vulkan::textures::TexturesPipeline;
use crate::engine::system::vulkan::PipelineCreateError;

pub struct VulkanPipelines {
    pub line: LinePipeline,
    pub texture: TexturesPipeline,
    pub beautiful_line: BeautifulLinePipeline,
    #[cfg(feature = "ui-egui")]
    pub egui: crate::engine::system::vulkan::egui::EguiPipeline,
}

impl TryFrom<&VulkanSystem> for VulkanPipelines {
    type Error = PipelineCreateError;

    fn try_from(vs: &VulkanSystem) -> Result<Self, Self::Error> {
        Ok(Self {
            line: LinePipeline::try_from(vs)?,
            texture: TexturesPipeline::try_from(vs)?,
            beautiful_line: BeautifulLinePipeline::try_from(vs)?,
            #[cfg(feature = "ui-egui")]
            egui: crate::engine::system::vulkan::egui::EguiPipeline::try_from(vs)?,
        })
    }
}
