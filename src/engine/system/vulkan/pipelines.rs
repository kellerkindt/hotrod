use crate::engine::system::vulkan::beautiful_lines::BeautifulLinePipeline;
use crate::engine::system::vulkan::lines::LinePipeline;
use crate::engine::system::vulkan::system::VulkanSystem;
use crate::engine::system::vulkan::textured::TexturedPipeline;
use crate::engine::system::vulkan::triangles::TrianglesPipeline;
use crate::engine::system::vulkan::world2d::entities::World2dEntitiesPipeline;
use crate::engine::system::vulkan::world2d::terrain::World2dTerrainPipeline;
use crate::engine::system::vulkan::PipelineCreateError;

pub struct VulkanPipelines {
    pub line: LinePipeline,
    pub texture: TexturedPipeline,
    pub triangles: TrianglesPipeline,
    pub beautiful_line: BeautifulLinePipeline,
    pub world2d_terrain: World2dTerrainPipeline,
    pub world2d_entities: World2dEntitiesPipeline,
    #[cfg(feature = "ui-egui")]
    pub egui: crate::engine::system::vulkan::egui::EguiPipeline,
}

impl TryFrom<&VulkanSystem> for VulkanPipelines {
    type Error = PipelineCreateError;

    fn try_from(vs: &VulkanSystem) -> Result<Self, Self::Error> {
        Ok(Self {
            line: LinePipeline::try_from(vs)?,
            texture: TexturedPipeline::try_from(vs)?,
            triangles: TrianglesPipeline::try_from(vs)?,
            beautiful_line: BeautifulLinePipeline::try_from(vs)?,
            world2d_terrain: World2dTerrainPipeline::try_from(vs)?,
            world2d_entities: World2dEntitiesPipeline::try_from(vs)?,
            #[cfg(feature = "ui-egui")]
            egui: crate::engine::system::vulkan::egui::EguiPipeline::try_from(vs)?,
        })
    }
}
