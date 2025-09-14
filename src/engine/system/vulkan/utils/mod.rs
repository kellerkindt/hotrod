use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::ValidationError;

pub mod pipeline;

#[macro_export]
macro_rules! shader_from_path {
    ($device:expr, $ty:literal, $path:literal) => {{
        mod shader {
            pub const ENTRY_POINT: &'static str = "main";
            vulkano_shaders::shader!(
                ty: $ty,
                path: $path
            );
        }
        shader::load($device)
            .map_err(ShaderLoadError::from)
            .and_then(|shader|
                shader
                    .entry_point(shader::ENTRY_POINT)
                    .ok_or_else(|| ShaderLoadError::MissingEntryPoint($ty, shader::ENTRY_POINT))
            )
    }}
}

pub trait Draw {
    fn hotrod_draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<&mut Self, Box<ValidationError>>;

    fn hotrod_draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<&mut Self, Box<ValidationError>>;
}

impl<T> Draw for AutoCommandBufferBuilder<T> {
    #[inline(always)]
    fn hotrod_draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        #[cfg(not(debug_assertions))]
        unsafe {
            Ok(self.draw_unchecked(vertex_count, instance_count, first_vertex, first_instance))
        }
        #[cfg(debug_assertions)]
        unsafe {
            self.draw(vertex_count, instance_count, first_vertex, first_instance)
        }
    }

    #[inline(always)]
    fn hotrod_draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        #[cfg(not(debug_assertions))]
        unsafe {
            Ok(self.draw_indexed_unchecked(
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            ))
        }
        #[cfg(debug_assertions)]
        unsafe {
            self.draw_indexed(
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        }
    }
}
