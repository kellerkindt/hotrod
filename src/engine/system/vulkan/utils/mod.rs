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
