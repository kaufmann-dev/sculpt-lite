mod app;
mod camera;
mod history;
mod mesh;
mod renderer;
mod sculpt;
mod stl;

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let initial_path = std::env::args_os().nth(1).map(std::path::PathBuf::from);

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("SculptLite")
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([900.0, 600.0]),
        renderer: eframe::Renderer::Wgpu,
        depth_buffer: renderer::REQUIRED_DEPTH_BITS,
        ..Default::default()
    };

    eframe::run_native(
        "SculptLite",
        options,
        Box::new(move |creation_context| {
            Ok(Box::new(app::SculptLiteApp::new(
                creation_context,
                initial_path,
            )))
        }),
    )
}
