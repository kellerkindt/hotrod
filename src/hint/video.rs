pub fn prefer_wayland() {
    std::env::set_var("SDL_VIDEODRIVER", "wayland,x11");
}
