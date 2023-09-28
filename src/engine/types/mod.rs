pub mod world2d {
    pub type Pos<T> = cgmath::Point2<T>;
    pub type Dim<T> = cgmath::Vector2<T>;

    #[derive(Debug, Copy, Clone, PartialEq)]
    pub struct Rect<T> {
        pub pos: Pos<T>,
        pub dim: Dim<T>,
    }

    impl<T> Rect<T> {
        #[inline]
        pub const fn new(pos: Pos<T>, dim: Dim<T>) -> Self {
            Self { pos, dim }
        }
    }
}
