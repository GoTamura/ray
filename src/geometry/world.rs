struct World {
    vp: ViewPlane,
    background_color: RGBColor,
    sphere: Sphere,
    tracer: &Tracer,
}

impl World {
    fn new();
    fn build() {
        vp.set_hres(200);
        vp.set_vres(200);
        vp.set_pixel_size(1.0);
        vp.set_gamma(1.0);

        self.background_color = black;
        self.tracer = SingleSphere(&self);

        self.sphere.set_center(0.0);
        self.sphere.set_radius(85.0);
    }
    fn render_scene() {
        let mut ray = Ray::new();
        let z = 100.;
        open_window(self.vp.hres, self.vp.vres);
        ray.d = Vector3f {x: 0., y: 0., z: -1.);

        for r in (0..self.vp.vres) {
            for c in (0..self.vp.hres) {
                let x = self.vp.s * (c - 0.5 * (self.vp.hres - 1.0));
                let y = self.vp.s * (r - 0.5 * (self.vp.vres - 1.0));
                ray.o = Point3f{x, y, z};
                let pixel_color = self.tracer.trace_ray(ray);
                display_pixel(r, c, pixel_color);
            }
        }
    }
    fn open_window(hres: i32, vres: i32);
    fn display_pixel(row: i32, column: i32, pixel_color: &RGBColor);
}

struct ViewPlane {
    hres: i32,
    vres: i32,
    s: f64,
    gamma: f64,
    inv_gamma: f64,
}
