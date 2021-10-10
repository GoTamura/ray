use crate::ray::Ray;
use crate::utils::*;

pub trait Camera {
    fn generate_ray() -> Ray;
}
pub struct SingleLensCamera {
    origin: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
    lens_radius: f64,
    time0: f64,
    time1: f64,
}

impl SingleLensCamera {
    pub fn new(
        lookfrom: &Point3,
        lookat: &Point3,
        vup: &Vec3,
        vfov: f64,
        aspcet_ratio: f64,
        aperture: f64,
        focus_dist: f64,
        time0: f64,
        time1: f64,
    ) -> Self {
        let theta = degrees_to_radians(vfov);
        let h = f64::tan(theta / 2.);
        let viewport_height = 2.0 * h;
        let viewport_width = viewport_height * aspcet_ratio;

        let w = (lookfrom - lookat).normalize();
        let u = vup.cross(&w).normalize();
        let v = w.cross(&u);

        let origin = *lookfrom;
        let horizontal = focus_dist * viewport_width * u;
        let vertical = focus_dist * viewport_height * v;
        let lower_left_corner = origin - horizontal / 2. - vertical / 2. - focus_dist * w;
        let lens_radius = aperture / 2.;
        Self {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
            u,
            v,
            w,
            lens_radius,
            time0,
            time1,
        }
    }

    pub fn generate_ray(&self, s: f64, t: f64) -> Ray {
        let rd = self.lens_radius * random_in_unit_disk();
        let offset = self.u * rd.x + self.v * rd.y;
        Ray::with_time(
            &(self.origin + offset),
            &(self.lower_left_corner + s * self.horizontal + t * self.vertical
                - self.origin
                - offset),
            random_double(self.time0, self.time1),
        )
    }
}
