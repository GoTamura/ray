use core::f64;

use nalgebra::{Point2, Point3, Scalar, Vector2, Vector3, center, distance};

use crate::ray::Ray;

#[derive(Debug)]
struct Bounds2 {
    pub p_min: Point2<f64>,
    pub p_max: Point2<f64>,
}

impl Bounds2 {
    pub fn area(&self) -> f64 {
        let d = self.p_max - self.p_min;
        d.x * d.y
    }

    pub fn diagonal(&self) -> Vector2<f64> {
        self.p_max - self.p_min
    }
}

#[derive(Debug)]
pub struct Bounds3 {
    pub p_min: Point3<f64>,
    pub p_max: Point3<f64>,
}

impl Bounds3 {
    pub fn volume(&self) -> f64 {
        let d = self.diagonal();
        d.x * d.y * d.z
    }

    pub fn diagonal(&self) -> Vector3<f64> {
        self.p_max - self.p_min
    }

    pub fn inside(&self, p: &Point3<f64>) -> bool {
        inside(p, &self)
    }

    pub fn bounding_sphere(&self) -> (Point3<f64>, f64) {
        let center = center(&self.p_min, &self.p_max);
        let radius = if self.inside(&center) {
            distance(&center, &self.p_max)
        } else {
            0.
        };
        (center, radius)
    }

    pub fn intersect_p(&self, ray: &Ray, t_min: f64, t_max: f64) -> bool {
        let inverse_ray_direction = Vector3::new(1., 1., 1.).component_div(&ray.d());
        let min = (self.p_min - ray.o()).component_mul(&inverse_ray_direction);
        let max = (self.p_max - ray.o()).component_mul(&inverse_ray_direction);
        let (min, max) = min.inf_sup(&max);
        let t_min = f64::min(min.max(), t_min);
        let t_max = f64::max(max.min(), t_max);
        t_max > t_min
    }

    pub fn union(&self, bounds: &Bounds3) -> Bounds3 {
        let p_min = self.p_min.inf(&bounds.p_min);
        let p_max = self.p_max.sup(&bounds.p_max);
        Bounds3 { p_min, p_max }
    }
}

pub fn inside(p: &Point3<f64>, b: &Bounds3) -> bool {
    p >= &b.p_min && p < &b.p_max
}
