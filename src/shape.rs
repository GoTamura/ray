use std::ops::Deref;
use std::sync::{Arc, RwLock};

use nalgebra::{Matrix3, Scalar, Vector3};

use crate::bounds::Bounds3;
use crate::material::Material;
use crate::ray::Ray;
use crate::utils::*;
pub struct HitRecord {
    pub p: Point3,
    pub normal: Vec3,
    pub material: Arc<dyn Material + Send + Sync>,
    pub t: f64,
    pub front_face: bool,
    pub u: f64,
    pub v: f64,
}

impl HitRecord {
    pub fn set_face_normal(&mut self, ray: &Ray, outward_normal: &Vec3) {
        self.front_face = ray.d().dot(outward_normal) < 0.;
        self.normal = if self.front_face {
            *outward_normal
        } else {
            -*outward_normal
        };
    }
}

pub trait Hittable {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
    fn bounds(&self, time0: f64, time1: f64) -> Option<Bounds3>;
}

pub struct Sphere {
    pub center: Point3,
    pub radius: f64,
    pub material: Arc<dyn Material + Send + Sync>,
}

impl Sphere {
    pub fn uv(p: &Point3) -> (f64, f64) {
        let theta = f64::acos(-p.y);
        let phi = f64::atan2(-p.z, p.x) + std::f64::consts::PI;

        let u = phi / (2. * std::f64::consts::PI);
        let v = theta / std::f64::consts::PI;
        (u, v)
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc = ray.o() - self.center;
        let a = ray.d().magnitude_squared();
        let half_b = oc.dot(&ray.d());
        let c = oc.magnitude_squared() - self.radius * self.radius;

        let discriminant = half_b * half_b - a * c;

        if discriminant < 0. {
            return None;
        };

        let sqrtd = discriminant.sqrt();

        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            root = (-half_b - sqrtd) / a;
            if root < t_min || t_max < root {
                return None;
            };
        }

        let outward_normal = (ray.at(root) - self.center) / self.radius;
        let (u, v) = Self::uv(&outward_normal.into());
        Some({
            let mut hit_record = HitRecord {
                p: ray.at(root),
                normal: (ray.at(root) - self.center) / self.radius,
                t: root,
                front_face: false,
                material: self.material.clone(),
                u,
                v,
            };
            hit_record.set_face_normal(ray, &outward_normal);
            hit_record
        })
    }

    fn bounds(&self, time0: f64, tim1: f64) -> Option<Bounds3> {
        Some(Bounds3 {
            p_min: self.center - Vec3::new(self.radius, self.radius, self.radius),
            p_max: self.center + Vec3::new(self.radius, self.radius, self.radius),
        })
    }
}

pub type HittableList = std::vec::Vec<std::sync::Arc<dyn Hittable + Send + Sync>>;

impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut hit_record = None;
        let mut closest_so_far = t_max;
        for i in self.iter() {
            if let Some(rec) = i.hit(ray, t_min, t_max) {
                if rec.t < closest_so_far {
                    closest_so_far = rec.t;
                    hit_record = Some(rec);
                }
            }
        }
        hit_record
    }

    fn bounds(&self, time0: f64, time1: f64) -> Option<Bounds3> {
        let mut first_box = true;
        let mut output_bounds = self[0].bounds(time0, time1)?;
        for i in self.iter() {
            if let Some(temp_bounds) = i.bounds(time0, time1) {
                output_bounds = if first_box {
                    temp_bounds
                } else {
                    output_bounds.union(&temp_bounds)
                };
            }
        }
        Some(output_bounds)
    }
}

trait Shape {
    fn object_bound(&self) -> Bounds3;
    fn world_bound(&self) -> Bounds3;
}

pub struct MovingSphere {
    pub center0: Point3,
    pub center1: Point3,
    pub time0: f64,
    pub time1: f64,
    pub radius: f64,
    pub material: Arc<dyn Material + Send + Sync>,
}

impl MovingSphere {
    fn center(&self, time: f64) -> Point3 {
        self.center0
            + ((time - self.time0) / (self.time1 - self.time0)) * (self.center1 - self.center0)
    }
}

impl Hittable for MovingSphere {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc = ray.o() - self.center(ray.time);
        let a = ray.d().magnitude_squared();
        let half_b = oc.dot(&ray.d());
        let c = oc.magnitude_squared() - self.radius * self.radius;

        let discriminant = half_b * half_b - a * c;

        if discriminant < 0. {
            return None;
        };

        let sqrtd = discriminant.sqrt();

        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            root = (-half_b - sqrtd) / a;
            if root < t_min || t_max < root {
                return None;
            };
        }

        let outward_normal = (ray.at(root) - self.center(ray.time)) / self.radius;
        Some({
            let mut hit_record = HitRecord {
                p: ray.at(root),
                normal: (ray.at(root) - self.center(ray.time)) / self.radius,
                t: root,
                front_face: false,
                material: self.material.clone(),
                u: 0.,
                v: 0.,
            };
            hit_record.set_face_normal(ray, &outward_normal);
            hit_record
        })
    }

    fn bounds(&self, time0: f64, tim1: f64) -> Option<Bounds3> {
        let bounds0 = Bounds3 {
            p_min: self.center0 - Vec3::new(self.radius, self.radius, self.radius),
            p_max: self.center0 + Vec3::new(self.radius, self.radius, self.radius),
        };
        let bounds1 = Bounds3 {
            p_min: self.center1 - Vec3::new(self.radius, self.radius, self.radius),
            p_max: self.center1 + Vec3::new(self.radius, self.radius, self.radius),
        };
        Some(bounds0.union(&bounds1))
    }
}

struct BVHNode {
    left: Arc<RwLock<dyn Hittable>>,
    right: Arc<RwLock<dyn Hittable>>,
    bounds: Bounds3,
}

impl BVHNode {
    fn intersect_p(&self, ray: &Ray, t_min: f64, t_max: f64) -> bool {
        if !self.bounds.intersect_p(ray, t_min, t_max) {
            return false;
        }

        let intersect_left = self
            .left
            .read()
            .unwrap()
            .bounds(t_min, t_max)
            .unwrap()
            .intersect_p(ray, t_min, t_max);
        let intersect_right = self
            .right
            .read()
            .unwrap()
            .bounds(t_min, t_max)
            .unwrap()
            .intersect_p(ray, t_min, t_max);
        intersect_left || intersect_right
    }

    // fn new(world: &Vec<Arc<dyn Hittable>>, start: usize, end: usize, time0: f64, time1: f64) -> Self {

    // }
}

struct TriangleMesh {}

struct Triangle {
    pub index: Vector3<usize>,
    pub positions: Arc<Vec<Point3>>,
    pub normals: Arc<Vec<Vec3>>,
    pub tex_coords: Arc<Vec<Point2>>,
}

impl Hittable for Triangle {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let p0 = self.positions[self.index[0]];
        let p1 = self.positions[self.index[1]];
        let p2 = self.positions[self.index[2]];

        let p = Matrix3::from_columns(&[p0.coords, p1.coords, p2.coords])
            - Matrix3::from_columns(&[ray.o().coords, ray.o().coords, ray.o().coords]);

        // 最も大きい方向成分をzにするように並び替える
        let kz = ray.d().iamax();
        let kx = if kz == 2 { 0 } else { kz + 1 };
        let ky = if kx == 2 { 0 } else { kx + 1 };

        let d = permute(&ray.d(), kx, ky, kz);
        let mut p = permute_matrix(&p, kx, ky, kz);

        let s = Matrix3::new(1. , 0. ,-d.x / d.z, 0., 1., -d.y / d.z, 0., 0., 1./d.z);
        
        let p = s * p;

        let e = p.row(0).cross(&p.row(1));

        if (e.x < 0. || e.y < 0. || e.z < 0.) && (e.x > 0. || e.y > 0. || e.z > 0.) {
            return None;
        }

        let det = e.x + e.y + e.z;
        if det == 0. {
            return None;
        }

        let t_scaled = e.dot(&p.row(2));
        if det < 0. && (t_scaled >= 0. || t_scaled < t_max * det) {
            return None;
        } else if det > 0. && (t_scaled <= 0. || t_scaled > t_max * det) {
            return None;
        }

        let inv_det = 1. / det;
        let b = e * inv_det;
        let t = t_scaled * inv_det;

        // calculate tangent vector
        //

        let duv02 = self.tex_coords[0] - self.tex_coords[1];
        let duv12 = self.tex_coords[1] - self.tex_coords[2];
        let dp02 = p.column(0) - p.column(2);
        let dp12 = p.column(1) - p.column(2);

        let mut dpdu;
        let mut dpdv;
        let determinant = duv02.x * duv12.y - duv02.y * duv12.x;
        if determinant == 0. {
            todo!();
        } else{
            let inv_det = 1. / determinant;
            dpdu = (duv12.y * dp02 - duv02.y * dp12) * inv_det;
            dpdv = (duv12.x * dp02 - duv02.x * dp12) * inv_det;
        }

        let p_hit = b * p;
        let uv_hit = Point2::from(
            b.x * self.tex_coords[0].coords
                + b.y * self.tex_coords[1].coords
                + b.z * self.tex_coords[2].coords,
        );
        
        let normal = (b.x * self.normals[0] + b.y * self.normals[1] + b.z * self.normals[2]).normalize();

        Some(HitRecord {
            p: todo!(),
            normal,
            material: todo!(),
            t: todo!(),
            front_face: true,
            u: uv_hit.x,
            v: uv_hit.y,
        })
    }

    fn bounds(&self, time0: f64, time1: f64) -> Option<Bounds3> {
        todo!()
    }
}

impl Triangle {
    pub fn uv(p: &Point3) -> (f64, f64) {
        let theta = f64::acos(-p.y);
        let phi = f64::atan2(-p.z, p.x) + std::f64::consts::PI;

        let u = phi / (2. * std::f64::consts::PI);
        let v = theta / std::f64::consts::PI;
        (u, v)
    }
}

fn permute(p: &Vec3, x: usize, y: usize, z: usize) -> Point3 {
    Point3::new(p[x], p[y], p[z])
}

fn permute_matrix<T: Clone + Scalar>(p: &Matrix3<T>, x: usize, y: usize, z: usize) -> Matrix3<T> {
    Matrix3::from_rows(&[p.row(x), p.row(y), p.row(z)])
}


struct SurfaceInteraction {
    p: Point3,
    n: Vec3,
    uv: Point2,
    dpdu: Vec3,
    dpdv: Vec3,
    dndu: Vec3,
    dndv: Vec3,
}