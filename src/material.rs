use std::sync::Arc;

use crate::texture::Texture;
use crate::utils::*;
use crate::ray::Ray;
use crate::shape::HitRecord;

pub trait Material {
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord) -> Option<(Color, Ray)>;
    fn emitted(&self, u: f64, v: f64, p: &Point3) -> Color {
        Color::new(0.,0.,0.)
    }
}

pub struct Lambertian {
    pub albedo: Arc<dyn Texture+Send+Sync>,
}

impl Material for Lambertian {
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord) -> Option<(Color, Ray)> {
        let mut scatter_direction = hit_record.normal + random_unit_vector();
        if near_zero(&scatter_direction) {
            scatter_direction = hit_record.normal;
        };
        let attenuation = self.albedo.value(hit_record.u, hit_record.v, &hit_record.p);
        let scattered = Ray::with_time(&hit_record.p, &scatter_direction, ray.time);
        Some((attenuation, scattered))
    }
}

pub struct Metal {
    pub albedo: Color,
    pub fuzz: f64,
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord) -> Option<(Color, Ray)> {
        let reflected = reflect(&ray.d().normalize(), &hit_record.normal);
        let attenuation = self.albedo;
        let scattered = Ray::with_time(&hit_record.p, &(reflected + self.fuzz * random_in_unit_sphere()), ray.time); 
        Some((attenuation, scattered))
    }
}

pub struct Dielectric {
    pub ir: f64,
}

impl Dielectric {
    pub fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
        let r0 = (1. - ref_idx) / (1. + ref_idx);
        let r0 = r0 * r0;
        r0 + (1. - r0) * (1. - cosine).powf(5.)
    }
}

impl Material for Dielectric {
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord) -> Option<(Color, Ray)> {
        let attenuation = Color::new(1., 1., 1.);
        let refraction_ratio = if hit_record.front_face {
            1.0 / self.ir
        } else {
            self.ir
        };

        let unit_direction = ray.d().normalize();
        let cos_theta = f64::min(hit_record.normal.dot(&-unit_direction), 1.);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let cannnot_refract = refraction_ratio * sin_theta > 1.0;
        let direction = if cannnot_refract
            || Dielectric::reflectance(cos_theta, refraction_ratio) > random_double(0., 1.)
        {
            reflect(&unit_direction, &hit_record.normal)
        } else {
            refract(&unit_direction, &hit_record.normal, refraction_ratio)
        };

        let scattered = Ray::with_time(&hit_record.p, &direction, ray.time);
        Some((attenuation, scattered))
    }
}

pub struct DiffuseLight {
    pub emit: Arc<dyn Texture+Send+Sync>,
}

impl Material for DiffuseLight {
    fn scatter(&self, ray: &Ray, hit_record: &HitRecord) -> Option<(Color, Ray)> {
        None
    }
    
    fn emitted(&self, u: f64, v: f64, p: &Point3) -> Color {
        self.emit.value(u, v, p) 
    }
}