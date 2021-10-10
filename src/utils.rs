use nalgebra::clamp;

pub type Vec3 = nalgebra::Vector3<f64>;
pub type Point3 = nalgebra::Point3<f64>;
pub type Point2 = nalgebra::Point2<f64>;
pub type Color = Vec3;

pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * std::f64::consts::PI / 180.0
}

pub fn random_double(min: f64, max: f64) -> f64 {
    let mut rng = rand::thread_rng();
    use rand::distributions::{Distribution, Uniform};
    let between = Uniform::from(min..max);
    between.sample(&mut rng)
}

pub fn random_vec3(min: f64, max: f64) -> Vec3 {
    let mut rng = rand::thread_rng();
    use rand::distributions::{Distribution, Uniform};
    let between = Uniform::from(min..max);
    Vec3::new(
        between.sample(&mut rng),
        between.sample(&mut rng),
        between.sample(&mut rng),
    )
}

pub fn random_in_unit_disk() -> Vec3 {
    loop {
        let p = Vec3::new(random_double(-1., 1.), random_double(-1., 1.), 0.);
        if p.magnitude_squared() >= 1. {
            continue;
        }
        return p;
    }
}

pub fn random_in_unit_sphere() -> Vec3 {
    loop {
        let p = random_vec3(-1., 1.);
        if p.magnitude_squared() >= 1. {
            continue;
        }
        return p;
    }
}

pub fn random_unit_vector() -> Vec3 {
    random_in_unit_sphere().normalize()
}

pub fn random_in_hemisphere(normal: &Vec3) -> Vec3 {
    let in_unit_sphere = random_in_unit_sphere();
    if in_unit_sphere.dot(normal) > 0.0 {
        in_unit_sphere
    } else {
        -in_unit_sphere
    }
}

pub fn near_zero(v: &Vec3) -> bool {
    const S: f64 = 1e-8;
    (v.x.abs() < S) && (v.y.abs() < S) && (v.z.abs() < S)
}

pub fn reflect(v: &Vec3, n: &Vec3) -> Vec3 {
    v - 2. * v.dot(n) * n
}

pub fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = f64::min(n.dot(&-uv), 1.0);
    let r_out_perp = etai_over_etat * (uv + cos_theta * n);
    let r_out_parallel = -(1.0 - r_out_perp.magnitude_squared()).abs().sqrt() * n;
    r_out_perp + r_out_parallel
}

pub fn color_to_rgb(color: &Color, samples_per_pixel: u32) -> image::Rgb<u8> {
    let r = color.x;
    let g = color.y;
    let b = color.z;
    let scale = 1.0 / samples_per_pixel as f64;
    let gamma = 2.2;
    let r = (r * scale).powf(1. / gamma);
    let g = (g * scale).powf(1. / gamma);
    let b = (b * scale).powf(1. / gamma);
    image::Rgb([
        (256. * clamp(r, 0.0, 0.999)) as u8,
        (256. * clamp(g, 0.0, 0.999)) as u8,
        (256. * clamp(b, 0.0, 0.999)) as u8,
    ])
}
