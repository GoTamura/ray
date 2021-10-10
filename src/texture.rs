use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use image::RgbaImage;
use nalgebra::clamp;

use crate::utils::*;

pub trait Texture {
    fn value(&self, u: f64, v: f64, p: &Point3) -> Color;
}

pub struct SolidColor {
    pub color_value: Color,
}

impl Texture for SolidColor {
    fn value(&self, u: f64, v: f64, p: &Point3) -> Color {
        self.color_value
    }
}

pub struct Checker {
    pub odd: Arc<dyn Texture + Send + Sync>,
    pub even: Arc<dyn Texture + Send + Sync>,
}

impl Texture for Checker {
    fn value(&self, u: f64, v: f64, p: &Point3) -> Color {
        let sines = f64::sin(10. * p.x) * f64::sin(10. * p.y) * f64::sin(10. * p.z);
        if sines < 0. {
            self.odd.value(u, v, p)
        } else {
            self.even.value(u, v, p)
        }
    }
}

pub struct ImageTexture {
    image: RgbaImage,
}

impl ImageTexture {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();

        let path = std::path::Path::new(env!("OUT_DIR")).join("res").join(path);
        let img = image::open(path).unwrap();
        Self {
            image: img.to_rgba8(),
        }
    }
}

impl Texture for ImageTexture {
    fn value(&self, u: f64, v: f64, p: &Point3) -> Color {
        let u = clamp(u, 0., 1.);
        let v = 1.-  clamp(v, 0., 1.);
        
        let x = (u*self.image.width()as f64).floor() as u32;
        let y =(v*self.image.height()as f64).floor() as u32;
        
        let x = if x >= self.image.width() {self.image.width()-1} else {x};
        let y = if y >= self.image.width() {self.image.height()-1} else {y};
        
        let color_scale = 1./ 255.;
        let pixel = self.image.get_pixel(x, y);
        color_scale * Color::new(pixel[0] as f64, pixel[1] as f64, pixel[2] as f64)
    }
}
