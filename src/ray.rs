use crate::utils::*;
pub struct Ray {
    o: Point3,
    d: Vec3,
    pub time: f64,
    pub t_max: f64,
}
impl Ray {
    pub fn new(o: &Point3, d: &Vec3) -> Self {
        Self {
            o: *o,
            d: *d,
            time: 0.,
            t_max: std::f64::INFINITY,
        }
    }

    pub fn with_time(o: &Point3, d: &Vec3, time: f64) -> Self {
        Self {
            o: *o,
            d: *d,
            time,
            t_max: std::f64::INFINITY,
        }
    }
    pub fn at(&self, t: f64) -> Point3 {
        self.o + self.d * t
    }
    
    pub fn o(&self) -> Point3 {
        self.o
    }

    pub fn d(&self) -> Vec3 {
        self.d
    }
}