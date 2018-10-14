use geometry::point::lerp;
use geometry::point::Point3f;
use geometry::vector::Vector3f;
use std::f64;
use std::ops::*;

#[derive(Debug, PartialEq, Clone)]
pub struct Bounds3 {
    pub p_min: Point3f,
    pub p_max: Point3f,
}

pub trait New<T> {
    fn new(T) -> Bounds3;
}

impl Bounds3 {
    /*
    pub fn new() -> Bounds3 {
        let min_num = f64::MIN;
        let max_num = f64::MAX;
        Bounds3 {
            p_min: Point3f {
                x: min_num,
                y: min_num,
                z: min_num,
            },
            p_max: Point3f {
                x: max_num,
                y: max_num,
                z: max_num,
            },
        }
    }
    */

    pub fn corner(&self, corner: u32) -> Point3f {
        Point3f {
            x: self[(corner & 1)].x,
            y: self[(corner & 2) >> 1].y,
            z: self[(corner & 4) >> 2].z,
        }
    }

    pub fn diagonal(&self) -> Vector3f {
        self.p_max - self.p_min
    }

    pub fn surface_area(&self) -> f64 {
        let d = self.diagonal();
        2. * (d.x * d.y + d.x * d.z + d.y * d.z)
    }

    pub fn volume(&self) -> f64 {
        let d = self.diagonal();
        d.x * d.y * d.z
    }

    pub fn maximum_extent(&self) -> i32 {
        let d = self.diagonal();
        if d.x > d.y && d.x > d.z {
            0
        } else if d.y > d.z {
            1
        } else {
            2
        }
    }

    //pub fn lerp(&self, t: &Point3f) -> Point3f {
    //    Point3f {
    //        x: lerp(t.x, self.p_min.x, self.p_max.x),
    //        y: lerp(t.y, self.p_min.y, self.p_max.y),
    //        z: lerp(t.z, self.p_min.z, self.p_max.z),
    //    }
    //}

    pub fn offset(&self, p: &Point3f) -> Vector3f {
        let mut o = *p - self.p_min;
        if self.p_max.x > self.p_min.x { o.x /= self.p_max.x - self.p_min.x };
        if self.p_max.y > self.p_min.y { o.y /= self.p_max.y - self.p_min.y };
        if self.p_max.z > self.p_min.z { o.z /= self.p_max.z - self.p_min.z };
        o
    }

    //pub fn bounding_sphere(&self) -> (Point3f, f64) {
    //    let center = (self.p_min + self.p_max) / 2.;
    //    let radius = self.inside(&center, &self) ? distance(&center, p_max) : 0;
    //    (center, radius)
    //}

    pub fn inside(p: &Point3f, b: &Bounds3) -> bool {
        p.x >= b.p_min.x && p.x <= b.p_max.x && p.y >= b.p_min.y &&
            p.y <= b.p_max.y && p.z >= b.p_min.z && p.z <= b.p_max.z
    }

    pub fn distance(p1: &Point3f, p2: &Point3f) -> f64 {
        (*p1 - *p2).length()
    }

}

impl Index<u32> for Bounds3 {
    type Output = Point3f;

    fn index(&self, i: u32) -> &Point3f {
        match i {
            0 => &self.p_min,
            1 => &self.p_max,
            _ => &self.p_min,
        }
    }
}

impl<'a> New<&'a Point3f> for Bounds3 {
    fn new(p: &Point3f) -> Bounds3 {
        Bounds3 {
            p_min: *p,
            p_max: *p,
        }
    }
}

impl<'a, 'b> New<(&'a Point3f, &'b Point3f)> for Bounds3 {
    fn new(p: (&Point3f, &Point3f)) -> Bounds3 {
        Bounds3 {
            p_min: *p.0,
            p_max: *p.1,
        }
    }
}


