use geometry::point::lerp;
use geometry::point::*;
use geometry::vector::*;
use std::f64;
use std::ops::*;
use std::convert::Into;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Bounds3<T> {
    pub p_min: Point3<T>,
    pub p_max: Point3<T>,
}

pub trait New<T> {
    type Output;
    fn new(T) -> Self::Output;
}

impl<T> Bounds3<T>
    where T: PartialOrd + Sub<Output=T> + Mul<Output=T> + Add<Output=T> + Div<Output=T> + DivAssign + 
    Into<f64> + Copy  {
    pub fn corner(&self, corner: u32) -> Point3<T> {
        Point3 {
            x: self[(corner & 1)].x,
            y: self[(corner & 2) >> 1].y,
            z: self[(corner & 4) >> 2].z,
        }
    }

    pub fn diagonal(&self) -> Vector3<T> {
        self.p_max - self.p_min
    }

    pub fn surface_area(&self) -> f64 {
        let d = self.diagonal();
        (d.x * d.y + d.x * d.z + d.y * d.z).into() * 2.
    }

    pub fn volume(&self) -> T {
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

    pub fn offset(&self, p: &Point3<T>) -> Vector3<T> {
        let mut o = *p - self.p_min;
        if self.p_max.x > self.p_min.x { o.x /= self.p_max.x - self.p_min.x };
        if self.p_max.y > self.p_min.y { o.y /= self.p_max.y - self.p_min.y };
        if self.p_max.z > self.p_min.z { o.z /= self.p_max.z - self.p_min.z };
        o
    }

    pub fn bounding_sphere(&self) -> (Point3f, f64) {
        let center = (self.p_min + self.p_max) / 2.;
        let radius =  if Self::inside(&center, &self) { Self::distance(&center, &self.p_max)} 
            else {0.};
        (center, radius)
    }

    pub fn inside(p: &Point3<T>, b: &Bounds3<T>) -> bool {
        p.x >= b.p_min.x && p.x <= b.p_max.x && p.y >= b.p_min.y &&
            p.y <= b.p_max.y && p.z >= b.p_min.z && p.z <= b.p_max.z
    }

    pub fn distance(p1: &Point3<T>, p2: &Point3<T>) -> f64 {
        (*p1 - *p2).length()
    }

}

impl<T> Index<u32> for Bounds3<T> {
    type Output = Point3<T>;

    fn index(&self, i: u32) -> &Point3<T> {
        match i {
            0 => &self.p_min,
            1 => &self.p_max,
            _ => &self.p_min,
        }
    }
}

impl Default for Bounds3<f64> {
    fn default() -> Bounds3<f64> {
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
}


impl<'a, T: Copy> New<&'a Point3<T>> for Bounds3<T> {
    type Output = Bounds3<T>;
    fn new(p: &Point3<T>) -> Bounds3<T> {
        Bounds3 {
            p_min: *p,
            p_max: *p,
        }
    }
}

impl<'a, 'b, T: Copy> New<(&'a Point3<T>, &'b Point3<T>)> for Bounds3<T> {
    type Output = Bounds3<T>;
    fn new(p: (&Point3<T>, &Point3<T>)) -> Bounds3<T> {
        Bounds3 {
            p_min: *p.0,
            p_max: *p.1,
        }
    }
}


