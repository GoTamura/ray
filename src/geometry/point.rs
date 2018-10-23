use geometry::vector::*;
use std::ops::*;

pub type Point3f = Point3<f64>;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Point3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Add<Output=T>> Add<Point3<T>> for Point3<T> {
    type Output = Point3<T>;

    fn add(self, other: Point3<T>) -> Point3<T> {
        Point3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<T: Add<Output=T>> Add<Vector3<T>> for Point3<T> {
    type Output = Point3<T>;

    fn add(self, other: Vector3<T>) -> Point3<T> {
        Point3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<T: Add<Output=T> + Copy> AddAssign for Point3<T> {
    fn add_assign(&mut self, other: Point3<T>) {
        *self = *self + other;
    }
}

impl<T: Sub<Output=T>> Sub<Point3<T>> for Point3<T> {
    type Output = Vector3<T>;

    fn sub(self, other: Point3<T>) -> Vector3<T> {
        Vector3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<T: Sub<Output=T>> Sub<Vector3<T>> for Point3<T> {
    type Output = Point3<T>;

    fn sub(self, other: Vector3<T>) -> Point3<T> {
        Point3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<T: Sub<Output=T> + Copy> SubAssign<Vector3<T>> for Point3<T> {
    fn sub_assign(&mut self, other: Vector3<T>) {
        *self = *self - other;
    }
}

impl<T: Div<Output=T> + Copy> Div<T> for Point3<T> {
    type Output = Point3<T>;

    fn div(self, rhs: T) -> Point3<T> {
        Point3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl<T: Div<Output=T> + Copy> DivAssign<T> for Point3<T> {
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

impl<T: Mul<Output=T> + Copy> Mul<T> for Point3<T> {
    type Output = Point3<T>;

    fn mul(self, rhs: T) -> Point3<T> {
        Point3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<Point3f> for f64 {
    type Output = Point3f;

    fn mul(self, rhs: Point3f) -> Point3f {
        Point3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

impl<T: Mul<Output=T> + Copy> MulAssign<T> for Point3<T> {
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T> Index<u32> for Point3<T> {
    type Output = T;

    fn index(&self, i: u32) -> &T {
        match i {
            0 => &self.x,
            1 => &self.y,
            _ => &self.z,
        }
    }
}

pub fn lerp(t: f64, p0: &Point3f,  p1: &Point3f) -> Point3f {
    (1. - t) * *p0 + t * *p1
}

#[cfg(test)]
mod test {
    #[test]
    fn test_add_assign() {
        println!("hello");
    }
}
