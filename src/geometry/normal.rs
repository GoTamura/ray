use std::ops::*;

pub type Normal3f = Normal3<f64>;

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Normal3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Add<Output=T>> Add for Normal3<T> {
    type Output = Normal3<T>;

    fn add(self, other: Normal3<T>) -> Normal3<T> {
        Normal3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<T: Add<Output=T> + Copy> AddAssign for Normal3<T> {
    fn add_assign(&mut self, other: Normal3<T>) {
        *self = *self + other;
    }
}

impl<T: Sub<Output=T>> Sub for Normal3<T> {
    type Output = Normal3<T>;

    fn sub(self, other: Normal3<T>) -> Normal3<T> {
        Normal3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<T: Sub<Output=T> + Copy> SubAssign for Normal3<T> {
    fn sub_assign(&mut self, other: Normal3<T>) {
        *self = *self - other;
    }
}

impl<T: Div<Output=T> + Copy> Div<T> for Normal3<T> {
    type Output = Normal3<T>;

    fn div(self, rhs: T) -> Normal3<T> {
        Normal3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl<T: Div<Output=T> + Copy> DivAssign<T> for Normal3<T> {
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

impl<T: Mul<Output=T> + Copy> Mul<T> for Normal3<T> {
    type Output = Normal3<T>;

    fn mul(self, rhs: T) -> Normal3<T> {
        Normal3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T: Mul<Output=T> + Copy> MulAssign<T> for Normal3<T> {
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl Normal3f {
    pub fn length_squared(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }
}

impl<T> Index<u32> for Normal3<T> {
    type Output = T;

    fn index(&self, i: u32) -> &T {
        match i {
            0 => &self.x,
            1 => &self.y,
            _ => &self.z,
        }
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_add_assign() {
        println!("hello");
    }
}
