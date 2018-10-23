use geometry::Point3f;
use geometry::Vector3f;
use geometry::Normal3f;
pub struct Ray {
    pub o: Point3f,
    pub d: Vector3f,
}

pub struct ShadeRec {
    pub hit_an_object: bool,
    pub local_hit_point: Point3f,
    pub normal: Normal3f,
}

pub trait Hit {
    fn hit(&self, ray: &Ray, tmin: &mut f64, sr: &mut ShadeRec) -> bool;
}
