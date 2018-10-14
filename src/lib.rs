pub mod geometry;

#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        use super::geometry::*;
        let a = Vector3f { x: 10., y: 20., z: 35.};
        let b = Vector3f { x: 10., y: 25., z: 30.};
        let c = Vector3f { x: 10., y: 20., z: 30.};
        let d = a+b;
        assert_eq!(Vector3f { x: 20., y: 45., z: 65. }, d );
        let e = c * 3.0;
        assert_eq!(Vector3f { x: 30., y: 60., z: 90. }, e );
        let f = e.cross(d);
        assert_eq!(Vector3f { x: -150., y: -150., z: 150. }, f );
        assert_eq!(-150., f[0]);
        assert_eq!(-150., f[1]);
        assert_eq!(150., f[2]);
        assert_eq!(150., f[3]);
        assert!(Vector3f { x: 1., y: 2., z: 3. } == Vector3f { x: 1., y: 2., z: 3. });
        assert!(Vector3f { x: 1., y: 3., z: 3. } != Vector3f { x: 1., y: 2., z: 3. });

        let p1 = Point3f {x:1., y:2., z:3.};
        let p2 = Point3f {x:10., y:-10., z:5.};

        //let b1 = Bounds3::new();
        let b2 = Bounds3::new(&p1);
        let b3 = Bounds3::new((&p1, &p2));
    }

}
