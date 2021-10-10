use anyhow::{Context, Error, Result};
use image::ImageBuffer;
use ray::texture::{Checker, ImageTexture, SolidColor};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use ray::camera::SingleLensCamera;
use ray::material::{Dielectric, DiffuseLight, Lambertian, Metal};
use ray::ray::Ray;
use ray::shape::{Hittable, HittableList, MovingSphere, Sphere};
use ray::utils::*;
use rayon::prelude::*;

fn ray_color<T: Hittable>(ray: &Ray, world: &T, depth: i32) -> Color {
    if depth <= 0 {
        return Color::new(0., 0., 0.);
    }
    if let Some(hit_record) = world.hit(ray, 0.001, f64::INFINITY) {
        let emitted = hit_record
            .material
            .emitted(hit_record.u, hit_record.v, &hit_record.p);

        if let Some((attenuation, scattered)) = hit_record.material.scatter(ray, &hit_record) {
            emitted + attenuation.component_mul(&ray_color(&scattered, world, depth - 1))
        } else {
            emitted
        }
    } else {
        // Color::new(0., 0., 0.)
        let unit_direction = ray.d().normalize();
        let t = 0.5 * (unit_direction.y + 1.);
        (1.0 - t) * Color::new(1., 1., 1.) + t * Color::new(0.5, 0.7, 1.0)
    }
}

fn random_scene() -> HittableList {
    let ground_material = Arc::new(Lambertian {
        albedo: Arc::new(SolidColor {
            color_value: Color::new(0.5, 0.5, 0.5),
        }),
    });
    let material1 = Arc::new(Dielectric { ir: 1.5 });
    let material2 = Arc::new(Lambertian {
        albedo: Arc::new(SolidColor {
            color_value: Color::new(0.3, 0.2, 0.1),
        }),
    });
    let material3 = Arc::new(Metal {
        albedo: Color::new(0.7, 0.6, 0.5),
        fuzz: 0.0,
    });
    let material4 = Arc::new(DiffuseLight {
        emit: Arc::new(SolidColor {
            color_value: Color::new(0.8, 0.8, 0.85),
        }),
    });
    let checker = Arc::new(Lambertian {
        albedo: Arc::new(Checker {
            odd: Arc::new(SolidColor {
                color_value: Color::new(0.2, 0.3, 0.1),
            }),
            even: Arc::new(SolidColor {
                color_value: Color::new(0.9, 0.9, 0.9),
            }),
        }),
    });

    let image = Arc::new(Lambertian {
        albedo: Arc::new(ImageTexture::new(
            // "one_weekend.png"
            "vlcsnap-2016-07-08-19h13m16s219.png",
        )),
    });

    let mut world: HittableList = vec![
        Arc::new(Sphere {
            center: Point3::new(0., -1000., 0.),
            radius: 1000.,
            material: checker,
        }),
        Arc::new(Sphere {
            center: Point3::new(0., 1., 0.),
            radius: 1.0,
            material: material1,
        }),
        Arc::new(Sphere {
            center: Point3::new(-4., 1., 0.),
            radius: 1.0,
            material: material2,
        }),
        Arc::new(Sphere {
            center: Point3::new(4., 1., 0.),
            radius: 1.0,
            material: image,
        }),
    ];

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = random_double(0., 1.);
            let center = Point3::new(
                a as f64 + 0.9 * random_double(0., 1.),
                0.2,
                b as f64 + 0.9 * random_double(0., 1.),
            );

            if (center - Point3::new(4., 0.2, 0.)).magnitude() > 0.9 {
                if choose_mat < 0.8 {
                    let albedo = Arc::new(SolidColor {
                        color_value: random_vec3(0., 1.).component_mul(&random_vec3(0., 1.)),
                    });
                    let material = Arc::new(Lambertian { albedo });
                    // let center2 = center + Vec3::new(0., random_double(0., 0.5), 0.);
                    // world.push(Arc::new(MovingSphere {
                    //     center0: center,
                    //     center1: center2,
                    //     radius: 0.2,
                    //     material,
                    //     time0: 0.0,
                    //     time1: 1.0,
                    // }));
                    world.push(Arc::new(Sphere {
                        center,
                        radius: 0.2,
                        material,
                    }));
                } else if choose_mat < 0.95 {
                    let albedo = random_vec3(0.5, 1.);
                    let fuzz = random_double(0., 0.5);
                    let material = Arc::new(Metal { albedo, fuzz });
                    world.push(Arc::new(Sphere {
                        center,
                        radius: 0.2,
                        material,
                    }));
                } else {
                    let material = Arc::new(Dielectric { ir: 1.5 });
                    world.push(Arc::new(Sphere {
                        center,
                        radius: 0.2,
                        material,
                    }));
                };
            }
        }
    }
    world
}

fn main() -> Result<(), Error> {
    let term = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(&term))?;
    const ASPECT_RATIO: f64 = 3. / 2.;
    const IMAGE_WIDTH: u32 = 1200;
    const IMAGE_HEIGHT: u32 = (IMAGE_WIDTH as f64 / ASPECT_RATIO) as u32;
    const SAMPLES_PER_PIXEL: u32 = 50;
    const MAX_DEPTH: i32 = 50;

    use std::thread::sleep;
    use std::time::{Duration, Instant};
    println!("Let's ray tracing");
    let start = Instant::now();

    let camera = {
        let lookfrom = Point3::new(13., 2., 3.);
        let lookat = Point3::new(0., 0., 0.);
        let vup = Vec3::new(0., 1., 0.);
        let dist_to_focus = 10.0;
        let aperture = 0.1;
        SingleLensCamera::new(
            &lookfrom,
            &lookat,
            &vup,
            20.,
            ASPECT_RATIO,
            aperture,
            dist_to_focus,
            0.,
            1.,
        )
    };

    let mut imgbuf = image::ImageBuffer::new(IMAGE_WIDTH, IMAGE_HEIGHT);

    let world = random_scene();

    render1(
        &mut imgbuf,
        &camera,
        &world,
        SAMPLES_PER_PIXEL,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        MAX_DEPTH,
        term,
    );
    imgbuf.save("test.png").unwrap();
    let end = start.elapsed();
    println!("{}.{:03}", end.as_secs(), end.subsec_nanos() / 1_000_000);
    Ok(())
}

fn render1<T: Hittable + Send + Sync>(
    imgbuf: &mut image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    camera: &SingleLensCamera,
    world: &T,
    samples_per_pixel: u32,
    image_height: u32,
    image_width: u32,
    max_depth: i32,
    term: Arc<AtomicBool>,
) {
    let bar = {
        let mut bar = ProgressBar::new(image_width as u64 * image_height as u64);
        bar.set_style(ProgressStyle::default_bar().template("{spinner}{percent:>3}% {bar:100.cyan/blue} {pos:>7}/{len:7} [{elapsed_precise}, {per_sec}, {eta_precise}] {msg}"));
        // .progress_chars("##-"));
        bar
    };
    imgbuf
        .enumerate_pixels_mut()
        .collect::<Vec<(u32, u32, &mut image::Rgb<u8>)>>()
        .par_iter_mut()
        .progress_with(bar)
        .try_for_each(|(x, y, pixel)| {
            let color = draw1(
                *x,
                *y,
                &camera,
                world,
                samples_per_pixel,
                image_height,
                image_width,
                max_depth,
            );
            **pixel = color_to_rgb(&color, samples_per_pixel);
            (!term.load(Ordering::Relaxed)).then(|| ()).ok_or(())
        });
}

fn draw1<T: Hittable>(
    x: u32,
    y: u32,
    camera: &SingleLensCamera,
    world: &T,
    samples_per_pixel: u32,
    image_height: u32,
    image_width: u32,
    max_depth: i32,
) -> Color {
    let mut rng = rand::thread_rng();
    use rand::distributions::{Distribution, Uniform};
    let between = Uniform::from(0.0..1.0);
    let mut color = Color::new(0., 0., 0.);
    for _ in 0..samples_per_pixel {
        let y = image_height - y;
        let x = x;
        let u = (x as f64 + between.sample(&mut rng)) / (image_width - 1) as f64;
        let v = (y as f64 + between.sample(&mut rng)) / (image_height - 1) as f64;
        let ray = camera.generate_ray(u, v);
        color += ray_color(&ray, world, max_depth);
    }
    color
}

fn render2<T: Hittable + Send + Sync>(
    imgbuf: &mut image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    camera: &SingleLensCamera,
    world: &T,
    samples_per_pixel: u32,
    image_height: u32,
    image_width: u32,
    max_depth: i32,
    term: Arc<AtomicBool>,
) {
    let bar = {
        let bar = ProgressBar::new(samples_per_pixel as u64);
        bar.set_style(ProgressStyle::default_bar().template("{spinner}{percent:>3}% {bar:100.cyan/blue} {pos:>7}/{len:7} [{elapsed_precise}, {per_sec}] {msg}"));
        // .progress_chars("##-"));
        bar
    };

    let colors = Arc::new(vec![
        vec![
            Arc::new(Mutex::new(Color::new(0., 0., 0.)));
            image_width as usize + 1
        ];
        image_height as usize + 1
    ]);
    for i in 0..samples_per_pixel {
        imgbuf
            .enumerate_pixels_mut()
            .collect::<Vec<(u32, u32, &mut image::Rgb<u8>)>>()
            .par_iter_mut()
            .for_each(|(x, y, pixel)| {
                let mut rng = rand::thread_rng();
                use rand::distributions::{Distribution, Uniform};
                let between = Uniform::from(0.0..1.0);
                let y = image_height - *y;
                let x = *x;
                let u = (x as f64 + between.sample(&mut rng)) / (image_width - 1) as f64;
                let v = (y as f64 + between.sample(&mut rng)) / (image_height - 1) as f64;
                let ray = camera.generate_ray(u, v);
                let colors = colors.clone();
                let mut color = colors[y as usize][x as usize].lock().unwrap();
                *color += ray_color(&ray, world, max_depth);
                **pixel = color_to_rgb(&*color, i + 1);
            });
        bar.inc(1);
        if term.load(Ordering::Relaxed) {
            break;
        }
    }
    bar.finish();
}
