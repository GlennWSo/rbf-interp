//! A library for multidimensional interpolation.

#![allow(unused)]
use std::fmt::Debug;
use std::{marker::PhantomData, ops::Neg, time::Instant};

// use nalgebra_sparse::na::{Point3, UnitVector3};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use nalgebra as na;
use nalgebra::{
    DMatrix, DMatrixSlice, DVector, Dynamic, MatrixSlice1x2, Point3, SliceStorage, Unit,
    UnitQuaternion, Vector3, SVD, U1,
};
pub type Scatter<T> = Scatter2<T>;

pub trait BasisFunction: 'static + Clone {
    /// a constant parameter for that may be used by the function
    const R: f32 = 1.0;
    const S: f32 = 1.0 / Self::R;
    fn eval(v: f32) -> f32;
}

/**
Gaussian basis function: φ(r) = exp(−(ε*r).pow(2)) where ε = 1/R0
R0 = R / 1000
The main benefit of this function is that it is compact - it is small at distances about 3·R0 from the center. At
distances equal to 6·R0 or larger this function can be considered zero. As result,
we have to solve linear system with sparse matrix. However, significant drawback
is that we have one parameter to tune - R0. Too small R0 will make model sharp and
non-smooth, and too large one will result in system with ill-conditioned matrix.
Gaussian(f32),
**/
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ThinPlateSpline;
impl BasisFunction for ThinPlateSpline {
    fn eval(v: f32) -> f32 {
        if v < 1e-12 {
            return 0.0;
        }
        v.powi(2) * v.ln()
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Gaussian<const R1000: u32>;
impl<const R1000: u32> BasisFunction for Gaussian<R1000> {
    const R: f32 = R1000 as f32 / 1000_f32;
    const S: f32 = 1.0 / Self::R;
    fn eval(r: f32) -> f32 {
        let cutoff = Self::R * 6.0;
        if r > cutoff {
            return 0.0;
        }
        (Self::S * r).powi(2).neg().exp()
    }
}

pub type Vec3 = na::MatrixMN<f32, na::Dynamic, na::U3>;
pub type Vec2 = na::MatrixMN<f32, na::Dynamic, na::U2>;
pub type Vec1 = na::MatrixMN<f32, na::Dynamic, na::U1>;
pub type Row3 = na::RowVector3<f32>;
pub type Row2 = na::RowVector2<f32>;
pub type XY = na::Matrix1x2<f32>;

pub struct HMapScatter<B: BasisFunction = ThinPlateSpline> {
    pub scatter: Scatter2<B>,
    pub quat: na::UnitQuaternion<f32>,
    pub axis: Normal,
}

pub type Normal = Unit<Vector3<f32>>;

pub type Points = Vec<Point3<f32>>;

impl<B: BasisFunction> HMapScatter<B> {
    pub fn create(h_axis: Normal, mut points: Points) -> Self {
        let z = Vector3::new(0.0, 0.0, 1.0);
        let quat =
            UnitQuaternion::rotation_between(&h_axis, &z).unwrap_or(UnitQuaternion::identity());
        for point in points.iter_mut() {
            *point = quat.transform_point(&point);
        }
        let n_rows = points.len();
        let xy = Vec2::from_fn(n_rows, |r, c| match c {
            0 => points[r].x,
            1 => points[r].y,
            _ => unreachable!(),
        });
        let vals = Vec1::from_fn(n_rows, |r, _c| points[r].z);
        // quat.transform_point(centers.row(0).into());

        Self {
            scatter: Scatter2::create(xy, vals),
            quat,
            axis: h_axis,
        }
    }
    pub fn new(h_axis: [f32; 3], coords: &[[f32; 3]]) -> Self {
        let h_axis = Unit::new_normalize(Vector3::from(h_axis));
        let z = Vector3::new(0.0, 0.0, 1.0);
        let quat =
            UnitQuaternion::rotation_between(&h_axis, &z).unwrap_or(UnitQuaternion::identity());
        let n_rows = coords.len();

        let points: Vec<_> = coords.iter().map(|v| Point3::from(*v)).collect();
        // let coords = Vec2::from(centers);
        Self::create(h_axis, points)
    }
    pub fn eval_points(&self, points: &mut [Point3<f32>]) {
        let invquat = self.quat.inverse();
        for point in points {
            let mut local = self.quat.transform_point(point);
            local.z = self.scatter.eval(&[local.x, local.y]);
            *point = invquat.transform_point(&local);
        }
    }
    pub fn evals(&self, input: &mut [[f32; 3]]) {
        let mut points: Points = input.iter().map(|v| (*v).into()).collect();
        self.eval_points(&mut points);
        for (output, point) in input.iter_mut().zip(points.iter()) {
            *output = [point.x, point.y, point.z];
        }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Scatter2<B: BasisFunction> {
    phantom_basis: PhantomData<B>,
    centers: Vec2,
    deltas: Vec1,
    // data: Vec3,
}

impl<B: BasisFunction> Scatter2<B> {
    pub fn eval(&self, coord: &[f32; 2]) -> f32 {
        let centers = &self.centers;
        let n = self.deltas.len();
        let coord = MatrixSlice1x2::from_slice(coord);

        // TODO are you parrallel?
        let basis = DVector::from_fn(n, |row, _c| {
            let center = centers.row(row);
            let delta = coord - center;
            let r = (delta).norm();
            B::eval(r)
        });

        basis.dot(&self.deltas)
    }

    fn create(centers: Vec2, vals: Vec1) -> Scatter2<B> {
        let n = vals.len();
        let mat = DMatrix::from_fn(n, n, |r, c| {
            B::eval((centers.row(r) - centers.row(c)).norm())
        });
        let n_non_zero = mat.iter().filter(|e| **e != 0.0).count() as f32;
        let len = mat.len() as f32;

        let rbf_solve_time = Instant::now();
        let svd = SVD::new(mat.transpose(), true, true);
        let eps = 1e-12;
        // let vals = data.column(2);
        let deltas = svd.solve(&vals, eps).unwrap();
        // data.set_column(2, &deltas);

        Scatter2 {
            phantom_basis: PhantomData,
            centers,
            deltas,
        }
    }

    pub fn new(
        // centers: Vec2,
        centers: &[[f32; 2]],
        vals: &[f32],
        // vals: Vec1,
    ) -> Scatter2<B> {
        let n = vals.len();
        let centers: Vec2 = Vec2::from_fn(n, |r, c| centers[r][c]);
        let vals = Vec1::from_row_slice(vals);
        Self::create(centers, vals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sinplane() -> (Vec<[f32; 2]>, Vec<f32>) {
        let mut coords: Vec<_> = Vec::with_capacity(21);
        let mut vals = Vec::with_capacity(21);
        for c in -10..=10 {
            let x = (c as f32) / 3.0;
            for r in -10..=10 {
                let y = (r as f32) / 5.0;
                let r = (x.powi(2) + y.powi(2)).sqrt();
                let z = r.cos();
                coords.push([x, y]);
                vals.push(z);
            }
        }
        (coords, vals)
    }

    #[test]
    fn test_hmap_points() {
        let coords: Vec<_> = {
            let (coords, vals) = sinplane();
            coords
                .into_iter()
                .zip(vals.into_iter())
                .map(|(c, v)| [c[0], c[1], v])
                .collect()
        };

        let mut points: Points = coords.iter().map(|v| (*v).into()).collect();
        let input = points.clone();
        let normal = Vector3::new(0.0, 0.5, 1.0);
        let hmap =
            HMapScatter::<ThinPlateSpline>::create(Unit::new_normalize(normal), points.clone());

        // offset along axis
        for point in points.iter_mut() {
            point.x += hmap.axis.x * 3.0;
            point.y += hmap.axis.y * 3.0;
            point.z += hmap.axis.z * 3.0;
        }

        hmap.eval_points(&mut points);

        for (input, output) in input.iter().zip(points) {
            let diff = input - output;
            println!("diff: {}, {}, {}", diff.x, diff.y, diff.z);
            let tol = 0.001;
            let adiff = diff.norm();
            assert!(adiff < tol);
        }

        // let mut out = coords.clone();
    }
    #[test]
    fn test_hmap() {
        let coords: Vec<_> = {
            let (coords, vals) = sinplane();
            coords
                .into_iter()
                .zip(vals.into_iter())
                .map(|(c, v)| [c[0], c[1], v])
                .collect()
        };

        let points: Points = coords.iter().map(|v| (*v).into()).collect();
        let normal = Vector3::new(0.0, 0.5, 1.0);
        let hmap = HMapScatter::<ThinPlateSpline>::create(Unit::new_normalize(normal), points);

        // let hmap = HMapScatter::<ThinPlateSpline>::new([0.0, 0.5, 1.0], &coords);
        let mut out = coords.clone();
        hmap.evals(&mut out);
        for (input, output) in coords.iter().zip(out) {
            println!("inout: {input:?}, {output:?}");
            let tol = 0.001;
            let adiff = (input[0] - output[0]).abs();
            assert!(adiff < tol);
            let adiff = (input[1] - output[1]).abs();
            assert!(adiff < tol);
            let adiff = (input[2] - output[2]).abs();
            assert!(adiff < tol);
        }
        dbg!(hmap.axis);
    }

    #[test]
    fn test_scatter2d() {
        let (coords, vals) = sinplane();
        let scatter = Scatter2::<ThinPlateSpline>::new(&coords, &vals);
        for (c, v) in coords.iter().zip(vals) {
            let h = scatter.eval(c);
            // println!("c: {:?}, v: {}, h: {}", c, v, h);
            assert!((v - h).abs() < 0.001);
        }
    }
}
