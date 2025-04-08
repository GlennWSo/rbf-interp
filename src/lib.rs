//! A library for multidimensional interpolation.

#![allow(unused)]
use std::fmt::Debug;
use std::{marker::PhantomData, ops::Neg, time::Instant};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use nalgebra as na;
use nalgebra::{DMatrix, DMatrixSlice, DVector, Dynamic, MatrixSlice1x2, SliceStorage, SVD, U1};
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

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Scatter2<B: BasisFunction> {
    phantom_basis: PhantomData<B>,
    centers: Vec2,
    deltas: Vec1,
    // data: Vec3,
}

type Column =
    nalgebra::Matrix<f32, Dynamic, U1, SliceStorage<'static, f32, Dynamic, U1, U1, Dynamic>>;

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

    pub fn create(
        // centers: Vec2,
        centers: &[[f32; 2]],
        vals: &[f32],
        // vals: Vec1,
    ) -> Scatter2<B> {
        let n = vals.len();
        let centers: Vec2 = Vec2::from_fn(n, |r, c| centers[r][c]);
        let vals = Vec1::from_row_slice(vals);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sinplane() -> (Vec<[f32; 2]>, Vec<f32>) {
        let mut coords: Vec<_> = Vec::with_capacity(41);
        let mut vals = Vec::with_capacity(41);
        for c in -20..=20 {
            let x = (c as f32) / 5.0;
            for r in -20..=20 {
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
    fn test_scatter2d() {
        let (coords, vals) = sinplane();
        let scatter = Scatter2::<ThinPlateSpline>::create(&coords, &vals);
        for (c, v) in coords.iter().zip(vals) {
            let h = scatter.eval(c);
            println!("c: {:?}, v: {}, h: {}", c, v, h);
            assert!((v - h).abs() < 0.01);
        }
        // panic!();
    }
}
