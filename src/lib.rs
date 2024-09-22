//! A library for multidimensional interpolation.

use std::{marker::PhantomData, ops::Neg, time::Instant};

pub use nalgebra as na;
use nalgebra::{DMatrix, DMatrixSlice, DVector, Dynamic, SliceStorage, SVD, U1};

#[derive(Clone)]
pub enum BasisKind {
    /**
    aka thinplate spline
    φ(r) = ln(r) * r.pow(N)
    pros: do not need tuning
    cons: not compact, ie is allways non zero. >> dense matrix >> less performance
    **/
    PolyHarmonic,
    /**
    Gaussian basis function: φ(r) = exp(−(ε*r).pow(2)) where ε = 1/R0
    The main benefit of this function is that it is compact - it is small at distances about 3·R0 from the center. At
    distances equal to 6·R0 or larger this function can be considered zero. As result,
    we have to solve linear system with sparse matrix. However, significant drawback
    is that we have one parameter to tune - R0. Too small R0 will make model sharp and
    non-smooth, and too large one will result in system with ill-conditioned matrix.
    Gaussian(f32),
    **/
    Gaussian,
}

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
#[derive(Clone)]
pub struct ThinPlateSpline;
impl BasisFunction for ThinPlateSpline {
    fn eval(v: f32) -> f32 {
        if v < 1e-12 {
            return 0.0;
        }
        v.powi(2) * v.ln()
    }
}

#[derive(Clone)]
pub struct Gaussian<const R1000: u32>;
impl<const R1000: u32> BasisFunction for Gaussian<R1000> {
    const R: f32 = R1000 as f32 / 1000 as f32;
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
pub type XY = na::Matrix1x2<f32>;

#[derive(Clone, Debug)]
pub struct Scatter<B: BasisFunction> {
    /// define basis function trough generics for static performance
    phantom_basis: PhantomData<B>,
    // TODO(explore): use matrix & slicing instead (fewer allocs).
    // An array of n vectors each of size m.
    // centers: Vec2,
    // An m x n' matrix, where n' is the number of basis functions (including polynomial),
    // and m is the number of coords.
    // deltas: Vec1,
    data: Vec3,
}

type column =
    nalgebra::Matrix<f32, Dynamic, U1, SliceStorage<'static, f32, Dynamic, U1, U1, Dynamic>>;

impl<B: BasisFunction> Scatter<B> {
    pub fn eval(&self, coord: XY) -> f32 {
        let centers = self.data.columns_range(..2);
        let basis = DVector::from_fn(self.data.nrows(), |row, _c| {
            let center = centers.row(row);
            let delta = coord - center;
            let r = (delta).norm();
            B::eval(r)
            // linear component
        });
        let deltas = self.data.column(2);
        let res = basis.dot(&deltas);
        res
    }

    // The order for the polynomial part, meaning terms up to (order - 1) are included.
    // This usage is consistent with Wilna du Toit's masters thesis "Radial Basis
    // Function Interpolation"
    pub fn create(
        // centers: Vec2,
        // vals: Vec1,
        samples: Vec3,
        // order: usize,
    ) -> Scatter<B> {
        dbg!("creating rbf");
        let mut data = samples;
        let n = dbg!(data.nrows());
        // n x m matrix. There's probably a better way to do this, ah well.
        // let mut vals = DMatrix::from_columns(&vals).transpose();
        // We translate the system to center the mean at the origin so that when
        // the system is degenerate, the pseudoinverse below minimizes the linear
        // coefficients.
        // let means: Vec<_> = Vec::new();
        // let vals_mat = DMatrix::f(&[vals]);
        let centers = data.columns_range(..2);
        let mat = DMatrix::from_fn(n, n, |r, c| {
            B::eval((centers.row(r) - centers.row(c)).norm())
        });
        let n_non_zero = mat.iter().filter(|e| **e != 0.0).count() as f32;
        let len = dbg!(mat.len()) as f32;
        dbg!(n_non_zero / len);
        // let coo = na::sp

        dbg!("solving rbf deltas");
        let rbf_solve_time = Instant::now();
        let svd = SVD::new(mat.transpose(), true, true);
        let eps = 1e-12;
        let vals = data.column(2);
        let deltas = svd.solve(&vals, eps).unwrap();
        data.set_column(2, &deltas);
        dbg!(rbf_solve_time.elapsed());
        dbg!("rbf created");
        let scatter = Scatter {
            phantom_basis: PhantomData,
            data,
        };

        scatter
    }
}
