//! A library for multidimensional interpolation.

use std::{marker::PhantomData, ops::Neg};

use nalgebra::{self as na, DMatrix, DMatrixSlice, DVector, SVD};

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

pub trait BasisFunction {
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
pub struct ThinPlateSpline;
impl BasisFunction for ThinPlateSpline {
    fn eval(v: f32) -> f32 {
        if v < 1e-12 {
            return 0.0;
        }
        v.powi(2) * v.ln()
    }
}

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
pub type XY = na::Matrix1x2<f32>;

#[derive(Clone)]
pub struct Scatter<B: BasisFunction> {
    /// define basis function trough generics for static performance
    phantom_basis: PhantomData<B>,
    // TODO(explore): use matrix & slicing instead (fewer allocs).
    // An array of n vectors each of size m.
    centers: Vec2,
    // An m x n' matrix, where n' is the number of basis functions (including polynomial),
    // and m is the number of coords.
    deltas: Vec1,
}

impl<B: BasisFunction> Scatter<B> {
    pub fn eval(&self, coord: XY) -> f32 {
        let basis = DVector::from_fn(self.deltas.len(), |row, _c| {
            // component from basis functions
            let center = self.centers.row(row);
            let delta = coord - center;
            let r = (delta).norm();
            B::eval(r)
            // linear component
        });
        let res = basis.dot(&self.deltas);
        res
    }

    // The order for the polynomial part, meaning terms up to (order - 1) are included.
    // This usage is consistent with Wilna du Toit's masters thesis "Radial Basis
    // Function Interpolation"
    pub fn create(
        centers: Vec2,
        vals: Vec1,
        // order: usize,
    ) -> Scatter<B> {
        dbg!("creating rbf");
        let n = vals.len();
        // n x m matrix. There's probably a better way to do this, ah well.
        // let mut vals = DMatrix::from_columns(&vals).transpose();
        // We translate the system to center the mean at the origin so that when
        // the system is degenerate, the pseudoinverse below minimizes the linear
        // coefficients.
        // let means: Vec<_> = Vec::new();
        // let vals_mat = DMatrix::f(&[vals]);
        let mat = DMatrix::from_fn(n, n, |r, c| {
            B::eval((centers.row(r) - centers.row(c)).norm())
        });
        // inv is an n' x n' matrix.
        let svd = SVD::new(mat, true, true);
        // Use pseudo-inverse here to get "least squares fit" when there's
        // no unique result (for example, when dimensionality is too small).
        let eps = 1e-4;
        let inv = svd.pseudo_inverse(eps).expect("error inverting matrix");
        // Again, this transpose feels like I don't know what I'm doing.
        let deltas = inv * vals;

        dbg!("rbf created");
        let scatter = Scatter {
            phantom_basis: PhantomData,
            centers,
            deltas,
        };

        println!("test run");
        let coord0 = XY::from([0.0, 0.0]);
        let coord1 = XY::from([0.2, 0.0]);
        let coord2 = XY::from([0.5, 0.0]);
        let res0 = scatter.eval(coord0);
        let res1 = scatter.eval(coord1);
        let res2 = scatter.eval(coord2);
        println!("res0 {}", res0);
        println!("res1 {}", res1);
        println!("res2 {}", res2);

        scatter
    }
}
