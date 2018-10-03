mod cluster;
mod dendrogram;
mod io;
mod reverse_dendrogram;

extern crate num_cpus;
extern crate rayon;

#[macro_use]
extern crate rulinalg;

use io::Parameters;
use io::write_vec;
use std::env;
use reverse_dendrogram::ReverseDendrogram;
use std::sync::Arc;
use rulinalg::matrix::Matrix;

fn main() {

    let mut arg_iter = env::args();

    let mut parameters_raw = Parameters::read(&mut arg_iter);

    let matrix = Arc::new(parameters_raw.counts.take().unwrap());

    let parameters = Arc::new(parameters_raw);

    // let test_matrix: Matrix<f64> = Matrix::new::<Vec<f64>>(10,2,vec![1,1,2,2,3,3,6,6,8,8,9,9,110,110,140,140,150,150,200,200].into_iter().map(|x| x as f64).collect());

    let mut dendrogram = ReverseDendrogram::new(&matrix,&parameters);

    dendrogram.establish_points();

    eprintln!("Established points");

    dendrogram.establish_similarity();

    eprintln!("Established similarity");

    // dendrogram.adjust_similarity();
    dendrogram.adjust_coordinates();
    dendrogram.establish_similarity();

    eprintln!("Adjusted");

    dendrogram.establish_connectivity();

    eprintln!("Connectivity");

    dendrogram.least_spanning_tree();

    eprintln!("Trees");

    dendrogram.establish_density();

    eprintln!("Density");
    // dendrogram.establish_nearest_neighbors();

    // eprintln!("Nearest Neighbors");


    // eprintln!("{:?}", dendrogram.points);

    let clusters = dendrogram.cluster();

    eprintln!("Clusters: {:?}", clusters);

    write_vec(clusters,&parameters.report_address);
    // eprintln!("Nearest Neighbors:");
    // eprintln!("{:?}",dendrogram.nearest_neighbors);
    // eprintln!("{:?}",root.cluster_samples());
    // eprintln!("{:?}",root.leaf_members());



}
