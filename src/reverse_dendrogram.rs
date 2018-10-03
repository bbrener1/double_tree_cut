use io::{Parameters,Distance,Similarity};
use std::sync::Arc;
use rulinalg::matrix::{Matrix,MatrixSlice,BaseMatrix,BaseMatrixMut,Axes};
use rulinalg::vector::Vector;
use rulinalg::utils::{ele_mul,dot,argmax};
use std::cmp::{Ordering,PartialEq};
use std::collections::HashMap;
use std::f64;
use std::collections::HashSet;

use rayon::prelude::*;

use std::iter::once;
use std::iter::repeat;

use io::cosine_distance_matrix;

#[derive(Clone,Debug)]
pub struct Point {
    id: usize,
    coordinates: Vector<f64>,
    density: f64,
    links: Vec<usize>,
    step_density: Vec<Option<(usize,f64)>>,
}

impl Point {
    pub fn new(id: usize, coordinates:Vector<f64>,smoothing:usize) -> Point {
        Point {
            id: id,
            coordinates: coordinates,
            density: 0.,
            links: vec![],
            step_density: vec![None;smoothing+1],
        }
    }

    // pub fn density(&self) -> f64 {
    //     for link in
    // }
}

pub struct ReverseDendrogram {
    samples: usize,
    features: usize,
    parameters: Arc<Parameters>,
    similarity: Similarity,
    row_sim: Matrix<f64>,
    column_sim: Matrix<f64>,
    // pub nearest_neighbors: NearestNeigborIndex,
    smoothing: usize,
    pub points: Vec<Point>,
    // links: Vec<Link>,
    distance_matrix: Matrix<Option<f64>>,
    connectivity_matrix: Matrix<Option<f64>>,
    coordinates: Arc<Matrix<f64>>,
}

impl ReverseDendrogram {

    pub fn new(matrix:&Arc<Matrix<f64>>,parameters:&Arc<Parameters>) -> ReverseDendrogram {

        eprintln!("STARTING:");
        // eprintln!("{:?}",matrix);
        // eprintln!("{:?}",matrix.rows());
        // eprintln!("{:?}",matrix.cols());

        let samples = matrix.rows();
        let features = matrix.cols();

        let mut availability_matrix = Matrix::from_fn(matrix.rows(),matrix.rows(), |i,j| { None } );

        for i in 0..samples {
            availability_matrix[[i,i]] = None;
        };

        let mut prototype = ReverseDendrogram {
            samples: samples,
            features: features,
            similarity: parameters.similarity,
            parameters: parameters.clone(),
            row_sim: Matrix::identity(samples),
            column_sim: Matrix::identity(features),
            // nearest_neighbors: NearestNeigborIndex::blank(0..samples),
            smoothing: 1,
            points: vec![],
            // links: vec![],
            distance_matrix: availability_matrix.clone(),
            connectivity_matrix: availability_matrix,
            coordinates: matrix.clone(),
        };

        // eprintln!("{:?}",matrix);

        prototype
    }

    // pub fn establish_nearest_neighbors(&mut self) {
    //     for point in 0..self.samples {
    //         self.set_nearest_neighbor(&point);
    //     }
    //
    //     eprintln!("{:?}",self.nearest_neighbors);
    // }

    pub fn establish_similarity(&mut self) {
        self.row_sim = cosine_distance_matrix(self.coordinates.as_slice());
        self.column_sim = cosine_distance_matrix(self.coordinates.transpose().as_slice());
    }

    // pub fn establish_similarity(&mut self) {
    //     for i in 0..self.coordinates.rows() {
    //         eprintln!("R{:?}",i);
    //         for j in i+1..self.coordinates.rows() {
    //             if i != j {
    //                 let similarity = self.similarity.measure(self.coordinates.row(i).as_slice(),self.coordinates.row(j).as_slice());
    //                 // eprintln!("{:?}",self.coordinates.row(i).as_slice().into_matrix());
    //                 // eprintln!("{:?}",self.coordinates.row(j).as_slice().into_matrix());
    //                 // eprintln!("R:{:?}",similarity);
    //                 self.row_sim[[i,j]] = similarity;
    //                 self.row_sim[[j,i]] = similarity;
    //             }
    //             else {
    //                 self.row_sim[[i,j]] = 1.
    //             }
    //         }
    //     }
    //
    //     for i in 0..self.coordinates.cols() {
    //         eprintln!("C{:?}",i);
    //         for j in i+1..self.coordinates.cols() {
    //             if i != j {
    //                 let similarity = self.similarity.measure(self.coordinates.col(i).as_slice(),self.coordinates.col(i).as_slice());
    //                 self.column_sim[[i,j]] = similarity;
    //                 self.column_sim[[j,i]] = similarity;
    //             }
    //             else {
    //                 self.column_sim[[i,j]] = 1.
    //             }
    //         }
    //     }
    //
    //     // eprintln!("{:?}",self.column_sim);
    //     // eprintln!("{:?}",self.row_sim);
    // }

    pub fn adjust_coordinate(&self,coordinate_vec:Vector<f64>) -> Matrix<f64> {
        let coordinate_mtx = Matrix::from(coordinate_vec);
        let adjusted_coordinate = (&self.column_sim * &coordinate_mtx).elemul(&coordinate_mtx);
        adjusted_coordinate
    }

    pub fn adjust_coordinates(&mut self) {
        let coordinate_mtx = Arc::make_mut(&mut self.coordinates).clone();
        // let column_sim = self.column_sim
        let mut adjusted_mtx = &coordinate_mtx * self.column_sim.as_slice();
        adjusted_mtx = adjusted_mtx.elemul(&coordinate_mtx);
        self.coordinates = Arc::new(adjusted_mtx);
    }

    pub fn n_nearest_available_neighbors(&self,point_id:&usize,n:usize) -> Vec<(usize,f64)> {

        let mut nearest: Vec<(usize,f64)> = Vec::with_capacity(n+1);

        for (i,similarity_opt) in self.distance_matrix.row(*point_id).iter().enumerate() {
            if let Some(similarity) = similarity_opt {
                // eprintln!("NN:{:?}",nearest);
                // eprintln!("N:{:?}",n);
                // eprintln!("SIM:{:?}",similarity);
                let mut insert = None;
                if nearest.len() <= n {
                    insert = Some((nearest.len(),(i,*similarity)));
                }
                for (j,near) in nearest.iter().enumerate() {
                    if near.1 < *similarity {
                        insert = Some((j,(i,*similarity)))
                    }
                    else { break }
                }
                // eprintln!("II:{:?}",insert);
                if let Some((insert_position,neighbor)) = insert {
                    nearest.insert(insert_position,neighbor);
                }
                nearest.truncate(n);
            }
        }

        nearest

    }
    //
    // pub fn nearest_available_neighbor(&self,point_id:&usize) -> Option<(usize,f64)> {
    //     let local_density = self.points[*point_id].density;
    //     eprintln!("Trying to find nearest neighbor");
    //     eprintln!("{:?}",point_id);
    //     eprintln!("{:?}",self.connectivity_matrix.row(*point_id).iter().collect::<Vec<&Option<f64>>>());
    //     eprintln!("{:?}",local_density);
    //     self.connectivity_matrix.row(*point_id)
    //         .iter()
    //         .enumerate()
    //         .flat_map(|x| once(x.0).zip(x.1.iter().cloned()))
    //         .filter(|x| self.points[x.0].density >= local_density)
    //         .max_by(|x,y| x.1.partial_cmp(&y.1).unwrap_or(Ordering::Greater))
    // }
    //
    // pub fn set_nearest_neighbor(&mut self, point_id:&usize) -> Option<()> {
    //
    //     if self.nearest_neighbors.keys.contains(point_id) {
    //
    //         if let Some((ref old_neighbor, ref similarity)) = self.nearest_neighbors.outbound[point_id] {
    //             if self.nearest_neighbors.keys.contains(old_neighbor) {
    //                 let try_remove = self.nearest_neighbors.inbound[old_neighbor].iter().position(|x| x.0 == *point_id);
    //
    //                 if let Some(remove) = try_remove {
    //                     self.nearest_neighbors.inbound.get_mut(&old_neighbor).unwrap().remove(remove);
    //                 }
    //             }
    //         }
    //
    //         if let Some((new_neighbor,similarity)) = self.nearest_available_neighbor(point_id) {
    //             eprintln!("PICKED NEW NEIGHBOR");
    //             eprintln!("PP:{:?}", point_id);
    //             eprintln!("NN:{:?}", new_neighbor);
    //             eprintln!("NS:{:?}", similarity);
    //             self.nearest_neighbors.outbound[point_id] = Some((new_neighbor,similarity));
    //             self.nearest_neighbors.inbound[&new_neighbor].push((*point_id,similarity));
    //             Some(())
    //         }
    //         else { None }
    //
    //     }
    //
    //     else {
    //         self.nearest_neighbors.keys.insert(*point_id);
    //         self.nearest_neighbors.inbound.insert(*point_id, vec![]);
    //         self.nearest_neighbors.outbound.insert(*point_id, None);
    //         self.set_nearest_neighbor(point_id)
    //     }
    //
    // }

    // pub fn adjust_similarity(&mut self) {
    //
    //     for s in 0..self.smoothing {
    //
    //         for (i,p1r) in self.coordinates.row_iter().enumerate() {
    //             eprintln!("S:{:?}",i);
    //             let p1 = p1r.iter().cloned().collect();
    //             // eprintln!("{:?}",p1);
    //             let p1_adjusted = self.adjust_coordinate(p1);
    //             for (j,p2r) in self.coordinates.row_iter().enumerate() {
    //                 if i != j {
    //                     let p2 = p2r.iter().cloned().collect();
    //                     let p2_adjusted = self.adjust_coordinate(p2);
    //                     let similarity = self.similarity.measure(p1_adjusted.iter(),p2_adjusted.iter());
    //                     self.row_sim[[i,j]] = similarity;
    //                     self.row_sim[[j,i]] = similarity;
    //                 }
    //                 else {
    //                     self.row_sim[[i,j]] = 1.
    //                 }
    //             }
    //         }
    //
    //         // eprintln!("{:?}",self.row_sim);
    //
    //         for (i,p1c) in self.coordinates.col_iter().enumerate() {
    //             eprintln!("S:{:?}",i);
    //             let p1 = p1c.as_slice().into_matrix();
    //             // eprintln!("{:?}",p1);
    //             let p1_adjusted = (&self.row_sim * &p1).elemul(&p1);
    //             for (j,p2r) in self.coordinates.col_iter().enumerate() {
    //                 if i != j {
    //                     let p2 = &p2r.as_slice().into_matrix();
    //                     let p2_adjusted = (&self.row_sim * p2).elemul(&p2);
    //                     let similarity = self.similarity.measure(p1_adjusted.iter(),p2_adjusted.iter());
    //                     self.column_sim[[i,j]] = similarity;
    //                     self.column_sim[[j,i]] = similarity;
    //                 }
    //                 else {
    //                     self.column_sim[[i,j]] = 1.
    //                 }
    //             }
    //         }
    //
    //         // eprintln!("{:?}",self.column_sim);
    //
    //     }
    // }

    pub fn establish_connectivity(&mut self) {
        for i in 0..self.samples {
            for j in 0..self.samples {
                self.distance_matrix[[i,j]] = Some(self.row_sim[[i,j]]);
            }
        }

        for i in 0..self.samples {
            self.distance_matrix[[i,i]] = None;
        }

        // eprintln!("Connectivity:");
        // eprintln!("{:?}",self.connectivity_matrix);
    }

    //
    //
    // pub fn establish_density(&mut self) {
    //
    //     for i in 0..self.samples {
    //
    //         let distances: Vec<f64> = self.n_nearest_available_neighbors(&i, self.parameters.smoothing)
    //                                     .iter()
    //                                     .map(|(n_i,distance)| *distance)
    //                                     .collect();
    //
    //         eprintln!("DD:{:?}", distances);
    //
    //         let mean_squared_density = distances
    //                         .iter()
    //                         .map(|x| x.powi(2))
    //                         .sum::<f64>() / distances.len() as f64;
    //
    //         let density = mean_squared_density.sqrt();
    //
    //         self.points[i].density = density;
    //
    //     }
    //
    //     // eprintln!("{:?}",self.points);
    // }

    pub fn establish_density(&mut self) {

        let mut densities = vec![];

        let smoothing = self.parameters.smoothing;

        for point in 0..self.points.len() {

            // let distances = &self.n_step_distances(&point.id, self.parameters.smoothing);
            //
            // let density = distances.iter().sum::<f64>() / distances.len() as f64;
            //
            densities.push(self.n_step_density(&point, smoothing));



        }

        for (i,density) in densities.into_iter().enumerate() {
            self.points[i].density = density.unwrap().1;
        }

        // eprintln!("{:?}",self.points);
    }

    pub fn n_step_links(&self, point_index: &usize, n: usize) -> Vec<(usize,f64)> {
        if n > 0 {
            let mut linked_points: Vec<(usize,f64)> = vec![];
            let point = &self.points[*point_index];
            for linked in &point.links {
                // linked_points.append(&mut self.links(linked));
                linked_points.append(&mut self.n_step_links(&linked, n-1));
            }
            linked_points
        }
        else {
            vec![]
        }

    }

    pub fn links(&self, point_index:&usize) -> Vec<(usize,f64)> {
        let mut link_vec = vec![];
        let point = &self.points[*point_index];
        for link in &point.links {
            link_vec.push((*link,self.connectivity_matrix[[point.id,*link]].unwrap()));
        }
        link_vec
    }

    pub fn n_step_density(&mut self, point_index: &usize, n: usize) -> Option<(usize,f64)> {
        if n > 1 {
            if let Some((weight,density)) = self.points[*point_index].step_density[n] {
                return Some((weight,density))
            }
            else {

                let closer = self.n_step_density(point_index, n-1);
                if let Some((closer_weight,closer_density)) = closer {
                    let far_links = self.n_step_links(point_index, n);
                    let weight = far_links.iter().map(|x| x.0).sum::<usize>();
                    let distances = far_links.iter().map(|x| x.0 as f64 * x.1).sum::<f64>();
                    let result = (weight,distances/weight as f64);
                    self.points[*point_index].step_density[n] = Some(result);
                    Some(result)
                }
                else { None }
            }


        }
        else {
            None
        }
    }


    pub fn n_step_distances(&self, point_index: &usize, n: usize) -> Vec<f64> {
        if n > 0 {
            let mut linked_points: Vec<f64> = vec![];
            let point = &self.points[*point_index];
            for linked in &point.links {
                linked_points.push(self.connectivity_matrix[[*point_index,*linked]].unwrap());
                linked_points.append(&mut self.n_step_distances(&linked, n-1));
            }
            linked_points
        }
        else {
            vec![]
        }

    }


    // pub fn walk_n_links (&self,point_index: &usize, n: usize) -> Vec<f64> {
    //     if n > 0 {
    //         let mut walked_links: Vec<f64> = vec![];
    //         let p1 = &self.points[*point_index];
    //         for linked in &p1.links {
    //             let p2 = &self.points[*linked];
    //             let length = self.similarity.measure(&p1.coordinates,&p2.coordinates);
    //             walked_links.push(length);
    //             walked_links.append(&mut self.walk_n_links(&linked, n-1));
    //         }
    //         walked_links
    //     }
    //     else {
    //         vec![]
    //     }
    // }

    pub fn walk_uphill (&self,point_index: &usize) -> usize {

        eprintln!("PP:{:?}",point_index);

        let mut current = *point_index;

        loop {
            let step = self.step_uphill(&current);
            // eprintln!("step:{:?}",step);
            if current == step {
                break
            }
            else {
                current = step;
            }
        }

        current

    }

    pub fn step_uphill(&self, point_index: &usize) -> usize {

        let point = &self.points[*point_index];

        let local_density = point.density;

        let mut uphill_link = (*point_index,local_density);

        eprintln!("UH:{:?}",local_density);

        for link in &point.links {
            let target = &self.points[*link];
            eprintln!("TT:{:?}",target.density);
            if uphill_link.1 < target.density {
                uphill_link = (target.id, target.density);
            }
        }

        uphill_link.0
    }

    pub fn establish_points(&mut self) {
        for (i,row) in self.coordinates.row_iter().enumerate() {
            self.points.push(Point::new(i,row.iter().cloned().collect(),self.parameters.smoothing));
            // eprintln!("{:?}", i);
        }
    }
    //
    // pub fn density_slope(&self,point_id: &usize, smoothing: usize) -> f64 {
    //     let steps = self.walk_uphill(point_id, smoothing);
    //     steps.iter().sum::<f64>() / steps.len() as f64
    // }

    // pub fn smooth_density(&self, point_id: &usize, smoothing: usize) -> f64 {
    //     let links = self.walk_n_links(point_id, smoothing);
    //     links.iter().sum::<f64>() / links.len() as f64
    // }

    // pub fn smooth_density(&self, point_id: &usize,smoothing:usize) -> f64 {
    //     let origin = &self.points[*point_id];
    //     let links = self.n_step_links(point_id, smoothing);
    //     let distance_sum: f64 = links
    //         .iter()
    //         .map(|x| &self.points[*x])
    //         .map(|point| self.similarity.measure(&point.coordinates,&origin.coordinates))
    //         .sum();
    //     distance_sum / links.len() as f64
    // }
    //
    // pub fn density(&self, point_id: &usize) -> f64 {
    //     self.smooth_density(point_id, 1)
    // }

    pub fn nearest_neighbor_pair (&self) -> Option<(usize,usize,f64)> {

        let mut max_pair = None;

        for i in 0..self.samples {
            let i_density = self.points[i].density;
            for j in 0..self.samples {
                if let Some(similarity) = self.distance_matrix[[i,j]] {
                    if let Some((max_i,max_j,max_sim)) = max_pair {
                        let j_density = self.points[j].density;
                        if max_sim < similarity
                            // (i_density > similarity || j_density > similarity)
                        {
                            max_pair = Some((i,j,similarity))
                        }
                    }
                    else {
                        max_pair = Some((i,j,similarity))
                    }
                }
            }
        }

        // eprintln!("NN:{:?}",max_pair);
        if let Some((p1,p2,sim)) = max_pair.clone() {
            // eprintln!("{:?}", self.points[p1].density);
            // eprintln!("{:?}", self.points[p2].density);
            // eprintln!("{:?}", sim);
        }
        max_pair
    }

    // pub fn furthest_nearest_neighbor_pair (&self) -> Option<(usize,usize,f64)> {
    //
    //     let mut min_pair = None;
    //
    //     for i in 0..self.samples {
    //         let mut max_pair: Option<(usize,usize,f64)> = None;
    //         for j in 0..self.samples {
    //             if let Some(similarity) = self.distance_matrix[[i,j]] {
    //                 eprintln!("MP:{:?}",max_pair);
    //                 eprintln!("PP:{:?}",Some((i,j,similarity)));
    //                 if let Some((max_i,max_j,max_sim)) = max_pair {
    //                     if max_sim < similarity {
    //                         max_pair = Some((i,j,similarity))
    //                     }
    //                 }
    //                 else {
    //                     max_pair = Some((i,j,similarity))
    //                 }
    //             }
    //             // else {break}
    //         }
    //
    //         if let Some((max_i,max_j,max_s)) = max_pair {
    //             if let Some((min_i,min_j,min_s)) = min_pair {
    //                 let i_max_density = self.points[max_i].density;
    //                 let j_max_density = self.points[max_j].density;
    //                 if i_max_density < max_s || j_max_density < max_s {
    //                     if max_s < min_s {
    //                         eprintln!("{:?}",i_max_density);
    //                         eprintln!("{:?}", j_max_density);
    //                         eprintln!("{:?}", max_s);
    //                         min_pair = Some((max_i,max_j,max_s))
    //                     }
    //                 }
    //             }
    //             else {min_pair = Some((max_i,max_j,max_s))}
    //         }
    //
    //         eprintln!("MINP:{:?}",min_pair);
    //
    //
    //     }
    //
    //
    //
    //     eprintln!("NN:{:?}",min_pair);
    //     if let Some((p1,p2,sim)) = min_pair.clone() {
    //         eprintln!("{:?}", self.points[p1].density);
    //         eprintln!("{:?}", self.points[p2].density);
    //         eprintln!("{:?}", sim);
    //     }
    //     min_pair
    // }


    pub fn link_points(&mut self,p1_id:&usize,p2_id:&usize,similarity:f64) {

        eprintln!("LINKING:{:?}", (p1_id,p2_id,similarity));

        // eprintln!("DISTANCE{:?}",self.distance_matrix.row(*p1_id).iter().collect::<Vec<&Option<f64>>>());
        // eprintln!("DISTANCE{:?}",self.distance_matrix.row(*p2_id).iter().collect::<Vec<&Option<f64>>>());

        let removed_links: Vec<usize> = self.distance_matrix.row(*p1_id).iter()
            .zip(self.distance_matrix.row(*p2_id).iter())
            .enumerate()
            .filter_map(|(i,(x,y))| if x.is_none() || y.is_none() {Some(i)} else {None})
            .collect();

        // eprintln!("REMOVING:{:?}",removed_links);

        for i in &removed_links {
            for j in &removed_links {
                self.distance_matrix[[*j,*i]] = None;
                self.distance_matrix[[*i,*j]] = None;
            }
        }

        self.connectivity_matrix[[*p1_id,*p2_id]] = Some(similarity);
        self.connectivity_matrix[[*p2_id,*p1_id]] = Some(similarity);

        // eprintln!("DISTANCE:{:?}",self.distance_matrix.row(*p1_id).iter().collect::<Vec<&Option<f64>>>());
        // eprintln!("DISTANCE:{:?}",self.distance_matrix.row(*p2_id).iter().collect::<Vec<&Option<f64>>>());
        //
        // eprintln!("CONNECTIVITY:{:?}",self.connectivity_matrix.row(*p1_id).iter().collect::<Vec<&Option<f64>>>());
        // eprintln!("CONNECTIVITY:{:?}",self.connectivity_matrix.row(*p2_id).iter().collect::<Vec<&Option<f64>>>());

        self.points[*p1_id].links.push(*p2_id);
        self.points[*p2_id].links.push(*p1_id);
        // self.points[*p1_id].density = self.density_slope(p1_id, self.parameters.smoothing);
        // self.points[*p2_id].density = self.density_slope(p2_id, self.parameters.smoothing);

        // eprintln!("Linked, try to set neighbors:");

        // if
        //     self.set_nearest_neighbor(p1_id).is_none() &&
        //     self.set_nearest_neighbor(p2_id).is_none() {
        //         None
        //     }
        // else { Some(()) }
    }

    pub fn least_spanning_tree(&mut self) {

        // eprintln!("{:?}", self.nearest_neighbors);

        while let Some((p1,p2,sim)) = self.nearest_neighbor_pair() {
            self.link_points(&p1,&p2,sim);
        }

    }

    pub fn density(&self) -> Vec<f64> {
        self.points.iter().map(|x| x.density).collect()
    }

    pub fn cluster(&self) -> Vec<usize> {
        self.points.iter().map(|x| self.walk_uphill(&x.id)).collect()
    }
}

        // let link: (usize,(Vec<(usize,f64)>,Vec<(usize,f64)>)) = self.nearest_neighbors
        // if let Some((p1,(inbound,outbound))) = self.nearest_neighbors
        //     .iter()
        //     // .map(|x: (&usize,&(Vec<(usize,f64)>,Vec<(usize,f64)>))| (*x.0,x.1))
        //     // .map(|x: (usize,&(Vec<(usize,f64)>,Vec<(usize,f64)>))| x)
        //     .max_by(|x,y| (x.1).1[0].1.partial_cmp(&(y.1).1[0].1).unwrap_or(Ordering::Greater))
        //     .map(|x| (*x.0,((x.1.clone()).0,(x.1.clone()).1)))
        //
        //     {
        //         eprintln!("Linking");
        //         eprintln!("{:?}",(&p1, &outbound[0].0));
        //         self.link_points(&p1, &outbound[0].0)
        //     }
        //
        // else {
        //     eprintln!("Failed max?");
        //     eprintln!("{:?}", self.nearest_neighbors.iter().max_by(|x,y| (x.1).1[0].partial_cmp(&(y.1).1[0]).unwrap_or(Ordering::Greater)));
        //     None
        // }
            // .unwrap();

        // let (p1,(_,p2v)) = link;

    // }
    //
    // pub fn build_trees(&mut self) {
    //
    //     // eprintln!("{:?}", self.nearest_neighbors);
    //     // eprintln!("{:?}", self.nearest_neighbors
    //     //     .iter()
    //     //     // .map(|x: (&usize,&(Vec<(usize,f64)>,Vec<(usize,f64)>))| (*x.0,x.1))
    //     //     // .map(|x: (usize,&(Vec<(usize,f64)>,Vec<(usize,f64)>))| x)
    //     //     .max_by(|x,y| (x.1).1[0].1.partial_cmp(&(y.1).1[0].1).unwrap_or(Ordering::Greater))
    //     // );
    //
    //     while let Some(()) = self.agglomerate() {
    //         eprintln!("{:?}", self.connectivity_matrix);
    //     }
    // }



//
//     pub fn agglomerate(&mut self) {
//
//         eprintln!("{:?}",self.nearest_neighbors);
//
//
//
//         eprintln!("MERGING");
//         eprintln!("{:?}",(c1_id,c2_id));
//
//         let inbound_clusters1: &Vec<(usize,f64)> = &self.nearest_neighbors.remove(&c1_id).unwrap()[0];
//         let inbound_clusters2: &Vec<(usize,f64)> = &self.nearest_neighbors.remove(&c2_id).unwrap()[0];
//
//         eprintln!("INBOUND:");
//         eprintln!("{:?}",inbound_clusters1);
//         eprintln!("{:?}",inbound_clusters2);
//
//
//         let c1 = self.active_clusters.remove(&c1_id).unwrap();
//         let c2 = self.active_clusters.remove(&c2_id).unwrap();
//         let new_cluster = Cluster::merge_clusters(c1, c2, self.total_clusters);
//
//         let new_cluster_id = new_cluster.id.clone();
//
//         self.active_clusters.insert(new_cluster.id,new_cluster);
//         self.nearest_neighbors.insert(new_cluster_id,[vec![],vec![]]);
//
//         for (cluster,_) in inbound_clusters1.iter().chain(inbound_clusters2.iter()) {
//             self.set_nearest_neighbor(cluster);
//         }
//         eprintln!("RESET PREVIOUS CLUSTERS");
//
//         self.set_nearest_neighbor(&new_cluster_id);
//
//         self.total_clusters += 1;
//
//     }
//
//     pub fn link_tree(&mut self) {
//         while self.active_clusters.len() > 1 {
//             self.agglomerate();
//         }
//         let root_id = self.active_clusters.keys().cloned().collect::<Vec<usize>>()[0];
//         self.root = self.active_clusters.remove(&root_id);
//         eprintln!("LINKED");
//         eprintln!("{:?}",self.root.as_ref().unwrap().leaf_coordinates());
//     }
// }
//
//

















// *
