use cluster::Cluster;
use io::{Parameters,Distance,Similarity};
use std::sync::Arc;
use rulinalg::matrix::{Matrix,MatrixSlice,BaseMatrix,Axes};
use rulinalg::vector::Vector;
use rulinalg::utils::{ele_mul,dot,argmax};
use std::cmp::{Ordering,PartialEq};
use std::collections::HashMap;


pub struct Dendrogram {
    pub root: Option<Cluster>,
    samples: usize,
    features: usize,
    parameters: Arc<Parameters>,
    similarity: Similarity,
    // matrix: Arc<Matrix<f64>>,
    row_sim: Matrix<f64>,
    column_sim: Matrix<f64>,
    nearest_neighbors: HashMap<usize,[Vec<(usize,f64)>;2]>,
    smoothing: usize,
    active_clusters: HashMap<usize,Cluster>,
    total_clusters: usize,
    coordinates: Arc<Matrix<f64>>,
}

impl Dendrogram {

    pub fn new(matrix:&Arc<Matrix<f64>>,parameters:&Arc<Parameters>) -> Dendrogram {

        eprintln!("STARTING:");
        eprintln!("{:?}",matrix);
        eprintln!("{:?}",matrix.rows());
        eprintln!("{:?}",matrix.cols());

        let samples = matrix.rows();
        let features = matrix.cols();
        let mut prototype = Dendrogram {
            root: None,
            samples: samples,
            features: features,
            similarity: parameters.similarity,
            parameters: parameters.clone(),
            row_sim: Matrix::identity(samples),
            column_sim: Matrix::identity(features),
            nearest_neighbors: HashMap::new(),
            smoothing: 1,
            active_clusters: HashMap::new(),
            total_clusters: 0,
            coordinates: matrix.clone(),
        };

        eprintln!("{:?}",matrix);

        prototype
    }

    pub fn establish_nearest_neighbors(&mut self) {
        for c1 in self.active_clusters.keys() {
            let (neighbor,similarity) = self.nearest_neighbor(c1);
            {let [_,out_neighbors] = self.nearest_neighbors.entry(*c1).or_insert([vec![],vec![]]);
            out_neighbors.push((neighbor,similarity));}
            {let [in_neighbors,_] = self.nearest_neighbors.entry(neighbor).or_insert([vec![],vec![]]);
            in_neighbors.push((*c1,similarity))}
        }

        eprintln!("{:?}",self.nearest_neighbors);
    }

    pub fn establish_similarity(&mut self) {
        for i in 0..self.coordinates.rows() {
            for j in i+1..self.coordinates.rows() {
                if i != j {
                    let similarity = self.similarity.measure(self.coordinates.row(i).as_slice(),self.coordinates.row(j).as_slice());
                    // eprintln!("{:?}",self.coordinates.row(i).as_slice().into_matrix());
                    // eprintln!("{:?}",self.coordinates.row(j).as_slice().into_matrix());
                    // eprintln!("R:{:?}",similarity);
                    self.row_sim[[i,j]] = similarity;
                    self.row_sim[[j,i]] = similarity;
                }
                else {
                    self.row_sim[[i,j]] = 1.
                }
            }
        }

        for i in 0..self.coordinates.cols() {
            for j in i+1..self.coordinates.cols() {
                if i != j {
                    let similarity = self.similarity.measure(self.coordinates.col(i).as_slice(),self.coordinates.col(i).as_slice());
                    self.column_sim[[i,j]] = similarity;
                    self.column_sim[[j,i]] = similarity;
                }
                else {
                    self.column_sim[[i,j]] = 1.
                }
            }
        }

        eprintln!("{:?}",self.column_sim);
        eprintln!("{:?}",self.row_sim);
    }

    pub fn adjust_coordinate(&self,coordinate_vec:Vector<f64>) -> Matrix<f64> {
        let coordinate_mtx = Matrix::from(coordinate_vec);
        let adjusted_coordinate = (&self.column_sim * &coordinate_mtx).elemul(&coordinate_mtx);
        adjusted_coordinate
    }

    pub fn adjust_coordiantes(&mut self) {
        let coordinate_mtx = Arc::make_mut(&mut self.coordinates).clone();
        // let column_sim = self.column_sim
        let mut adjusted_mtx = &coordinate_mtx * self.column_sim.as_slice();
        adjusted_mtx = adjusted_mtx.elemul(&coordinate_mtx);
        self.coordinates = Arc::new(adjusted_mtx);
    }

    pub fn nearest_neighbor_adjusted(&self, c1_id:usize) -> (usize,f64) {
        let c1 = &self.active_clusters[&c1_id];
        let c1_adjusted = self.adjust_coordinate(c1.center());
        let mut most_similar = None;
        for c2_id in self.active_clusters.keys() {
            if c2_id != &c1_id {
                let c2 = &self.active_clusters[c2_id];
                let c2_adjusted = self.adjust_coordinate(c2.center());
                let similarity = self.similarity.measure(c1_adjusted.iter(),c2_adjusted.iter());
                if let Some((cluster,best_similarity)) = most_similar {
                    if best_similarity < similarity {
                        most_similar = Some((c2,similarity));
                    }
                }
                else {
                    most_similar = Some((c2,similarity));
                }
            }
        }
        most_similar.map(|x| (x.0.id,x.1)).unwrap_or((c1.id,1.))
    }

    pub fn nearest_neighbor(&self, c1_id:&usize) -> (usize,f64) {
        let c1 = &self.active_clusters[c1_id];
        let c1_c = c1.center();
        let mut most_similar = None;
        for c2_id in self.active_clusters.keys() {
            if c2_id != c1_id {
                let c2 = &self.active_clusters[c2_id];
                let c2_c = c2.center();
                let similarity = self.similarity.measure(c1_c.iter(),c2_c.iter());
                if let Some((cluster,best_similarity)) = most_similar {
                    if best_similarity < similarity {
                        most_similar = Some((c2,similarity));
                    }
                }
                else {
                    most_similar = Some((c2,similarity));
                }
            }
        }
        eprintln!("{:?}",c1_id);
        eprintln!("{:?}",c1_c);
        eprintln!("{:?}",most_similar.map(|x| (x.0.id,x.1)));
        eprintln!("{:?}",most_similar.map(|x| (x.0.center(),x.1)));
        most_similar.map(|x| (x.0.id,x.1)).unwrap_or((c1.id,1.))



    }

    pub fn set_nearest_neighbor(&mut self, cluster_id:&usize) {

        if self.nearest_neighbors.contains_key(cluster_id) {

            if self.nearest_neighbors[cluster_id][1].len() > 0 {
                let old_neighbor = self.nearest_neighbors[cluster_id][1][0].0;
                if self.nearest_neighbors.contains_key(&old_neighbor) {
                    if let Some(remove) = self.nearest_neighbors[&old_neighbor][0].iter().position(|x| x.0 == *cluster_id) {
                        self.nearest_neighbors.get_mut(&old_neighbor).unwrap()[0].remove(remove);
                    }
                }
            }

            let (new_neighbor,similarity) = self.nearest_neighbor(cluster_id);

            eprintln!("PICKED NEW NEIGHBOR");

            {
                let [_inbound, outbound] = self.nearest_neighbors.get_mut(cluster_id).unwrap();
                outbound.clear();
                outbound.push((new_neighbor,similarity));
            }
            {
                let [inbound, _outbound] = self.nearest_neighbors.get_mut(&new_neighbor).unwrap();
                inbound.push((*cluster_id,similarity))
            }
        }

    }

    pub fn adjust_similarity(&mut self) {

        for s in 0..self.smoothing {

            for (i,p1r) in self.coordinates.row_iter().enumerate() {
                let p1 = p1r.iter().cloned().collect();
                // eprintln!("{:?}",p1);
                let p1_adjusted = self.adjust_coordinate(p1);
                for (j,p2r) in self.coordinates.row_iter().enumerate() {
                    if i != j {
                        let p2 = p2r.iter().cloned().collect();
                        let p2_adjusted = self.adjust_coordinate(p2);
                        let similarity = self.similarity.measure(p1_adjusted.iter(),p2_adjusted.iter());
                        self.row_sim[[i,j]] = similarity;
                        self.row_sim[[j,i]] = similarity;
                    }
                    else {
                        self.row_sim[[i,j]] = 1.
                    }
                }
            }

            eprintln!("{:?}",self.row_sim);

            for (i,p1c) in self.coordinates.col_iter().enumerate() {
                let p1 = p1c.as_slice().into_matrix();
                // eprintln!("{:?}",p1);
                let p1_adjusted = (&self.row_sim * &p1).elemul(&p1);
                for (j,p2r) in self.coordinates.col_iter().enumerate() {
                    if i != j {
                        let p2 = &p2r.as_slice().into_matrix();
                        let p2_adjusted = (&self.row_sim * p2).elemul(&p2);
                        let similarity = self.similarity.measure(p1_adjusted.iter(),p2_adjusted.iter());
                        self.column_sim[[i,j]] = similarity;
                        self.column_sim[[j,i]] = similarity;
                    }
                    else {
                        self.column_sim[[i,j]] = 1.
                    }
                }
            }

            eprintln!("{:?}",self.column_sim);

        }
    }

    pub fn grow_leaves(&mut self) {
        for sample in 0..self.samples {
            self.active_clusters.insert(self.total_clusters,Cluster::from_point(sample, self.total_clusters, &self.coordinates, &self.parameters));
            self.total_clusters += 1;

        }
        for cluster in &self.active_clusters {
            eprintln!("C:{:?}",cluster.1.leaf_coordinates());
        }

    }

    pub fn agglomerate(&mut self) {

        eprintln!("{:?}",self.nearest_neighbors);

        let (c1_id,c2_id) = self.nearest_neighbors
                        .iter()
                        .max_by(|(_,value1),(_,value2)|
                            {
                                value1[1][0].1.partial_cmp(&value2[1][0].1)
                                .unwrap_or(Ordering::Greater)
                            }
                        )
                        .map(|(key,value)| (*key,value[1][0].0))
                        .unwrap();

        eprintln!("MERGING");
        eprintln!("{:?}",(c1_id,c2_id));

        let inbound_clusters1: &Vec<(usize,f64)> = &self.nearest_neighbors.remove(&c1_id).unwrap()[0];
        let inbound_clusters2: &Vec<(usize,f64)> = &self.nearest_neighbors.remove(&c2_id).unwrap()[0];

        eprintln!("INBOUND:");
        eprintln!("{:?}",inbound_clusters1);
        eprintln!("{:?}",inbound_clusters2);


        let c1 = self.active_clusters.remove(&c1_id).unwrap();
        let c2 = self.active_clusters.remove(&c2_id).unwrap();
        let new_cluster = Cluster::merge_clusters(c1, c2, self.total_clusters);

        let new_cluster_id = new_cluster.id.clone();

        self.active_clusters.insert(new_cluster.id,new_cluster);
        self.nearest_neighbors.insert(new_cluster_id,[vec![],vec![]]);

        for (cluster,_) in inbound_clusters1.iter().chain(inbound_clusters2.iter()) {
            self.set_nearest_neighbor(cluster);
        }
        eprintln!("RESET PREVIOUS CLUSTERS");

        self.set_nearest_neighbor(&new_cluster_id);

        self.total_clusters += 1;

    }

    pub fn link_tree(&mut self) {
        while self.active_clusters.len() > 1 {
            self.agglomerate();
        }
        let root_id = self.active_clusters.keys().cloned().collect::<Vec<usize>>()[0];
        self.root = self.active_clusters.remove(&root_id);
        eprintln!("LINKED");
        eprintln!("{:?}",self.root.as_ref().unwrap().leaf_coordinates());
    }
}



















// *
