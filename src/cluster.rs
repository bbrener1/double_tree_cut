use rulinalg::matrix::{Matrix,BaseMatrix,Axes};
use rulinalg::vector::Vector;
use std::sync::Arc;
use io::Parameters;
use io::Distance;
use std::iter::once;

pub struct Cluster {
    pub id: usize,
    members: Vec<usize>,
    radius: f64,
    center: Vector<f64>,
    matrix: Arc<Matrix<f64>>,
    distance: Distance,
    children: Vec<Cluster>,
}

impl Cluster {
    pub fn from_point(point:usize,id:usize,array:&Arc<Matrix<f64>>,parameters:&Arc<Parameters>) -> Cluster {
        Cluster {
            id:id,
            members: vec![point],
            radius: 0.,
            center: array.row(point).iter().cloned().collect(),
            matrix: array.clone(),
            distance: parameters.distance,
            children: vec![],
        }
    }

    pub fn merge_clusters(c1:Cluster,c2:Cluster,id:usize) -> Cluster {
        let mut new_cluster = Cluster {
            id: id,
            members: c1.members.iter().chain(c2.members.iter()).cloned().collect(),
            radius: 0.,
            center: vector![0.],
            matrix: c1.matrix.clone(),
            distance: c1.distance,
            children: vec![c1,c2]
        };
        new_cluster.center = new_cluster.center();
        new_cluster.radius = new_cluster.radius();
        new_cluster
    }

    pub fn radius(&self) -> f64 {
        let mut acc = 0.;
        let center = &self.center;
        let samples = self.members.len() as f64;
        for member in &self.members {
            eprintln!("{:?}",self.distance.measure(self.matrix.row(*member).as_slice(),center)/samples);
            acc += self.distance.measure(self.matrix.row(*member).as_slice(),center)/samples
        }
        acc
    }

    pub fn center(&self) -> Vector<f64> {
        let mut acc: Vector<f64> = Vector::zeros(self.matrix.cols());
        for member in &self.members {
            acc += (self.matrix.row(*member).iter().cloned().collect::<Vector<f64>>() / self.members.len() as f64);
        }
        acc
    }

    pub fn children<'a>(&'a self) -> Vec<&'a Cluster> {
        self.children.iter().collect()
    }

    pub fn all_children<'a>(&'a self) -> Vec<&'a Cluster> {
        self.children
            .iter()
            .flat_map(|x| x.all_children())
            .chain(self.children.iter())
            .collect()
    }

    pub fn leaves<'a>(&'a self) -> Vec<&'a Cluster> {
        self.children
            .iter()
            .flat_map(|x| x.leaves())
            .chain(if self.children.len() < 1 {Some(self)} else {None})
            .collect()
    }

    pub fn leaf_members(&self) -> Vec<usize> {
        self.leaves()
            .iter()
            .flat_map(|x| x.members.iter())
            .cloned()
            .collect()
    }

    pub fn leaf_coordinates(&self) -> Vec<Vector<f64>> {
        self.leaves()
            .iter()
            .map(|x| x.center())
            .collect()
    }

    pub fn distance_to_children(&self) -> Vec<f64> {
        let center = self.center();
        self.children
            .iter()
            .map(|child| self.distance.measure(&self.center(), &child.center()))
            .collect()
    }

    // pub fn radius_of_children(&self) -> Vec<f64> {
    //
    // }

    pub fn descend_to_clusters(&self) -> Vec<&Cluster> {
        let mut clusters = vec![];
        if self.children.len() > 0 {
            eprintln!("M:{:?}",self.members);
            eprintln!("R:{:?}",self.radius);
            eprintln!("C:{:?}", self.center());
            let left = &self.children[0];
            eprintln!("LM:{:?}",left.members);
            eprintln!("LR:{:?}",left.radius);
            eprintln!("LRR:{:?}",left.radius());
            eprintln!("LC:{:?}",left.center());
            let right = &self.children[1];
            eprintln!("RM:{:?}",right.members);
            eprintln!("RR:{:?}",right.radius);
            eprintln!("RC:{:?}",right.center());
            let child_distance = self.distance.measure(left.center().iter(),right.center().iter());
            eprintln!("DISTANCE:{:?}",child_distance);
            if child_distance > (right.radius + left.radius) * 2. {
                
            }
            clusters.append(&mut left.descend_to_clusters());
            clusters.append(&mut right.descend_to_clusters());

        }

        clusters
    }

    pub fn cluster_samples(&self) -> Vec<usize> {
        let clusters = self.descend_to_clusters();
        let mut samples = vec![0;self.members.len()];
        for cluster in clusters {
            for member in &cluster.members {
                samples[*member] = cluster.id;
            }
        }
        samples
    }

}
