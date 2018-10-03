use std::fs::File;
use std::fs::OpenOptions;
use std::io::Error;

use std::io;
use std::io::prelude::*;
use std::collections::HashMap;
use num_cpus;
use std::f64;
use std::fmt::Debug;
use std::convert::Into;
use rayon::iter::IntoParallelIterator;

use rulinalg::matrix::{Matrix,BaseMatrix,BaseMatrixMut,Axes,MatrixSlice};
use rulinalg::vector::Vector;

use std::iter::IntoIterator;

#[derive(Debug,Clone)]
pub struct Parameters {
    auto: bool,
    pub command: Command,
    pub counts: Option<Matrix<f64>>,
    pub feature_names: Vec<String>,
    pub sample_names: Vec<String>,
    pub report_address: Option<String>,
    pub dump_error: Option<String>,

    pub smoothing: usize,
    pub refining: bool,
    pub distance: Distance,
    pub similarity: Similarity,

    count_array_file: String,
    feature_header_file: Option<String>,
    sample_header_file: Option<String>,

    processor_limit: usize,

}

impl Parameters {

    pub fn empty() -> Parameters {
        let arg_struct = Parameters {
            auto: false,
            command: Command::FitPredict,
            count_array_file: "".to_string(),
            counts: None,
            feature_header_file: None,
            feature_names: vec![],
            sample_header_file: None,
            sample_names: vec![],
            report_address: None,
            dump_error: None,
            distance: Distance::Euclidean,
            similarity: Similarity::Pearson,

            processor_limit: 1,

            smoothing: 0,
            refining: false,
        };
        arg_struct
    }

    pub fn read<T: Iterator<Item = String>>(args: &mut T) -> Parameters {

        let mut arg_struct = Parameters::empty();

        let _raw_command = args.next();

        arg_struct.command = Command::parse(&args.next().expect("Please enter a command"));

        let mut _supress_warnings = false;

        while let Some((i,arg)) = args.enumerate().next() {

                match &arg[..] {
                "-sw" | "-suppress_warnings" => {
                    if i!=1 {
                        eprintln!("If the supress warnings flag is not given first it may not function correctly.");
                    }
                _supress_warnings = true;
                },
                // "-auto" | "-a"=> {
                //     arg_struct.auto = true;
                //     arg_struct.auto()
                // },
                "-c" | "-counts" => {
                    arg_struct.count_array_file = args.next().expect("Error parsing count location!");
                    arg_struct.counts = Some(read_counts(&arg_struct.count_array_file));
                },
                "-stdin" => {
                    arg_struct.counts = Some(read_standard_in());
                }
                "-stdout" => {
                    arg_struct.report_address = None;
                }
                "-p" | "-processors" | "-threads" => {
                    if let Ok(processor_limit) = args.next().expect("arg err").parse::<usize>() {
                        arg_struct.processor_limit = processor_limit
                    }
                    else {
                        panic!("Failed to read processor limit!");
                    }
                },
                "-o" | "-output" => {
                    arg_struct.report_address = Some(args.next().expect("Error processing output destination"))
                },
                "-f" | "-h" | "-features" | "-header" => {
                    arg_struct.feature_header_file = Some(args.next().expect("Error processing feature file"));
                    arg_struct.feature_names = read_header(arg_struct.feature_header_file.as_ref());
                },
                "-s" | "-samples" => {
                    arg_struct.sample_header_file = Some(args.next().expect("Error processing feature file"));
                    arg_struct.sample_names = read_sample_names(arg_struct.sample_header_file.as_ref());
                }
                "-smoothing" => {
                    arg_struct.smoothing = args.next().map(|x| x.parse::<usize>()).expect("Smoothing distance parse error. Not a number?").expect("Smoothing parse error");
                }
                "-error" => {
                    arg_struct.dump_error = Some(args.next().expect("Error processing error destination"))
                },
                "-r" | "-refining" => {
                    arg_struct.refining = true;
                },
                "-d" | "-distance" => {
                    arg_struct.distance = args.next().map(|x| Distance::parse(&x)).expect("Distance parse error")
                }
                "-sim" | "-similarity" => {
                    arg_struct.similarity = args.next().map(|x| Similarity::parse(&x)).expect("Similarity parse error");
                }

                &_ => {
                    panic!("Not a valid argument: {}", arg);
                }

            }
        }

        arg_struct

    }



    // fn auto(&mut self) {
    //
    //     let counts = self.counts.as_ref().expect("Please specify counts file before the \"-auto\" argument.");
    //
    //     let features = counts.shape()[1];
    //     let samples = counts.shape()[0];
    //
    //     let mut output_features = ((features as f64 / (features as f64).log10()) as usize).min(features);
    //
    //     let input_features: usize;
    //
    //     if features < 3 {
    //         input_features = features;
    //         output_features = features;
    //     }
    //     else if features < 100 {
    //         input_features = ((features as f64 * ((125 - features) as f64) / 125.) as usize).max(1);
    //     }
    //
    //     else {
    //         input_features = ((features as f64 * (((1500 - features as i32) as f64) / 7000.).max(0.1)) as usize).max(1);
    //     }
    //
    //     let feature_subsample = output_features;
    //
    //     let sample_subsample: usize;
    //
    //     if samples < 10 {
    //         eprintln!("Warning, you seem to be using suspiciously few samples, are you sure you specified the right file? If so, trees may not be the right solution to your problem.");
    //         sample_subsample = samples;
    //     }
    //     else if samples < 1000 {
    //         sample_subsample = (samples/3)*2;
    //     }
    //     else if samples < 5000 {
    //         sample_subsample = samples/2;
    //     }
    //     else {
    //         sample_subsample = samples/4;
    //     }
    //
    //     let processors = num_cpus::get();
    //
    //
    //     println!("Automatic parameters:");
    //     println!("{:?}",feature_subsample);
    //     println!("{:?}",sample_subsample);
    //     println!("{:?}",input_features);
    //     println!("{:?}",output_features);
    //     println!("{:?}",processors);
    //
    //     self.auto = true;
    //
    //     self.feature_subsample.get_or_insert( feature_subsample );
    //     self.sample_subsample.get_or_insert( sample_subsample );
    //
    //
    //     self.processor_limit.get_or_insert( processors );
    //
    // }

    pub fn distance(&self, p1:&Vector<f64>,p2:&Vector<f64>) -> f64 {
        self.distance.measure(p1,p2)
    }

}


fn read_header(location_option: Option<&String>) -> Vec<String> {

    if let Some(location) = location_option
    {
        eprintln!("Reading header: {}", location);

        let mut header_map = HashMap::new();

        let header_file = File::open(location).expect("Header file error!");
        let mut header_file_iterator = io::BufReader::new(&header_file).lines();

        for (i,line) in header_file_iterator.by_ref().enumerate() {
            let feature = line.unwrap_or("error".to_string());
            let mut renamed = feature.clone();
            let mut j = 1;
            while header_map.contains_key(&renamed) {
                renamed = [feature.clone(),j.to_string()].join("");
                eprintln!("WARNING: Two individual features were named the same thing: {}",feature);
                j += 1;
            }
            header_map.insert(renamed,i);
        };

        let mut header_inter: Vec<(String,usize)> = header_map.iter().map(|x| (x.0.clone().clone(),x.1.clone())).collect();
        header_inter.sort_unstable_by_key(|x| x.1);
        let header_vector: Vec<String> = header_inter.into_iter().map(|x| x.0).collect();

        eprintln!("Read {} lines", header_vector.len());

        header_vector
    }
    else {
        vec![]
    }
}

fn read_sample_names(location_option: Option<&String>) -> Vec<String> {

    if let Some(location) = location_option {
        let mut header_vector = Vec::new();

        let sample_name_file = File::open(location).expect("Sample name file error!");
        let mut sample_name_lines = io::BufReader::new(&sample_name_file).lines();

        for line in sample_name_lines.by_ref() {
            header_vector.push(line.expect("Error reading header line!").trim().to_string())
        }

        header_vector
    }

    else {
        vec![]
    }
}



fn read_counts(location:&str) -> Matrix<f64> {


    let count_array_file = File::open(location).expect("Count file error!");
    let mut count_array_lines = io::BufReader::new(&count_array_file).lines();

    let mut counts: Vec<f64> = Vec::new();
    let mut samples = 0;

    for (i,line) in count_array_lines.by_ref().enumerate() {

        samples += 1;
        let mut gene_vector = Vec::new();

        let gene_line = line.expect("Readline error");

        for (j,gene) in gene_line.split_whitespace().enumerate() {

            if j == 0 && i%200==0{
                print!("\n");
            }

            if i%200==0 && j%200 == 0 {
                print!("{} ", gene.parse::<f64>().unwrap_or(-1.) );
            }

            // if !((gene.0 == 1686) || (gene.0 == 4660)) {
            //     continue
            // }

            match gene.parse::<f64>() {
                Ok(exp_val) => {

                    gene_vector.push(exp_val);

                },
                Err(msg) => {

                    if gene != "nan" && gene != "NAN" {
                        eprintln!("Couldn't parse a cell in the text file, Rust sez: {:?}",msg);
                        eprintln!("Cell content: {:?}", gene);
                    }
                    gene_vector.push(f64::NAN);
                }
            }

        }

        counts.append(&mut gene_vector);

        if i % 100 == 0 {
            eprintln!("{}", i);
        }


    };

    let matrix = Matrix::new(samples,counts.len()/samples, counts);

    eprintln!("===========");
    eprintln!("{},{}", matrix.rows(), matrix.cols());

    matrix

}

fn read_standard_in() -> Matrix<f64> {

    let stdin = io::stdin();
    let count_array_pipe_guard = stdin.lock();

    let mut counts: Vec<f64> = Vec::new();
    let mut samples = 0;

    for (_i,line) in count_array_pipe_guard.lines().enumerate() {

        samples += 1;
        let mut gene_vector = Vec::new();

        for (_j,gene) in line.as_ref().expect("readline error").split_whitespace().enumerate() {

            match gene.parse::<f64>() {
                Ok(exp_val) => {

                    gene_vector.push(exp_val);

                },
                Err(msg) => {

                    if gene != "nan" && gene != "NAN" {
                        eprintln!("Couldn't parse a cell in the text file, Rust sez: {:?}",msg);
                        eprintln!("Cell content: {:?}", gene);
                    }
                    gene_vector.push(f64::NAN);
                }
            }

        }

        counts.append(&mut gene_vector);

    };

    // eprintln!("Counts read:");
    // eprintln!("{:?}", counts);

    let matrix = Matrix::new(samples,counts.len()/samples, counts);

    matrix

}

#[derive(Debug,Clone,Copy,Hash)]
pub enum Distance {
    Manhattan,
    Euclidean,
    Cosine,
}

impl Distance {
    pub fn parse(argument: &str) -> Distance {
        match &argument[..] {
            "manhattan" | "m" | "cityblock" => Distance::Manhattan,
            "euclidean" | "e" => Distance::Euclidean,
            "cosine" | "c" | "cos" => Distance::Cosine,
            _ => {
                eprintln!("Not a valid distance option, defaulting to cosine");
                Distance::Cosine
            }
        }
    }

    pub fn measure<'a,T,U>(&self,p1:T,p2:U) -> f64
    where
        T: IntoIterator<Item = &'a f64>,
        U: IntoIterator<Item = &'a f64>,
    {
        match self {
            Distance::Manhattan => {
                p1.into_iter().zip(p2.into_iter()).map(|(x,y)| x-y).sum()
            },
            Distance::Euclidean => {
                p1.into_iter().zip(p2.into_iter()).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt()
            },
            Distance::Cosine => {
                let mut dot_sum = 0.;
                let mut p1ss = 0.;
                let mut p2ss = 0.;
                for (x,y) in p1.into_iter().zip(p2.into_iter()) {
                    dot_sum += x*y;
                    p1ss += x.powi(2);
                    p2ss += y.powi(2);
                }
                let p1ss = p1ss.sqrt();
                let p2ss = p2ss.sqrt();
                1.0 - (dot_sum / (p1ss * p2ss))
            }
        }
    }
}

#[derive(Debug,Clone,Copy,Hash)]
pub enum Similarity {
    Pearson,
    Cosine,
    Inverse,
}

impl Similarity {
    pub fn parse(argument: &str) -> Similarity {
        match &argument[..] {
            "pearson" | "correlation" => Similarity::Pearson,
            "cosine" | "cos" => Similarity::Cosine,
            "inverse" | "euclidean" => Similarity::Inverse,
            _ => {
                eprintln!("Not a valid distance option, defaulting to cosine");
                Similarity::Cosine
            }
        }
    }

    pub fn measure<'a,T,U>(&self,p1:T,p2:U) -> f64
    where
        T: IntoIterator<Item = &'a f64>,
        U: IntoIterator<Item = &'a f64>,
    {
        let x = match self {
            Similarity::Cosine => {
                let mut dot_sum = 0.;
                let mut p1ss = 0.;
                let mut p2ss = 0.;
                for (x,y) in p1.into_iter().zip(p2.into_iter()) {
                    dot_sum += x*y;
                    p1ss += x.powi(2);
                    p2ss += y.powi(2);
                }
                let p1ss = p1ss.sqrt();
                let p2ss = p2ss.sqrt();
                (dot_sum / (p1ss * p2ss))
            },
            Similarity::Inverse => {
                1. / p1.into_iter().zip(p2.into_iter()).map(|(x,y)| (x-y).powi(2)).sum::<f64>().sqrt()
            },
            Similarity::Pearson => {

                let p1v: Vec<f64> = p1.into_iter().cloned().collect();
                let p2v: Vec<f64> = p2.into_iter().cloned().collect();
                let p1m = p1v.iter().sum::<f64>()/(p1v.len() as f64);
                let p2m = p2v.iter().sum::<f64>()/(p2v.len() as f64);
                let p1ss = p1v.iter().map(|x| x.powi(2)).sum::<f64>();
                let p2ss = p1v.iter().map(|x| x.powi(2)).sum::<f64>();

                let product_vector: Vec<f64> = p1v.iter().zip(p2v.iter()).map(|(x,y)| x*y).collect();

                let exy = product_vector.iter().sum::<f64>() / (product_vector.len() as f64);
                let exey = p1m * p2m;

                let exs = p1ss / (p1v.len() as f64);
                let exss = p1m.powi(2);
                let eys = p2ss / (p2v.len() as f64);
                let eyss = p2m / (p2v.len() as f64);

                (exy - exey) / ((exs - exss).sqrt() * (eys - eyss).sqrt())

            }
        };

        if x.is_finite(){
            return x
        }
        else {
            return 0.
        }
    }
}

#[derive(Debug,Clone)]
pub enum Command {
    Fit,
    Predict,
    FitPredict,
}

impl Command {

    pub fn parse(command: &str) -> Command {

        match &command[..] {
            "fit" => Command::Fit,
            "predict" => Command::Predict,
            "fitpredict" | "fit_predict" | "combined" => Command::FitPredict,
            _ =>{
                eprintln!("Not a valid top-level command, please choose from \"fit\",\"predict\", or \"fitpredict\". Exiting");
                panic!()
            }
        }
    }
}


pub fn write_array<T: Debug>(input: Matrix<T>,target:&Option<String>) -> Result<(),Error> {
    let formatted =
        input
        .row_iter()
        .map(|x| x.iter()
            .map(|y| format!("{:?}",y))
            .collect::<Vec<String>>()
            .join("\t")
        )
        .collect::<Vec<String>>()
        .join("\n");

    match target {
        Some(location) => {
            let mut target_file = OpenOptions::new().create(true).append(true).open(location).unwrap();
            target_file.write(&formatted.as_bytes())?;
            target_file.write(b"\n")?;
            Ok(())
        }
        None => {
            let mut stdout = io::stdout();
            let mut stdout_handle = stdout.lock();
            stdout_handle.write(&formatted.as_bytes())?;
            stdout_handle.write(b"\n")?;
            Ok(())
        }
    }
}

pub fn write_vector<T: Debug>(input: Vector<T>,target: &Option<String>) -> Result<(),Error> {
    let formatted =
        input
        .iter()
        .map(|x| format!("{:?}",x))
        .collect::<Vec<String>>()
        .join("\n");

    match target {
        Some(location) => {
            let mut target_file = OpenOptions::new().create(true).append(true).open(location).unwrap();
            target_file.write(&formatted.as_bytes())?;
            target_file.write(b"\n")?;
            Ok(())
        }
        None => {
            let mut stdout = io::stdout();
            let mut stdout_handle = stdout.lock();
            stdout_handle.write(&formatted.as_bytes())?;
            stdout_handle.write(b"\n")?;
            Ok(())
        }
    }

}

pub fn write_vec<T:Debug>(input: Vec<T>,target: &Option<String>) -> Result<(),Error> {
    let formatted =
        input
        .iter()
        .map(|x| format!("{:?}",x))
        .collect::<Vec<String>>()
        .join("\n");

    eprintln!("Formatted:{:?}",formatted);

    match target {
        Some(location) => {
            let mut target_file = OpenOptions::new().create(true).append(true).open(location).unwrap();
            target_file.write(&formatted.as_bytes())?;
            target_file.write(b"\n")?;
            Ok(())
        }
        None => {
            let mut stdout = io::stdout();
            let mut stdout_handle = stdout.lock();
            stdout_handle.write(&formatted.as_bytes())?;
            stdout_handle.write(b"\n")?;
            Ok(())
        }
    }

}

pub fn cosine_distance_matrix(slice: MatrixSlice<f64>) -> Matrix<f64> {
    let mut products = slice * slice.transpose();
    eprintln!("Products");
    let mut geo = slice.elemul(&slice).sum_cols();
    eprintln!("geo");
    geo = geo.apply(&|x| x.sqrt());

    for i in 0..slice.rows() {
        for j in 0..slice.rows() {
            products[[i,j]] /= (&geo[i] * &geo[j])
        }
    }

    products
}
//
// fn tsv_format<T:Debug>(input:&Vec<Vec<T>>) -> String {
//
//     input.iter().map(|x| x.iter().map(|y| format!("{:?}",y)).collect::<Vec<String>>().join("\t")).collect::<Vec<String>>().join("\n")
//
// }










//
