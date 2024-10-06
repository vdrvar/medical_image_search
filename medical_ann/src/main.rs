use std::error::Error;
use std::fs::File;
use csv::ReaderBuilder;
use ndarray::{Array2, Array};

fn load_embeddings(file_path: &str) -> Result<Array2<f32>, Box<dyn Error>> {
    // Open the CSV file
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);

    // Create a vector to store the data
    let mut records = vec![];
    for result in rdr.records() {
        let record = result?;
        let parsed_record: Result<Vec<f32>, _> = record.iter().map(|s| s.parse::<f32>()).collect();
        records.push(parsed_record?);
    }

    // Get the number of rows and columns
    let rows = records.len();
    let cols = if rows > 0 { records[0].len() } else { 0 };
    // Flatten the records and create an Array2
    let flattened: Vec<f32> = records.into_iter().flatten().collect();
    let array = Array::from_shape_vec((rows, cols), flattened)?;

    Ok(array)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Specify the CSV file path
    let file_path = "../embeddings/COVID_embeddings.csv";
    
    // Load the embeddings
    let embeddings = load_embeddings(file_path)?;
    println!("Loaded embeddings with shape: {:?}", embeddings.dim());
    
    Ok(())
}
