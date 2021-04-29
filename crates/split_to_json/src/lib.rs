use regex::Regex;
use serde::Serialize;
use std::fs::File;
use std::io::prelude::*;
use std::vec::Vec;
use walkdir::WalkDir;

#[derive(Serialize, Debug)]
struct Bill {
    sections: Vec<Classifier>,
}
#[derive(Serialize, Debug)]
struct Classifier {
    content: String,
    classification: String,
}

pub fn split(input_path: &str, output_path: &str) -> Result<(), std::io::Error> {
    let re = Regex::new(r"\n\n|   ``|    \(").unwrap();
    let path_re = Regex::new(r"/|_").unwrap();
    for entry in WalkDir::new(input_path) {
        let value = entry?;
        println!("{:?}", value.path());
        if value.file_type().is_dir() {
            continue;
        }
        let mut file = File::open(value.path())?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let split = re.split(&contents);
        let mut list = Vec::new();
        for i in split {
            let a = Classifier {
                content: i.to_string(),
                classification: "PARAGRAPH".to_string(),
            };
            list.push(a);
        }
        let file = Bill { sections: list };
        let serialized = serde_json::to_string_pretty(&file).unwrap();
        let filename = value.path().file_name().unwrap();
        let name_split: Vec<&str> = path_re.split(filename.to_str().unwrap()).collect();
        println!("{:?}", name_split);
        let output = format!("{}/{}_split.json", output_path, name_split[0],);
        println!("Output: {}", output);
        let mut file_out = File::create(output)?;
        let _ = file_out.write(serialized.as_bytes());
    }
    Ok(())
}
