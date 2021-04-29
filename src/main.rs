use clap::{AppSettings, Clap};
use scraper::scrape;
use split_to_json::split;
use std::fs::create_dir_all;
use std::process::Command;

/// Cobi Cli tool is a reusable interface for machine learning tools
/// related to Congressional Bills. Starting with "layout" will give an overview
/// of the output that can be expected.
/// Scrape should be run first, then C or F commands (as shown in the help) in the order
/// that they're listed (C1, or F1, F2, F3, F4...)
#[derive(Clap)]
#[clap(version = "1.0", author = "Dan Thompson <danthompson41@gmail.com>")]
#[clap(setting = AppSettings::ColoredHelp)]
#[clap(setting = AppSettings::SubcommandRequiredElseHelp)]
struct Opts {
    #[clap(subcommand)]
    subcmd: SubCommand,
}

#[derive(Clap)]
enum SubCommand {
    Scrape(Scrape),
    FormatterSplit(FormatterSplit),
    FormatterToHtml(FormatterToHtml),
    FormatterToMd(FormatterToMd),
    FormatterTrain(FormatterTrain),
    FormatterFormat(FormatterFormat),
    Clustering(Clustering),
    Layout(Layout),
}

/// 1: Download the bills from Congressional website
#[derive(Clap)]
struct Scrape {}

/// F1: Split the downloaded bills for formatting
#[derive(Clap)]
struct FormatterSplit {}

/// F4: Convert split bills to HTML
#[derive(Clap)]
struct FormatterToHtml {}

/// F4: Convert split bills to Markdown
#[derive(Clap)]
struct FormatterToMd {}

/// F2: Train the Formatting Neural Network
#[derive(Clap)]
struct FormatterTrain {
    /// Path to the saved weights to use for formatting
    model_weights_path: Option<String>,
}

/// F3: Run the model with specified weights
#[derive(Clap)]
struct FormatterFormat {
    /// Path to the saved weights to use for formatting
    model_weights_path: String,
}

/// C1: Train the clustering algorithm on downloaded bills
#[derive(Clap)]
struct Clustering {}

/// Show the layout  of the  files
#[derive(Clap)]
struct Layout {}

fn set_up_files() {
    let _ = create_dir_all("data/1_scraping_metadata");
    let _ = create_dir_all("data/2_raw_text");
    let _ = create_dir_all("data/3_formatting/0_model_checkpoints");
    let _ = create_dir_all("data/3_formatting/1_unlabeled");
    let _ = create_dir_all("data/3_formatting/2_training");
    let _ = create_dir_all("data/3_formatting/3_output");
    let _ = create_dir_all("data/3_formatting/4_html_output");
    let _ = create_dir_all("data/3_formatting/4_md_output");
    let _ = create_dir_all("data/4_clustering/");
}

fn main() {
    set_up_files();
    let opts: Opts = Opts::parse();

    // You can handle information about subcommands by requesting their matches by name
    // (as below), requesting just the name used, or both at the same time
    match opts.subcmd {
        SubCommand::Scrape(_) => {
            println!("Running Scraper...");
            let _ = scrape("data/2_raw_text", "data/1_scraping_metadata/cache.json");
        }
        SubCommand::FormatterSplit(_) => {
            println!("Running Splitter...");
            split("data/2_raw_text", "data/3_formatting/1_unlabeled").unwrap();
        }
        SubCommand::FormatterToHtml(_) => {
            println!("Formatting HTML...");
            Command::new("poetry")
                .current_dir("python/formatting/")
                .arg("run")
                .arg("python")
                .arg("formatterHtml.py")
                .status()
                .expect("process failed to execute");
        }
        SubCommand::FormatterToMd(_) => {
            println!("Formatting Markdown...");
            Command::new("poetry")
                .current_dir("python/formatting/")
                .arg("run")
                .arg("python")
                .arg("formatterMD.py")
                .status()
                .expect("process failed to execute");
        }
        SubCommand::FormatterTrain(args) => {
            println!("Training Formatter...");
            match args.model_weights_path {
                Some(path) => {
                    Command::new("poetry")
                        .current_dir("python/formatting/")
                        .arg("run")
                        .arg("python")
                        .arg("formatterTraining.py")
                        .arg(path)
                        .status()
                        .expect("process failed to execute");
                }
                None => {
                    Command::new("poetry")
                        .current_dir("python/formatting/")
                        .arg("run")
                        .arg("python")
                        .arg("formatterTraining.py")
                        .status()
                        .expect("process failed to execute");
                }
            }
        }
        SubCommand::FormatterFormat(args) => {
            println!("Using weights from {:?}", args.model_weights_path);
            println!("Formatting splits...");
            Command::new("poetry")
                .current_dir("python/formatting/")
                .arg("run")
                .arg("python")
                .arg("formatterRun.py")
                .arg(args.model_weights_path)
                .arg("&")
                .status()
                .expect("process failed to execute");
        }
        SubCommand::Clustering(_) => {
            println!("Clustering Bills...");
            Command::new("poetry")
                .current_dir("python/clustering/")
                .arg("run")
                .arg("python")
                .arg("clustering.py")
                .arg("&")
                .status()
                .expect("process failed to execute");
        }
        SubCommand::Layout(_) => {
            let output = r#"
 ----------------------------     
 - Data Layout - 
 ----------------------------
  - data/                            <- Top level output directory 

  | - 1_scraping_metadata            <- Files related to scraping
  | | - cache.json                      - Cache of scraped pages

  | - 2_raw_text/                    <- Raw bill text from scraper
  | | - <billNumbera>_text.txt          - Individual bills, one per file
  | | - <billNumberb>_text.txt 
  | | - <billNumberc>_text.txt
  | | ...

  | - 3_formatting/                  <- Formatting project data
  | | - 0_model_checkpoints/            <- Saved checkpoints from running the formatting model
  | |   ...
  | | - 1_unlabeled/                 <- Raw data, automatically created from text 
  | | | - <billNumbera>_split.json      - Unlabeled json splits, one bill per file
  | | | - <billNumberb>_split.json
  | | | - <billNumberc>_split.json
  | |   ...
  | | - 2_training/                  <- Training data, manually created from splits
  | | | - <billNumbera>_split.json     - Manually labeled json splits, one bill per file
  | | | - <billNumberb>_split.json
  | | | - <billNumberc>_split.json
  | |   ...
  | | - 3_output/
  | | | - <billNumbera>_split.json     - Algorithm labeled json data, one bill per file
  | | | - <billNumberb>_split.json    
  | | | - <billNumberc>_split.json
  | |   ...
  | | - 4_html_output/
  | | | - <billNumbera>_text.html      - Generated HTML, one bill per file
  | | | - <billNumberb>_text.html
  | | | - <billNumberc>_text.html
  | |   ...
  | | - 4_md_output/
  | | | - <billNumbera>_text.md        - Generated Markdown, one bill per file
  | | | - <billNumberb>_text.md
  | | | - <billNumberc>_text.md
  | |   ...

  | - 4_clustering/                  <- Clustering project
  | | - 1_cluster_topics.json           -  JSON map of topic clusters, and relevant top words
  | | - 2_cluster_map.json              -  Map of individual bills to clusters

"#;
            println!("{}", output);
        }
    }

    // more program logic goes here...
}
