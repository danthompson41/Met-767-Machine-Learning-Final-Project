# Congressional Bill Data Analysis
Dan Thompson, BU Met 767 Machine Learning Spring 2021

# How to build the binary
This project is based around the "cobi" binary, which can be built from scratch using Cargo / Rust.
Instructions for installing Cargo are here: https://doc.rust-lang.org/cargo/getting-started/installation.html

Once cargo is installed, the binary can be build by running `cargo build --release`, which places it in target/release/cobi. The `cobi` binary in this repository (built for linux) is the output of `cargo build --release` on my machine

Alternatively, it can be run by running `cargo run` with subcommands after

The main utility is responsible for fetching new bills, splitting them for use with the formatter, and running the various python utilities included in the project

### Note about relative paths: Because this script calls python code, it relies on a specific folder layout. Only run `cargo run` or the `cobi` binary from this top level folder. 

# How to install the python dependencies
The project uses https://python-poetry.org/ to install and manage python dependencies. "poetry install" should be run in the following folders before using the cobi binary:
- python/formatting
- python/clustering

If you're not using poetry, the specific versions used for testing are listed in the pyproject.toml files in those folders. Even more precise versions are listed in the poetry.lock files

Generally speaking, Python "3.8", Numpy "1.19.2", and Tensorflow "2.4.1" were found to be compatable and used together (on linux)

# Example use of the binary:
  After running the above poetry installation, and in the top level folder:
  
```
cargo run layout # Show visual representation of the data layout
cargo run scrape # Download raw text of bills bills

# Format the raw text of the bills
cargo run formatter-split # Generate split JSON representations of the raw data for formatting
cargo run formatter-train # Train the formatter neural network, using the pre-labeled bills in data/3_formatting/2_training
cargo run formatter-format ../../data/3_formatting/0_model_checkpoints/cli-weights-improvement-90.hdf5 # Run formatter using an output from the training segment. This particular one is included in the example data linked below
cargo run formatter-to-md # Generate Markdown output from the splits labeled by the neural network
cargo run formatter-to-html # Generate HTML output from the splits labeled by the neral network

# Cluster the raw data
cargo run clustering # Runs the HDBScan based clustering algorithm
```


# Running Python components without Rust

All of the python scripts included in the repository rely on the data pipeline structure layed out by running the `cobi` binary at the top level. If not using the `cobi` utility, you can download an example data set. Unzip it, and place it with the name 'data' at the top level of this repository.

Example data set: https://drive.google.com/file/d/1Qc68zHth6MD8a3EmQkIEpa1eA8AeGOfp/view?usp=sharing
This data set is rather large (~233 MB compressed).

If you'd like to run the python programs used without installing Rust, follow the above instructions for installing python dependencies. The following files and their purposes are the relevant python scripts:

- **python/clustering/clustering.py**                : Runs HDBScan and BERT encoding on full contents of the raw bill files
- **python/clustering/BERT Clustering Bills.ipynb**  : Notebook for interactively viewing the clustering algorithm


- **python/formatting/formatterTraining.py**         : Trains the formatter, outputting the weights to ../../data/3_formatting/0_model_checkpoints 

- **python/formatting/formatterRun.py**              : Runs the formatter, which needs to be passed weights from ../../data/3_formatting/0_model_checkpoints

- **python/formatting/formatterMD.py**               : Format the splits that were labeled by formatterRunner, outputing MD files

- **python/formatting/formatterHtml.py**               : Format the splits that were labeled by formatterRunner, outputing HTML files
