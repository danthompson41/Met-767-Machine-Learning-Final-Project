# Congressional Bill Data Analysis
Dan Thompson, BU Met 767 Machine Learning Spring 2021

# How to build the binary
This project is based around the "cobi" binary, which can be built from scratch using Cargo / Rust.
Instructions for installing Cargo are here: https://doc.rust-lang.org/cargo/getting-started/installation.html

Once cargo is installed, the binary can be build by running `cargo build --release`, which places it in target/release/cobi

Alternatively, it can be run by running `cargo run` with subcommands after

The main utility is responsible for fetching new bills, splitting them for use with the formatter, and running the various python utilities included in the project

# How to install the python dependencies
The project uses https://python-poetry.org/ to install and manage python dependencies. "poetry install" should be run in the following folders before using the cobi binary:
- python/formatting
- python/clustering

If you're not using poetry, the specific versions used for testing are listed in the pyproject.toml files in those folders. Even more precise versions are listed in the poetry.lock files

Generally speaking, Python "3.8", Numpy "1.19.2", and Tensorflow "2.4.1" were found to be compatable and used together (on linux)

# Running Python components without Rust

If you'd like to run the python programs used without installing Rust, follow the above instructions for installing python depedencies. The following files and their purposes are the relevant python scripts:

- **python/clustering/clustering.py**                : Runs HDBScan and BERT encoding on full contents of the raw bill files
- **python/clustering/BERT Clustering Bills.ipynb** : Notebook for interactively viewing the clustering algorithm


- **python/formatting/formatterTraining.py**         : Trains the formatter, outputting the weights to ../../data/3_formatting/0_model_checkpoints 

- **python/formatting/formatterTraining.py**         : Runs the formatter, which needs to be passed weights from ../../data/3_formatting/0_model_checkpoints

