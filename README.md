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

# Running Python components without Rust

All of the python scripts included in the repository rely on the data pipeline structure layed out by running the `cobi` binary at the top level. If not using the `cobi` utility, you can download an example data set. Unzip it, and place it with the name 'data' at the top level of this repository.

Example data set: https://drive.google.com/file/d/1Qc68zHth6MD8a3EmQkIEpa1eA8AeGOfp/view?usp=sharing

If you'd like to run the python programs used without installing Rust, follow the above instructions for installing python dependencies. The following files and their purposes are the relevant python scripts:

- **python/clustering/clustering.py**                : Runs HDBScan and BERT encoding on full contents of the raw bill files
- **python/clustering/BERT Clustering Bills.ipynb**  : Notebook for interactively viewing the clustering algorithm


- **python/formatting/formatterTraining.py**         : Trains the formatter, outputting the weights to ../../data/3_formatting/0_model_checkpoints 

- **python/formatting/formatterRun.py**              : Runs the formatter, which needs to be passed weights from ../../data/3_formatting/0_model_checkpoints

- **python/formatting/formatterMD.py**               : Format the splits that were labeled by formatterRunner, outputing MD files

- **python/formatting/formatterHtml.py**               : Format the splits that were labeled by formatterRunner, outputing HTML files
