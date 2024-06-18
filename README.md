# Source code for post: Libpostal, Reborn!

<div style="float:left;margin-right: 10px" markdown="1">
<img src="images/libpostal-logo.png" align="left" style="float: left; margin-right: 10px;"/>
</div>

This repository contains the source code for the blog post [Libpostal, Reborn!]() Use it to reproduce the results show in the post as a jumping off point to adopting Libpostal.

There are two ways to run the code - Docker or local Python 3. The Docker method is easier, but the local Python 3 method will work with Visual Studio Code and Apple MPS GPUs.

## Running the Code

There is a Docker environment that runs Jupyter Lab so you can run the code a notebook. There are unit tests in [tests.py](tests.py) file for running with `pytest`.

## Pre-Requisites

To run the code in this repository, you need to have the following software installed.

- Docker - you can [install docker](https://docs.docker.com/engine/install/) and then check the [Get Started](https://www.docker.com/get-started/) page if you aren't familiar.

OR

- Python 3 - you can download Anaconda Python here: [https://www.anaconda.com/download](https://www.anaconda.com/download)

Supported operating systems:

- Mac OS X
- Ubuntu Linux
- Other Linux will work, but you may need to adjust the instructions for Fedora, CentOS, etc.

Windows is not supported. I'd appreciate Windows support as a contribution!

- Instructions for installing Libpostal are below.

## Running the Code with Jupyter on Docker

You can run Jupyter in Docker with the following command:

```bash
docker compose up -d
```

Note: _I have disabled tokens and passwords as this is a test environment. You should not do this in production._

Now open your browser and go to [http://localhost:8888/lab](http://127.0.0.1:8888/lab) to access Jupyter.

To shut down the Docker container, run:

```bash
docker compose down
```

## Development Environment Setup

If you're building software and want to experiment and explore this code and its dependencies, you can load the code in Visual Studio Code or another editor. I created `pytest` tests that reproduce the blog post's results in [`distance.py`](distance.py).

### Python Virtual Environment

We use a Python virtual environment to run the code in this repository in a local environment. This will help you avoid conflicts with other Python projects you may have on your system.

#### Anaconda Python

If you are using Anaconda Python, create a new conda environment named `libpostal` using the following command:

```bash
conda create -n libpostal python=3.11 -y
```

Then activate the environment:

```bash
conda activate libpostal
```

#### Other Python

You can use a Python venv environment to run the code in this repository. To create a new virtual environment named `libpostal`, use the following command:

```bash
python3 -m venv libpostal
```

Then activate the environment:

```bash
source libpostal/bin/activate
```

## Installing Libpostal

Libpostal installation instructions are available on the [Libpostal GitHub page](https://github.com/senzing/libpostal?tab=readme-ov-file#installation-maclinux).

### Install Prerequisites

Libpostal is a C library which we will build from source. To build it, you need to install the prerequisites below for your operating system.

#### Mac OS X

```bash
brew install curl autoconf automake libtool pkg-config
```

#### Ubuntu Linux

```bash
sudo apt-get install -y curl build-essential autoconf automake libtool pkg-config
```

### Build Libpostal

> If you're using an M1 Mac or ARM64-based linux distro, add --disable-sse2 to the ./configure command. This will result in poorer performance but the build will succeed.

```bash
git clone https://github.com/openvenues/libpostal
cd libpostal
./bootstrap.sh

# For Intel/AMD cpus, with Senzing model:
./configure --datadir=/tmp MODEL=senzing

# For Apple / ARM cpus, with Senzing model:
./configure --datadir=/tmp --disable-sse2 MODEL=senzing

make -j4
sudo make install

# On Linux it's probably a good idea to run
sudo ldconfig
```

## Installing Python Dependencies

The project uses [Python Poetry](https://python-poetry.org/) for package management. We use it to install the Libpostal Python wrapper [pypostal](https://github.com/openvenues/pypostal) from PyPI under the name [postal](https://pypi.org/project/postal/).

### Install Poetry

To install Poetry (see [install docs](https://python-poetry.org/docs/)), you can use [pipx](https://github.com/pypa/pipx) or `curl` a script.

#### Install Poetry Using `curl`

To install using `curl`, just run:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### Install Poetry using `pipx`

To first install `pipx`, on Mac OS X run:

```bash
brew install pipx
pipx ensurepath
sudo pipx ensurepath --global # optional to allow pipx actions with --global argument
```

On Ubuntu Linux, run:

```bash
sudo apt update
sudo apt install pipx
pipx ensurepath
sudo pipx ensurepath --global # optional to allow pipx actions with --global argument
```

Then install Poetry using `pipx`:

```bash
pipx install poetry
```

It will now be outside your virtual environment and available to all your Python projects.

### Install Dependencies

Now we can use `poetry` to install our Python dependencies. Run the following command:

```bash
poetry install
```

And our setup is complete!

## Running the Code

The code relies on a series of pytests in [`tests.py`](tests.py). To run the code, use the following command:

```bash
pytest
```
