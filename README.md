<div align="center">
    <br>
    <p align="center">
    <h1>privgem</h1>
    </p>
    <h2>Privacy-Preserving Generative Models</h2>
</div>

<p align="center">
    <a href="https://github.com/kasra-hosseini/privgem/workflows/Continuous%20integration/badge.svg">
        <img alt="Continuous integration badge" src="https://github.com/kasra-hosseini/privgem/workflows/Continuous%20integration/badge.svg">
    </a>
    <br/>
</p>


- [Credits](#credits)
- [Installation and setup](#installation)
  - [Method 1: Anaconda + install dependencies using `environment.yaml` file](#method-1)
  - [Method 2: Anaconda + install dependencies manually](#method-2)

## Credits

`privgem` uses codes from other libraries as listed below.
⚠️ Please read the list <ins>carefully and cite the original</ins> codes/papers as well.

| Method     	| Original version                                          	| Notes                                       	|
|------------	|------------------------------------------------------------	|---------------------------------------------	|
| PATE-CTGAN 	| [smartnoise-sdk](https://github.com/opendp/smartnoise-sdk) 	|                                             	|
| CTGAN      	| [sdv-dev](https://github.com/sdv-dev/CTGAN)                	| currently, `privgem` uses `ctgan==0.2.2.dev1` |

    
## Installation

We strongly recommend installation via Anaconda:

* Refer to [Anaconda website and follow the instructions](https://docs.anaconda.com/anaconda/install/).

### Method 1

* Create a conda environment and install the dependencies using `environment.yaml` file:

```bash
conda env create --file environment.yaml python=3.8 --name privgem_py38
```

* Activate the environment:

```bash
conda activate privgem_py38
```

* Clone `privgem` source code:

```bash
git clone https://github.com/kasra-hosseini/privgem.git
```

* Finally, install `privgem` library:

```
cd /path/to/privgem
pip install -v -e .
```

Alternatively:

```
cd /path/to/privgem
python setup.py install
```

* To allow the newly created `privgem_py38` environment to show up in Jupyter Notebook:

```bash
python -m ipykernel install --user --name privgem_py38 --display-name "Python (privgem_py38)"
```

### Method 2

* Create a new environment for `privgem` called `privgem_py38`:

```bash
conda create -n privgem_py38 python=3.8
```

* Activate the environment:

```bash
conda activate privgem_py38
```

* Clone `privgem` source code:

```bash
git clone https://github.com/kasra-hosseini/privgem.git
```

* Install `privgem` dependencies:

```
# Install dependencies
pip install ctgan==0.2.2.dev1
pip install opacus==0.9.0
pip install torch==1.6.0
pip install jupyterlab
pip install matplotlib
```

* Finally, install `privgem` library:

```
cd /path/to/privgem
pip install -v -e .
```

Alternatively:

```
cd /path/to/privgem
python setup.py install
```

* To allow the newly created `privgem_py38` environment to show up in Jupyter Notebook:

```bash
python -m ipykernel install --user --name privgem_py38 --display-name "Python (privgem_py38)"
```
