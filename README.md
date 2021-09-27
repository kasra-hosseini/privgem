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

- :warning: [Credits](#credits)
- :building_construction: [Installation and setup](#installation)
- :student: [Tutorials](./examples) are organized in Jupyter Notebooks as follows:
    - **Tabular data**
        - [PATE-CTGAN](./examples/PATE-CTGAN_example_001.ipynb)
        - [DP-CTGAN](./examples/DP-CTGAN_example_001.ipynb)
        - [CTGAN](./examples/CTGAN_example_001.ipynb)
        - [PGM and PATE-CTGAN](./examples/artificial_and_synthetic_data.ipynb), using generated/artificial n-class classification problem


## Credits

`privgem` uses codes from other libraries as listed below.
⚠️ Please read the list <ins>carefully and cite the original</ins> codes/papers as well.

| Method      | Original version                                                      | Notes                                                                                                                                   |
|-------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| PATE-CTGAN  | [smartnoise-sdk](https://github.com/opendp/smartnoise-sdk)            | `tabular_patectgan` of `privgem` is based on `smartnoise-sdk` with  some minor changes (e.g., data preproc., logging, plotting and etc). |
| DP-CTGAN    | [smartnoise-sdk](https://github.com/opendp/smartnoise-sdk)            | `tabular_dpctgan` of `privgem` is based on `smartnoise-sdk` with  some minor changes (e.g., data preproc., logging, plotting and etc).   |
| Private-PGM    | [private-data-generation](https://github.com/BorealisAI/private-data-generation)            | `tabular_ppgm` of `privgem` is based on [private-data-generation](https://github.com/BorealisAI/private-data-generation) with some minor changes.   |
| CTGAN       | [sdv-dev](https://github.com/sdv-dev/CTGAN)                           | currently, `privgem` uses a forked version of `ctgan`, [link](https://github.com/kasra-hosseini/CTGAN).                                  |


## Installation

We strongly recommend installation via Anaconda:

* Refer to [Anaconda website and follow the instructions](https://docs.anaconda.com/anaconda/install/).

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

```bash
pip install -r requirements.txt
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
