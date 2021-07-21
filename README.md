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
- [Examples on how to run privgem](./examples)

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

Install privgem

```bash
pip install git+https://github.com/kasra-hosseini/privgem.git@develop
```

### Developer

Install using [poetry](https://python-poetry.org/):

```bash
poetry install
```


### Notebooks 
Create a kernel called `privgem` which can use the poetry environment

```bash
poetry run python -m ipykernel install --user --name privgem
```

