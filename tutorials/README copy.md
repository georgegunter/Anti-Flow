# Flow Tutorials

## Setup

1. Make sure you have Python 3 installed (we recommend using the [Anaconda
   Python distribution](https://www.continuum.io/downloads)).
2. **Install Jupyter** with `pip install jupyter`. Verify that you can start
   a Jupyter notebook with the command `jupyter-notebook`.
3. **Install Anti-Flow/Flow** by executing the following [installation instructions](
   https://flow.readthedocs.io/en/latest/flow_setup.html).

## Tutorials

Each file ``tutorials/tutorial*.ipynb`` is a separate tutorial. They can be
opened in a Jupyter notebook by running the following commands:

```shell
source activate flow
cd <Anti-Flow-path>/tutorials
jupyter-notebook
```

We recommend that users become familiar with the original Flow software package before making significant structural changes to Anti-Flow. The following tutorials demonstrate how a user can initialize and start attacking a given traffic scenario without needing to alter/understand the underlying Flow architecture.

The content of each tutorial is as follows:

**Tutorial 1:** Use existing attack model to attack a ring-road.

**Tutorial 2:** Use existing defense model to detect the attack.

**Tutorial 3:** Use existing defense model to detect the attack.
