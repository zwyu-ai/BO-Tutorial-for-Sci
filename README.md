# Installation with `uv`

First, clone the repo [BO-Tutorialfor-Sci](https://github.com/zwyu-ai/BO-Tutorial-for-Sci) and navigate to it.

Then manage dependencies with `uv` as a good practice

```shell
cd $HOME/path/to/BO-Tutorial-for-Sci
uv init
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

# Run

```shell
python3 BO_tutorial.py
```

# Contribute

To add new datasets (e.g. HEA), create functions inside `HEA/utils.py` similar to the ones in other dataset directories.
Add them to the BO_tutorial.py script and to the final plot.