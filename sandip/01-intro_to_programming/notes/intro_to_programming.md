# 📘 Level 1 — Intro to Programming

## Task 1.1: Verify Python Installation

- [x] Open terminal
- [x] Run `python --version` or `python3 --version`
- [x] Write down the version number
- [x] Run `pip --version`
- [x] Screenshot or note both versions

```code
$ python --version
Python 3.14.3

$ pip --version
pip 25.1.1
```

**what is pip?**

- pip is a package installer for python. Basically, it lets you easily install, upgrade, and manage python libraries and dependencies. so, if you need a package say like numbPy or Request you just run a pip install command, and it handles the rest.

---

**what are common libraries and dependencies that is used in ai development?**

- NumPy for numerical operations
- Pandas for data manupulations.
- Matplotlib or Seaborn for visualization.
- Scikit-learn for machine learning.
- request fro web requests.

**Is there a search for easier installation of packages and dependencies?**

- `pip search <keyword>` // this is deprecated in newer versions of pip, but can use `pip index` as a workaround for searching.

**How do you get in and out of .venv?**

### commands

- source .venv/bin/activate
- deactivate

**what is .venv and why we need it?**

- .venv is just a directory that acts as a sandbox for your project. It keeps all the python dependencies and executables isolated, so different projects don't interfere with each other. it's super important because it ensures that you have a predictable environement same package versions, same python, all project specific. keeps your dev workflow clean and reproducible.

**Does .venv then lock the version1?**

- the venv itself isolates the environement, and if you pair it with a requirements.txt file. you can lock exact versions. Once you install packages, you can `run pip freeze > requirements.txt`, capturing all versions. then anyone else can recreate that exact environment by installing from that file.

### Commands

- `pip freeze > requirements.txt`
- `pip install -r requirements.txt`

- numpy==2.4.2: numpy is all about efficient numerical operations
- pandas==3.0.1: pandas is for data manipulations think data frames reading CSVs and analysis
- python-dateutil==2.9.0.post0: python dateutil is a utility for parsing dates easily extending python's standard datetime
- six==1.17.0: six is a compatibility library that lets you write code that works in both python 2 and python 3.
