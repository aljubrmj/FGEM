# Installation

## Step 1: Create a Python Environment

If you're not sure how to create a suitable Python environment, the easiest way is using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 
following their [instructions](https://docs.conda.io/en/latest/miniconda.html). Then you can create and activate a new Python environment by running:

```
conda create -n my-package python=3.9
conda activate my-package
```

## Step 2: Install FGEM
Install FGEM as follows:

```bash
pip install git+https://github.com/aljubrmj/FGEM
```

## Step 3: Clone GitHub Repository [OPTIONAL]

Our Github repo comes with examples, which can be useful to users. To access those examples locally, users can clone our Github repo using HTTPS, SSH, or GitHub CLI. See [GitHub docs](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) for information on the different cloning methods. 
If you run into issues, follow GitHub troubleshooting suggestions [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/troubleshooting-cloning-errors#https-cloning-errors).

### Using HTTPS

```bash
$ git clone https://github.com/aljubrmj/FGEM.git
```

### Using SSH-Key

If it your first time cloning a **repository through ssh**, you will need to set up your git with an ssh-key by following these [directions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

```bash
$ git clone git@github.com:aljubrmj/FGEM.git
```

After cloning the repo, you can change directory to access the examples:

```bash
$ cd FGEM/examples
```

<!-- 
## Step 2: Clone GitHub Repository

Users can clone the repository using HTTPS, SSH, or GitHub CLI. See [GitHub docs](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) for information on the different cloning methods. 
If you run into issues, follow GitHub troubleshooting suggestions [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/troubleshooting-cloning-errors#https-cloning-errors).

### Using HTTPS

```bash
$ git clone https://github.com/aljubrmj/FGEM.git
```

### Using SSH-Key

If it your first time cloning a **repository through ssh**, you will need to set up your git with an ssh-key by following these [directions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

```bash
$ git clone git@github.com:aljubrmj/FGEM.git
```

## Step 3: Install Required Packages
Change to the cloned **FGEM** directory, and install the required Python packages:

```bash
cd FGEM
pip install -r requirements.txt
```
 -->