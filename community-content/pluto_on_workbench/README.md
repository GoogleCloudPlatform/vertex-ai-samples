# Overview
*Pluto* is a programming environment for Julia, designed to be interactive and helpful. It provides a familiar notebook interface but it is not a Jupyter notebook. The biggest difference is that Pluto notebooks are reactive, changing a variable or function in one cell causes the cells that depend on that variable or function to be reevaluated. Pluto also provides useful interaction mechanisms that allow users to dynamically interact with the notebooks computation state.

The JuliaCon 2020 presentation: [Interactive notebooks ~ Pluto.jl]() provides a good introduction to Pluto. The source is at [fonsp/Pluto.jl]()

# Install Pluto

## Create a Vertex AI JupyterLab Instance

1. From the [GCP console](https://console.cloud.google.com) "hamburger menu"

   select Vertex AI > Workbench
2. Click NEW NOTEBOOK
   
   * Choose Python 3 if you won't be using a GPU
   * Choose Python 3 (CUDA Toolkit xx.y) if you do want use a GPU
3. Give the notebook an appropriate name
4. Edit Notebook properties if you have special requirements otherwise accept the defaults and click CREATE
5. When the notebook instance is ready click OPEN JUPYTERLAB

## Configure JupyterLab

1. Open a terminal by clicking the Terminal icon.
1. Install the plutoserver
pip3 install git+https://github.com/fonsp/pluto-on-jupyterlab.git
1. In a browser go to [julialang.org/downloads](https://julialang.org/downloads/)
1. In the Current stable release right click on the `Generic Linux on x86 / 64-bit (glibc)` link
Select copy link address
1. Back in the terminal switch to root via
sudo -i
1. Download the release to /opt and install julia in /usr/local/bin
    ```bash
    cd /opt
    wget <paste the release link address>
    tar xf <name of the downloaded tar file>
    ln -s /opt/<julia-x.y.z>/bin/julia /usr/local/bin
    ^d
    ```
1. Add the Pluto package to Julia
   ```bash
    julia
    julia> ]add Pluto
    julia> bksp
    julia> using Pluto
    julia> ^d
    ```
1. From the JupyterLab menu bar select File > Shut Down

# Start Pluto
1. Click OPEN JUPYTERLAB in the Workbench
1. In the Notebook section of the Launcher click Pluto.jl
1. The welcome to Pluto.jl screen should appear
