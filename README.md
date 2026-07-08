# Global Flood Model (GFM)

The Global Flood Model (GFM) is an extension of the research conducted in the [Aqueduct Coastal Flooding](https://github.com/Deltares-research/aqueduct-coastal-flooding) project. This model is designed to simulate and analyze global flood scenarios effectively.

## Prerequisites

Before you begin, you must have the following tools installed on your system:

1.  **Git:** For cloning the repository.
2.  **Pixi:** For managing the project environment. Follow the official [Pixi installation guide](https://pixi.sh/latest/installation/).
3.  **C++ Build Tools (Windows Users Only):** If you are on Windows, you must install the Visual Studio Build Tools.
    * Download the [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio).
    * Run the installer and select the **"Desktop development with C++"** workload.

## Installation and Setup

The project environment is fully managed by Pixi. The following commands will install all necessary tools (Python, Julia, Rust) and dependencies.

1.  **Clone the repository:**
    ```shell
    git clone https://github.com/Deltares-research/GFM.git
    ```

2.  **Navigate into the project directory:**
    ```shell
    cd GFM
    ```

3.  **Install the environment:**
    This single command reads the `pixi.lock` file and installs the correct versions of all tools and packages.
    ```shell
    pixi install
    ```

## Usage

### 1. Build the Executable

Compile the Julia and Rust source code into the main `aqueduct.exe` application. This only needs to be done once after installation or after making code changes.

```shell
pixi run build
```

### 2. Prepare Input Data (Optional)

If you need to prepare new input files for a simulation, run the preprocessing script.

```shell
python python/preprocessing.py
```

### 3. Run a Simulation
   To run a simulation, call the compiled executable `aqueduct.exe` and provide the path to a `.toml` configuration file. For example, to test the standard Ireland case, execute:
   ```shell
   .\build\aqueduct\aqueduct.exe P:\11210264-004-global-flood-modellin\data\output\ireland\aqueduct_rp1_slr_500.toml
   ```

   Alternative to run GFM:
   You can also run GFM using python by either opening `python\run_gfm.py` and using the command `output, error = run_aqueduct_simulation(exe_path, toml_config_path)` OR by directly calling `python python\run_gfm.py` (Note: Ireland example)


## Example: Results for Ireland 

![image](https://github.com/Deltares-research/GFM/assets/45360568/28ef0f96-ae56-4735-b323-79ffc74dd475)
No flood
 
![image](https://github.com/Deltares-research/GFM/assets/45360568/63aab624-1328-4e94-b229-494863aec0f5)
RP1 - 0.5m SLR
 
![image](https://github.com/Deltares-research/GFM/assets/45360568/e9f562dc-e44d-4e6e-835c-8c97f0b86266)
RP100 - 3m SLR
