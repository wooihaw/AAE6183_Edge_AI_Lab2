# Setup Guidelines
## Installing uv
uv is a fast Python package and project manager written in Rust. It can be installed on Windows using a standalone installer via PowerShell.
1. Open PowerShell as an administrator. You can do this by searching for "PowerShell" in the Start menu, right-clicking the application, and selecting Run as administrator.
2. In the PowerShell window, run the following command to download and execute the installer script:
```
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
1. Once the installation is complete, close the PowerShell window and open the terminal to ensure that the uv command is available in your system's PATH.
2. Verify the installation by running:
```
uv --version
```

## Installing Git
You can install Git for Windows using either the winget command-line tool or the standalone installer.  
1. Open Command Prompt or PowerShell.
2. Run the following command to install Git:
```
winget install --id Git.Git -e --source winget
```
3. The installer will run and complete the installation.

## Cloning Repository
1. Open the terminal and navigate to the desired folder.
2. Clone this repository by entering:
```
git clone https://github.com/wooihaw/AAE6183_Edge_AI_Lab2.git
```

## Restoring Environment and Synchronizing Dependencies
1. Open the terminal and navigate into the cloned repository folder.
2. Enter the following command to restore the environment and synchronize dependencies:
```
uv sync
```

## Running Python Scripts
1. Open the terminal and navigate into the cloned repository folder.
2. Run each Python script sequentially using the instructions below:
```
uv run benchmark_tf.py
uv run benchmark_onnx.py
uv run benchmark_ov.py
```

### Intel OpenVINO GPU Performance Profiling
1. Open the benchmark_openvino.py script in a text editor.
2. Locate the line of code responsible for compilation:
```
compiled_model = core.compile_model(model=ov_model, device_name="CPU")
```
3. Modify the target device string parameter from "CPU" to "GPU":
```
compiled_model = core.compile_model(model=ov_model, device_name="GPU")
```
4. Save the script files.

5. Run the modified script in your terminal:
```
uv run benchmark_ov.py
```