# Topological-localization-machine-learning
Master's thesis by Mateus Silva in the postgraduate program in electrical engineering at UNIFEI Itabira campus.

## Description
Topological localization system for smart vehicles using a **two-stage deep learning pipeline**.

The system identifies the vehicle position in a map of predefined nodes:

- **Model 1:** detects `straight` vs `intersection`
- **Model 2:** classifies the intersection into one of the nodes (16 classes)

## Repository structure (summary)
- main.py — application entry point.  
- gui.py — demonstration interface.  
- src/ — main package with modules: capture, gps, gui, Models, Navegation, tracker, utils.  
- src/Models/ — trained weights and modeling scripts.  
- data/ — logs and images used for training/testing.  
- maps/ — map configuration files.
- Validation/ — logs, mages, and videos used for field validation of the complete pipeline.
- notebooks — experiments and analyses (see the .ipynb files listed above).

## Quick installation
1. Create and activate a Python 3.10 environment (recommended).
2. Install dependencies:
   pip install -r requirements.txt

## Execution
- Graphical interface: run [gui.py](gui.py).  
- To reproduce experiments and train models, refer to the notebooks: [Model_training_1.ipynb](Model_training_1.ipynb) and [Model_training_2.ipynb](Model_training_2.ipynb). The model-building function is in [`build_model`](Final_Model.ipynb).

## Author information and contact details
Mateus Silva<br>
MSc Student in Electrical Engineering – UNIFEI<br>
Automation & Control Engineer<br>
Phone: +55 31 9 8818-8696<br>
E-mails: mateusfilipi22@unifei.edu.br | mateus.filipe.22@outlook.com