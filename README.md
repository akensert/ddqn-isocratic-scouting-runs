# Double Deep Q-learning for optimal selection of isocratic scouting runs

### Requirements
* Python >= 3.6
* TensorFlow >= 2.0
* Pip (package manager)
* Third party Python packages found in `requirements.txt`. To install these packages (including TensorFlow), run from terinal: `pip install -r requirements.txt`


### How to run
To train the agent, navigate into `src/` and run from terminal: `python train.py`. The training will take about 30-90 min.

To utilize the trained agent to select scouting runs, run from terminal: `python test.py`. This program is just a toy application, but illustrates how the agent could potentially be used to select isocratic scouting runs in practice.
