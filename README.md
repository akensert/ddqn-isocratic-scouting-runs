## Double Deep Q-learning for optimal selection of isocratic scouting runs

### Requirements
* Python >= 3.6
* TensorFlow >= 2.0
* Pip (package manager)
* Third party Python packages found in `requirements.txt`. To install these packages (including TensorFlow), run from terinal: `pip install -r requirements.txt`


### How to run

#### 1. Training the agent
To train the agent, navigate into `src/` and run from terminal: `python train.py`. The training will take about 30-90 min. To train for fewer or more episodes add the flag `--num_episodes={number of episodes}` after `python train.py`. And to train the model on a GPU (if available) add the flag `--use_gpu=True`.

#### 2. Predicting with the agent
After the agent has been trained (`python train.py`), run from terminal `python test.py` to utilize the trained agent to select scouting runs for a compound. The model estimated at the end is the Neue-Kuss model, which models the behavior of the compound. This model can then be used to predict retention factors (k) for scouting runs with a different fraction of organic modifier (phi) in the mobile phase. Notice: This program is just a toy application; but illustrates how the agent could potentially be used to select isocratic scouting runs for compounds in practice. 
