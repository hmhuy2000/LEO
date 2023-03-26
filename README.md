
<h1>Learning from Expert Observations (LEO)</h1>

This repository contains the code of the paper [Learning from Pixels with Expert Observations](https://sites.google.com/view/leo-rb) 

Project website:[https://sites.google.com/view/leo-rb](https://sites.google.com/view/leo-rb)

----

## Announcements

#### September 12, 2022
- <b>Official code published</b>
----
## Installation

1. Install [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

1. Clone this repo
    ```
    git clone https://github.com/hmhuy2000/LEO.git
    cd LEO
    ```
1. Create and activate conda environment
    ```
    conda create --name LEO python=3.7
    conda activate LEO
    ```
1. Install dependencies
    ```
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    pip install -r baseline_requirements.txt
    pip install .
    ```
1. Run demo

    Set base path in file [scripts/run_demo.sh](https://github.com/hmhuy2000/LEO/blob/main/scripts/run_demo.sh)
    ```
    ./scripts/run_demo.sh
    ```

## Arguments
### ```--env```: environment for training:
* block_stacking
* house_building_1
* house_building_2
* house_building_3
* house_building_4
### ```--use_equivariant```:
* True: use normal classifier
* False: use equivariant classifier
### ```--algorithm```: Algorithm for training
* DQN
* SDQfD
### ```--samples_per_class```: number of samples per class for collected dataset used for classifier
### ```--planner_episode```: number of expert episodes
### ```---max_train_step```: number of training steps
### ```--use_classifier```: 
* True: use our classifier
* False: Use perfect classifier
### ```---load_model_pre```: pretrained path for validation
### ```--train_phrase```: 
* True: training
* False: validation
## Usage

1. Collect data for classifier

    In file [scripts/run_collect_data.sh](https://github.com/hmhuy2000/LEO/blob/main/scripts/run_collect_data.sh):

    * Set PYTHONPATH
    * Set ```--env```
    * Set ```--samples_per_class```
    ```
    ./scripts/run_collect_data.sh
    ```

1. train classifier

    In file [scripts/run_train_classifier.sh](https://github.com/hmhuy2000/LEO/blob/main/scripts/run_train_classifier.sh):

    * Set PYTHONPATH
    * Set ```--env```
    * Set ```--use_equivariant```
    ```
    ./scripts/run_train_classifier.sh
    ```

1. train agent

    In file [scripts/run_train_agent.sh](https://github.com/hmhuy2000/LEO/blob/main/scripts/run_train_agent.sh):

    * Set PYTHONPATH
    * Set ```--env```
    * Set ```--use_equivariant```
    * Set ```--algorithm```
    * Set ```--planner_episode```
    * Set ```--max_train_step```
    * Set ```--use_classifier```
    * set ```--load_model_pre``` for validation
    * set ```--train_phrase```
    ```
    ./scripts/run_train_agent.sh
    ```
