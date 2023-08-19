
<h1>Learning from Pixels with Expert Observations</h1>

Code accompany [Learning from Pixels with Expert Observations](https://arxiv.org/pdf/2306.13872.pdf) 

Project website: [https://sites.google.com/view/leo-rb](https://sites.google.com/view/leo-rb)

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
    ```

## Arguments
![List of tasks](https://github.com/hmhuy2000/LEO/blob/8de1c5fa38878da40f91787fb11ae98f012e203c/all_tasks.png)
* ```--env```: Environment to train (block_stacking, house_building_1, house_building_2, house_building_3, house_building_4)
* ```--use_equivariant```: True (use non-equivariant state abstractor), False (use equivariant state abstractor)
* ```--algorithm```: Algorithm for training (DQN, SDQfD)
* ```--samples_per_class```: Number of samples per class for collected dataset used for classifier
* ```--planner_episode```: Number of expert episodes
* ```--max_train_step```: Number of training steps
* ```--load_model_pre```: Pretrained path for validation
* ```--train_phase```: True (training) or False (validation)

## Usage

1. Collect data for training state abstractor
    ```
    ./scripts/run_collect_data.sh
    ```

1. Train state abstractor
    ```
    ./scripts/run_train_classifier.sh
    ```

1. Train agent
    ```
    ./scripts/run_train_agent.sh
    ```

1. Evaluate agent
    ```
    ./scripts/run_evaluate.sh
    ```
