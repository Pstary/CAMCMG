# CAMCMG: A Change-aware Model for Commit Message Generation

## The architecture of the model
We propose CAMCMG, which is a Change-aware Model for Commit Message Generation. First, context plays an important role in commit message generation because the effectiveness of the generated message depends on the surrounding context, except for code changes. Therefore, we employ a sliding window to capture local contextual information around each hunk. It allows each token to pay attention to the neighboring tokens. Second, we further introduce a CMG-oriented change attention mechanism to capture semantic associations across hunks, which enables related tokens among different hunks to establish attention connections. Finally, we design a change alignment loss to optimize the model to enhance the generation of tokens in the commit message that correspond to the changes of diff (denoted as change-aligned tokens) during the training phase.
<img width="2000" height="1600" alt="architecture" src="https://github.com/user-attachments/assets/6362a3d1-cbcb-48c6-b7e9-6d40959fc306" />

## Environment
```
conda create -n CAMCMG python=3.9 -y
conda activate CAMCMG
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2 transformers==4.12.5
```

## Dataset
The dataset can be downloaded <a href="https://zenodo.org/records/7196966#.Y0juJHZBxmM">here</a>. It comes from <a href="https://github.com/DeepSoftwareAnalytics/RACE/">RACE</a>.

## Training
```
bash run.sh gpu_id lang
```
For example: `bash run.sh 1 java`

## Results
| Language   | Result Dir                   |
|-------------|------------------------------|
| Java        | results/java/preds.msg       |
| C#          | results/csharp/preds.msg   |
| C++         | results/cpp/preds.msg      |
| Python      | results/python/preds.msg   |
| JavaScript  | results/javascript/preds.msg|
