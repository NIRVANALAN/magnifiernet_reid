
# Magnifier: Towards Semantic Adversary and Fusion for Person Re-identification 
<!-- # LIMITATIONS: HARD-CODED MASK POLICY / DISABLED 3PART POLICY / DISABLED ATTR BY HARDCOPDED 0 / SEG SIZE IS RESTRICTED TO 28-28 OR 16-32 DYNAMICALLY / DYNAMIC LOSS IS HARDEDCODED IN / FORCED NORM FEATURE / FORCED SOFTMAX LOSS FOR PART BRANCH, EVEN IF YOU CHOOSE TRIPLET IT WONT WORK FOR PART -->
<!-- ## Doubts: Label Smoothing seems to not really work when sampler is not softmax-triplet?? -->

## This repo is expanded on [Bag of Tricks and A Strong ReID Baseline](https://github.com/michuanhaohao/reid-strong-baseline)


## Results (rank1/mAP)
| Model             | CUHK03-L   | DukeMTMC-reID |
| ----------------- | ---------- | ------------- |
| Standard baseline | 69.8(67.4) | 82.7(70.8)    |
| + Mask Branch     | 71.3(69.0) | 83.9(72.9)    |
| + SAB             | 76.6(74.6) | 87.1(76.7)    |
| + SFB             | 77.4(75.9) | 86.7(75.2)    |
| **+ All Modules** | 82.4(79.6) | 90.0(80.7)    |
| **+ Reranking**   | 87.3(89.1) | 91.8(90.6)    |

<!-- [model(Market1501)](https://drive.google.com/open?id=1hn0sXLZ5yJcxtmuY-ItQfYD7hBtHwt7A)

[model(DukeMTMC-reID)](https://drive.google.com/open?id=1LARvQe-gUbflbanidUM0keKmHoKTpLUj) -->


1. Prepare dataset

    Create a directory to store reid datasets under this repo or outside this repo. Remember to set your path to the root of the dataset in `config/defaults.py` for all training and testing or set in every single config file in `configs/` or set in every single command.
<!-- 
    You can create a directory to store reid datasets under this repo via

    ```bash
    cd reid-strong-baseline
    mkdir data
    ```

    （1）Market1501

    * Download dataset to `data/` from http://www.liangzheng.org/Project/project_reid.html
    * Extract dataset and rename to `market1501`. The data structure would like:

    ```bash
    data
        market1501 # this folder contains 6 files.
            bounding_box_test/
            bounding_box_train/
            ......
    ```
    （2）DukeMTMC-reID

    * Download dataset to `data/` from https://github.com/layumi/DukeMTMC-reID_evaluation#download-dataset
    * Extract dataset and rename to `dukemtmc-reid`. The data structure would like:

    ```bash
    data
        dukemtmc-reid
        	DukeMTMC-reID # this folder contains 8 files.
            	bounding_box_test/
            	bounding_box_train/
            	......
    ``` -->

2. If you want to know the detailed configurations and their meaning, please refer to `config/defaults.py`. If you want to set your own parameters, you can follow our method: create a new yml file, then set your own parameters.  Add `--config_file='configs/your yml file'` int the commands described below, then our code will merge your configuration. automatically.

## Train
You can run these commands in  `.sh ` files for training different datasets of differernt loss.  You can also directly run code `sh *.sh` to run our demo after your custom modification.

<!-- 1. Market1501, cross entropy loss + triplet loss

```bash
python3 tools/train.py --config_file='configs/softmax_triplet.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('market1501')" OUTPUT_DIR "('your path to save checkpoints and logs')"
```

2. DukeMTMC-reID, cross entropy loss + triplet loss + center loss


```bash
python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('your device id')" DATASETS.NAMES "('dukemtmc')" OUTPUT_DIR "('your path to save checkpoints and logs')"
``` -->