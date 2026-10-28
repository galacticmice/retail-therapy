<a id="readme-top"></a>

<br />
<div align="center">
<h1 align="center">Retail Therapy</h1>

  <p align="center">
    <i>What do they buy next?</i>
    <br />
    <br />
  </p>
</div>

## About

Predict the trajectory of Amazon customer purchases by tracking Amazon reviews across a subset of users.

Train a regression model with `user_id`, `category_id` to detect broad sequential categorical predictions of Amazon purchases.

Model-assisted labelling for finer object identification by extracting metadata from `product_id` (stretch).

Describe a temporal relationship of events using `timestamp` by transitioning to a time-series model (stretch).

<br />

## Getting Started

You will need ~15 GB of space to store a subset of the dataset locally.

<br />

### Setup
This is not a full featured dataset yet.
Existing code is used to explore labels only.
Will be removed later.

* data preparation
```sh
python3 dataset.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!--
### Usage

SETUP DESCRIPTION HERE


-->

## Roadmap

🤔 Preparations
- [x] Sample dataset to explore useful labels
- [x] Clean and compact dataset
    - [x] Prune all `user_id` that only appears <n counts(default=5)
    -  <img width="170" height="35" alt="Pasted Graphic" src="https://github.com/user-attachments/assets/2f31626d-5e07-4df8-b98a-57dff353f4f3" />
    - [x] Parallel import and cleaning while ensuring copy parity
        - *using `ParallelPoolExecutor`* and `polars` for large dataset

💪 Training
- [ ] Train working (broad) predictor
    - [ ] Evaluate quality and test on larger random dataset

✨ UI
- [ ] GUI for graphical user interactions
    - [ ] Search bar for labels
    - [ ] Multi-label dynamic prediction

🤓 Stretch Goals
- [ ] Add finer labels with assistance of second model
    - *Instead of "Electronics" -> "Keyboard"*
    - *Instead of "All_Beauty" -> "Shampoo"*
- [ ] Time series analysis with RBF kernel
    - *Transition from naive-sequential to time-weighted predictions*

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

### Dataset

This project uses the [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
dataset from McAuley Lab.

### Citation

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
