### MFAE: Multimodal Feature Adaptive Enhancement for Fake News Video Detection
Wenhao Wang, Mingxin Li, Jiao Qiao, Haotong Du, Xianghua Li, Chao Gao, Zhen Wang, MFAE: Multimodal feature adaptive enhancement for fake news video detection, Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM25), 2025, 3082-3092
### Abstract

With the rapid global growth of short video platforms, the spread of fake news has become increasingly prevalent, creating an urgent demand for effective automated detection methods. Current approaches typically rely on feature extractors to gather information from multiple modalities and then generate predictions through classifiers. However, these methods often fail to fully leverage complex cross-modal information and may overlook video editing cues, limiting overall performance. To address these issues, we propose MFAE, a novel framework for Multimodal Feature Adaptive Enhancement for Fake News Video Detection. The framework first extracts semantic and emotional features from text and uses them to build coarse multimodal representations. These representations are further refined by an Adaptive Enhancement module designed to strengthen visual and audio modalities. Spatial and temporal features are then extracted separately, with temporal features further improved via a Temporal Enhancement module. Finally, the individually enhanced features are fed into a multimodal integration module for interaction. Comprehensive experiments on two benchmark datasets demonstrate the effectiveness of MFAE, with accuracy improvements of 2.21% and 4.35% on FakeSV and FakeTT, respectively.

### Project Structure

```
/sda/home/temp/wangwenhao/Code/upload_cikm
├─ main.py                         
├─ model/
│  └─ MFAE.py                      
├─ utils/
│  └─ dataloader.py                
├─ baseline/
│  ├─ CLIP.py                      
│  ├─ RoBERTa.py                   
│  └─ Video.py                     
└─ mamba_ssm/                      
   ├─ __init__.py
   ├─ distributed/ ... 
   ├─ models/ ...
   ├─ modules/ ...
   ├─ ops/ ...
   └─ utils/ ...
```

### Installation

1) Recommended: Python 3.8


2) Install dependencies 

```bash
pip install -r requirements.txt
```

### Usage

- Entry point for training/inference: `main.py`
```bash
python main.py \
  --dataset fakett \  # fakett / fakesv
  --mode train \      # train / inference_test
  --epoches 10 \
  --batch_size 8 \
  --early_stop 3 \
  --seed 2025 \
  --gpu 0 \
  --lr 1e-4 \
  --alpha 1.0 \
  --beta 1.0
```

### Key Arguments

- Dataset and flow
  - `--dataset`: `fakett` / `fakesv`
  - `--mode`: `train` / `inference_test`
- Training hyperparameters
  - `--epoches`, `--batch_size`, `--early_stop`, `--seed`, `--gpu`
  - `--lr`, `--alpha`, `--beta` (used as loss/fusion weights; tune empirically)

### Model Architecture

-  AE (Efusion)
  - Map text emotion and text semantic features into 128-d and combine; map visual frames and audio features into 128-d
  - Cross-modal co-attention between text and vision (`co_attention`)
  - Diffusion-style inversion: build fused representation `diff_fuse`, iteratively denoise and align to targets (CLIP visual target / audio target)
  - Transformer encoder for semantic integration; output binary logits

- TE (Etem)
  - OCR features extracted, then MLP to 128-d representation
  - Lightweight state-space modeling (Mamba-SSM inspired) for temporal enhancement
  - Output binary logits

- FinalFusion
  - Convert both branch logits via linear layers and 1D conv to left/right streams
  - Concatenate, apply lightweight SSM and sigmoid gating to fuse
  - Final FC outputs binary prediction; branch logits kept for analysis/distillation

### Datasets

- Current implementation supports `fakett` and `fakesv` (initialized in `utils/dataloader.py`).
fakesv (https://github.com/ICTMCG/FakeSV)
fakett (https://github.com/ICTMCG/FakingRecipe)
### Run Baselines

- CLIP baseline (image-text)
```bash
python baseline/CLIP.py
# Fill CSV paths for train/val/test and image folder, as well as learning rate, inside the script
```

- RoBERTa baseline (text)
```bash
python baseline/RoBERTa.py
# Specify the local model name and CSV data paths at the top of the script
```

- TimesFormer baseline (video)
```bash
python baseline/Video.py
# Provide CSV paths and C3D feature HDF5 folder in train_and_evaluate()
```

### Acknowledgments

- This project uses or is inspired by:
  - OpenAI CLIP
  - HuggingFace Transformers (RoBERTa, TimesFormer, and training utilities)
  - Temporal modeling ideas from Mamba-SSM (used in a lightweight form here)

### License

This project is licensed under the MIT License.

### Citation

If this project helps your research, please cite (replace with your paper details):
```bibtex
@article{mfae2025,
  title   = {Multimodal Feature Adaptive Enhancement for Fake News Video Detection},
  author  = {Wenhao Wang, Mingxin Li, Jiao Qiao, Haotong Du, Xianghua Li, Chao Gao, Zhen Wang},
  journal = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  year    = {2025}
}
```
