# [ICASSP 2026] *Ister*: Linear Transformer for Efficient Multivariate Time Series Forecasting

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)  
[![Pytorch](https://img.shields.io/badge/pytorch-1.10%2B-red.svg)](https://pytorch.org/)  

📢 **Ister** (Inverted Seasonal-Trend Decomposition Transformer) is an efficient linear Transformer-based model designed for **multivariate time series forecasting**. It decomposes time series into seasonal and trend components and efficiently captures inter-series dependencies. The proposed **Dot-attention mechanism** significantly improves computational efficiency and prediction accuracy. 

🔗 **Paper**: [IEEE Xplore](https://ieeexplore.ieee.org/document/11463971)  

---

## 📣 News
* `2026/1/18` 💥💥 Ister is honored to be accepted by ICASSP 2026!

---

## 🚀 Features  

✅ **Dot-attention Mechanism** - Reduces computational complexity from **O(L²) to O(L)** while enhancing interpretability.  
✅ **State-of-the-Art Performance** - Outperforms existing models with **up to 10% lower MSE** on real-world benchmarks.  

---

## 🔍 Updates: Ister 2.0  

In the previous version of **Ister**, the model backbone primarily relied on **DualTransformer** to jointly model **inter-series dependencies** and **multi-periodicity**. While effective, this design led to **high training and inference costs**.  

🔬 **What's new in Ister 2.0?**  
- We found that **tailoring modeling approaches** based on dataset characteristics, combined with **hyperparameter tuning and architecture optimization**, can achieve **comparable accuracy** while **significantly reducing** model size, inference latency, and training costs.  
- Leveraging the **linear complexity of Dot-attention**, **Ister 2.0** exhibits **better scalability**, making it more practical for large-scale time series forecasting.  
- We encourage researchers to explore more **efficient ways** to jointly model **inter-series dependencies** and **multi-periodicity**.  

📌 **Two Specialized Variants:**  

| Model       | Suitable for |
|------------|-------------|
| **CD_Ister** | Designed for datasets with **strong channel dependencies** (camera-ready version) |
| **MP_Ister** | Best suited for datasets where **channels are independent** but **multi-periodicity is prominent** |

To facilitate further research, we provide **training scripts** for both models across **all datasets**. Check the `scripts/` directory for ready-to-use commands! 🚀  

---

## 📂 Installation  

### Environment Setup  
Ensure you have Python 3.8+ and PyTorch 1.10+ installed. You can create a virtual environment:  

```bash
conda create -n ister_env python=3.8
conda activate ister_env
cd ister
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118  # Adjust CUDA version
pip install -r requirements.txt
```

---

## 🔥 Quick Start
To run the experiment in paper:
```bash
bash scripts/CD_Ister/Traffic/Ister.sh
bash scripts/MP_Ister/Weather_script/Ister.sh
```

---

## 📜 Citation
If you find Ister useful, please consider citing our paper:
```bibtex
@INPROCEEDINGS{11463971,
  author={Cao, Fanpu and Yang, Shu and Chen, Zhengjian and Liu, Ye and Cui, Laizhong},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Ister: Linear Transformer for Efficient Multivariate Time Series Forecasting}, 
  year={2026},
  volume={},
  number={},
  pages={3571-3575},
  keywords={Feeds;Antennas;Filtering;Filters;LoRa;Protocols;HTTP;Data communication;Radio communication;Radio access networks;Multivariate time series forecasting;Channel alignment;Efficient attention mechanism},
  doi={10.1109/ICASSP55912.2026.11463971}
}
```
