# [ICASSP 2026] *Ister*: Linear Transformer for Efficient Multivariate Time Series Forecasting

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)  
[![Pytorch](https://img.shields.io/badge/pytorch-1.10%2B-red.svg)](https://pytorch.org/)  

📢 **Ister** (Inverted Seasonal-Trend Decomposition Transformer) is an efficient linear Transformer-based model designed for **multivariate time series forecasting**. It decomposes time series into seasonal and trend components and efficiently captures inter-series dependencies. The proposed **Dot-attention mechanism** significantly improves computational efficiency and prediction accuracy. 

🔗 **Paper**: [arXiv:2412.18798v2](https://arxiv.org/abs/2412.18798)  

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
git clone https://github.com/your_username/ister.git
cd ister
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118  # Adjust CUDA version
pip install -r requirements.txt
```

---

## 🔥 Quick Start
To run the experiment in paper:
```bash
bash scripts/CD_Ister/Traffic/Ister.sh
bash scripts/MP_Ister/ECL_script/Ister.sh
bash scripts/MP_Ister/Weather_script/Ister.sh
```

---

## 📜 Citation
If you find Ister useful, please consider citing our paper:
```bibtex
@misc{cao2025Ister,
      title={Ister: Inverted Seasonal-Trend Decomposition Transformer for Explainable Multivariate Time Series Forecasting}, 
      author={Fanpu Cao and Shu Yang and Zhengjian Chen and Ye Liu and Laizhong Cui},
      year={2025},
      eprint={2412.18798},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.18798}, 
}
```
