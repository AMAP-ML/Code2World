# <img src="figure/logo.png" alt="icon" width="50" height="50"> Code2World: A GUI World Model via Renderable Code Generation
<p align="center">
<a href="https://amap-ml.github.io/Code2World" target="_blank"><img src="https://img.shields.io/badge/Project-Page-brightgreen"></a>
<a href="https://arxiv.org/abs/2602.09856" target="_blank"><img src="https://img.shields.io/badge/arXiv-2511.02778-red"></a>
<a href="https://huggingface.co/GD-ML/Code2World" target="_blank"><img src="https://img.shields.io/badge/🤗%20Model-Code2World-ffd21e"></a>
<a href="https://huggingface.co/datasets/GD-ML/AndroidCode" target="_blank"><img src="https://img.shields.io/badge/🤗%20Dataset-AndroidCode-ffd21e"></a>
<a href="https://huggingface.co/papers/2602.09856" target="_blank"><img src="https://img.shields.io/badge/🤗%20Daily%20Paper-2602.09856-ffd21e"></a>
</p>
Official implementation for Code2World, a novel VLM-based GUI World Model that predicts dynamic transitions via renderable code generation.

## 🕹️ Usage
### Environment Setup
1. We recommend setting up a clean Python environment:
```bash
conda create -n c2w python=3.10 -y
conda activate c2w
cd Code2World
pip install -r requirements.txt
```
2. Set up the Android emulator environment by referring to the [Android World](https://github.com/google-research/android_world/tree/main) configuration.
3. Configure the OPENAI_API_KEY in the environment variables.
4. Download the pretrained Code2World weights from Hugging Face, and set the path at line 38 in android_world/agents/m3a_qwen.py.
```python
model_name="ckpt/Code2World"
```


### Use Code2World for Enhancing GUI Agent
To test AndroidWorld, you can run the following command:
```python
python -u run.py --agent_name='m3a_gemini25f_Code2World'  --console_port=5556 --grpc_port=8554  --checkpoint_dir='ckpt/m3a_gemini25f_Code2World' 
```


## 🏅 Experiments
Table 1. **Quantitative Comparison** of various image and  code generation models on Android Control **(ID)** to assess basic capabilities on the same device, and GUI Odyssey **(OOD)** to test generalization robustness across unseen devices and cross-app scenarios. The best scores are in **bold** while the second best are in <ins>underlined</ins>.
![Tab1](figure/Tab1.png)



## 📌 Examples
Here are more qualitative comparison of next GUI state generation over Code2World and three baselines.
![case1](figure/case1.png)
_Figure 2. Launch the email application from the home screen to access the inbox._ 
![case2](figure/case2.png)\
_Figure 3. Click on "All News" button in the Cerebra Research application to view news content._ 
![case3](figure/case3.png)
_Figure 4. Mark a reminder task as completed by tapping the "Complete" button in the Reminder app._ 
![case4](figure/case4.png)
_Figure 5. Apply product filters by tapping the "Apply Filter" button in the e-commerce app to refresh the item list._


## 📑 Citation
If you find our project useful, we hope you can star our repo and cite our paper as follows:
```

```

