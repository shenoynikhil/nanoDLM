# nanoDLM: nano Diffusion Language Models
Simple Implementation of Diffusion Language Models for educational purposes. I have tried to keep the code as close to the original codebase while keeping it simple.

Note: This is a work in progress. Would really appreciate any contributions!

#### Installation
```bash
uv venv # use uv because it's cool
uv pip install -r requirements.txt
```

#### Instructions
The following models have been implemented:
- [x] [MDLM: Simple and Effective Masked Diffusion Language Models (Sahoo et al. 2024)](https://arxiv.org/abs/2406.07524): To run the model, use `python mdlm.py`

TODO:
- [ ] [SEDD: Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (Lou et al. 2024)](https://arxiv.org/abs/2310.16834): To run the model, use `python sedd.py`

#### Notes
[27/04/2025] Currently, it doesn't seem like the MDLM model is not training efficiently (although it does learn something). Need more investigation.


#### References
- [MDLM codebase](https://github.com/lucidrains/mdlm) which is already quite neat! 
- Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT)
- DiT implementation from Meta AI research: [Codebase](https://github.com/facebookresearch/DiT)
