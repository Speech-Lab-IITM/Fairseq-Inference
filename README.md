# Fairseq-Inference
Inference script for CCC-Wav2vec2.0 and Data2Vec-AQC

# Requirements
* PyTorch version >= 1.10.0
* Python version >= 3.8

``` bash
git clone https://github.com/Speech-Lab-IITM/fairseq_inference.git
cd fairseq-0.12.2
pip install --editable ./
```
In our tests, we used following:
* Python == 3.8.10
* Pytorch == 1.12.1
* torchaudio == 0.12.1
* Cuda == 11.3

# Usage
``` bash
python3 infer.py model_path audio_path
```

