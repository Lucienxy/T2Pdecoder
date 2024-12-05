# Converting Transcriptomics to Proteomics via T2Pdecoder
T2Pdecoder is an integrative multi-omics deep learning model designed to predict relative protein abundances across a wide range of proteins

# workflow
![image](https://github.com/Lucienxy/T2Pdecoder/blob/main/img/T2Pdecoder_workflow.png)

# Requirement
```
- python>=3.7.9
- pytorch>=2.0.1
- numpy==1.24.3
- pandas==2.2.2
- scikit-learn=1.3.0
- scipy=1.11.1
- gseapy==1.0.6
- lifelines==0.27.7
```
# Train and Application
```
# pre-train CLIP model
python -u CLIP_pre_train.py
# fine-tune CLIP model
python -u CLIP_fine_tune.py
# use the fine-tuned CLIP model to get the embedding vector
python -u CLIP_ft_app.py
# train VAE model
python -u T2Pdecoder_VAE_train.py
# use the trained T2Pdecoder to generate protein
python -u T2Pdecoder_generator.py
```
