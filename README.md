# Protein Sequence Alignment Scoring using Deep Learning

## Project Description

**Background** A high-quality sequence alignment (SA) is the most important input feature for accurate protein structure prediction. For a protein sequence, there are many methods to generate a SA. However, when given a choice of more than one SA for a protein sequence, there are no methods to predict which SA may lead to more accurate models without actually building the models. In this work, we describe a method to predict the quality of a protein's SA.
**Methods** We created our own dataset by generating a variety of SAs for a set of 1,351 representative proteins and investigated various deep learning architectures to predict the local distance difference test (lDDT) scores of distance maps predicted with SAs as the input. These lDDT scores serve as indicators of the quality of the SAs. 
**Results** Using two independent test datasets consisting of CASP13 and CASP14 targets, we show that our method is effective for scoring and ranking SAs when a pool of SAs is available for a protein sequence. With an example, we further discuss that SA selection using our method can lead to improved structure prediction.

## Webserver

[http://deep.cs.umsl.edu/sa-scoring/](http://deep.cs.umsl.edu/sa-scoring/)

## Data

[http://deep.cs.umsl.edu/sascoring/download](http://deep.cs.umsl.edu/sascoring/download)

## Local installation
1. Download the repository
```
git clone https://github.com/ba-lab/Alignment-Score.git
OR
Download the zip file: https://github.com/ba-lab/Alignment-Score/archive/refs/heads/main.zip

cd Alignment-Score
```

2. Download additional model file
```
wget http://deep.cs.umsl.edu/sascoring/download/deep-msa-score/trRosetta_model2019_07.tar.xz

tar -xf trRosetta_model2019_07.tar.xz
```

3. Install virtual environments using conda
```
conda env create -f trRos-env.yml

conda env create -f msascore-env.yml
```
> Instructions to install conda can be found here: https://varhowto.com/install-miniconda-ubuntu-20-04/
OR,
Official site: https://docs.conda.io/en/latest/miniconda.html

## How to Run
```
chmod +x automating_script.sh

./automating_script.sh <path_to_msa> <target_name>

Eg command:
./automating_script.sh test_msa/T1024_DeepMSA2.a3m T1024
```
