# Protein Sequence Alignment Scoring using Deep Learning

## Project Description
In this work, our interest is in predicting the utility of Sequence Alignments (SAs) towards protein structure prediction, i.e. how informative they are for the methods that use them to drive structure prediction. Although it is well understood that the ultimate accuracy of structure prediction depends on many factors other than the input SA, in general, a SA is the first seed that guides the structure prediction process. In this work, we develop a deep learning based method to rank multiple sequence alignments based on their quality, i.e., their usefulness for building a 3D models. Given a protein sequence and a set of alignments (SAs) generated for it using multiple methods, our method Deep-MSA-Score, predicts local distance difference test (lDDT) scores of the distance maps that can be predicted from the SAs. These lDDT scores serve as the alignment quality scores. Irrespective of the method that is used to generate a SA we predict a score it. Subsequently, the SAs can be ranked by these score for selection.


## Installation
1. Download the repository
```
git clone https://github.com/ba-lab/Alignment-Score.git
OR
Download the zip file: https://github.com/ba-lab/Alignment-Score/archive/refs/heads/main.zip

cd Alignment-Score
```

2. Download additional model file
```
wget http://deep.cs.umsl.edu/sa-scoring/model2019_07.tar.xz

tar -xf model2019_07.tar.xz
```

3. Install virtual environments using conda
```
conda env create -f score-env1.yml

conda env create -f score-env2.yml
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
