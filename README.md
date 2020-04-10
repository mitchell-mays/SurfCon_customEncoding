# SurfCon
Implementation of SurfCon model for Paper "[SurfCon: Synonym Discovery on Privacy-Aware Clinical Data](https://arxiv.org/abs/1906.09285)", which studies Synonym Discovery on Privacy-Aware Clinical Data.

## 1. Introduction
This repository is the implementation for the SurfCon model that utilizes the surface form and global context information to mine synonyms for medical terms extracted from clinical texts, such as Electronic Medical Records (EMRs).

The surface form information provides connections between medical terms in the surface form level. Not surprisingly, the surface form information is critical in the medical domain. For example, term **hypocupremia** is the synonym of **copper deficiency** in which _hypo_ means _deficiency_, _cupre_ means _copper_ and _mia_ is connected with _blood_. Inspired by such intuition, we design a bi-level surface form encoder to capture the information in both character and word levels of the medical terms.

Moreover, the global context information provides semantic information between medical terms. To deal with OOV problem (e.g., query terms do not show up in existing data), we leverage an inductive context prediction module to predict the close neighbors for terms and aggregate the predicted neighbors to calculate a context score between terms. 

Please refer to our paper for more detailed information.

## 2. Dataset
The dataset used in current experiments contains medical term-term co-occurrence graphs extracted from EMRs. The dataset can be downloaded from the original paper, [Building the graph of medicine from millions of clinical narratives](https://datadryad.org/resource/doi:10.5061/dryad.jp917)

More importantly, you can apply our model to your own data. Our model and problem setting can be extended to any other text corpus with the privacy concerns as long as a co-occurrence graph is provided.



## 3. Run

### Testing

1. For testing our pretrained SurfCon model to discover synonyms, please download the [pretrained model and parameters](https://drive.google.com/file/d/1126dtSV4XI_FWP4l0hUogijjUimRYIEF/view?usp=sharing) and ensure the correct file paths.

2. The pretrained embeddings can be downloaded here: [GloVe](http://nlp.stanford.edu/data/glove.6B.zip), [charNgram](http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/jmt_pre-trained_embeddings.tar.gz), [node features](https://drive.google.com/file/d/1nKXDppoSsT6uHCl0yG_zlrC4QFyCyu41/view?usp=sharing).

3. If you want to test quickly with existing medicla terms, please refer to [here](https://drive.google.com/file/d/1RN0x45dnMAkRKQWAwIqoz2qNL_3hfsQv/view?usp=sharing) and put them in the corresponding path of *data* folder. If you have your own terms, please reivse the argument `--cand_terms_path` in `test_surfcon.sh` with a text file (one term per line).

4. Testing our pretrained SurfCon model to discover synonyms:

        > bash test_surfcon.sh

### Training

Steps for trianing the model from scratch:

Step 1: Training the inductive context prediction model:

        > bash train_ctx_pred.sh
        

Step 2: Training the final ranking model:

        > bash train_surfcon.sh


If you have any questions, please feel free to contact us! Also, feel free to check other tools in our group (https://github.com/sunlab-osu) :blush:

## 4. Citation
```
@inproceedings{wang2019surfcon,
  title={SurfCon: Synonym Discovery on Privacy-Aware Clinical Data},
  author={Wang, Zhen and Yue, Xiang and Moosavinasab, Soheil and Huang, Yungui and Lin, Simon and Sun, Huan},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2019},
  organization={ACM}
}
```
