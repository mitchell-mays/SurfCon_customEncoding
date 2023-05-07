# CS598 Deep Learning for Healthcare Project - Reproduction of SurfCon
Reproduction of SurfCon model for Paper "[SurfCon: Synonym Discovery on Privacy-Aware Clinical Data](https://arxiv.org/abs/1906.09285)", which studies Synonym Discovery on Privacy-Aware Clinical Data.

## 1. Reproducability Summary
This project recreates the work done in the above paper. The goal of the paper is to utilize aggregated Co-frequency data of medical terms from Clinical Notes as an indicator of relationships between these terms. For this paper, the authors demonstrated the ability to identify synonymous terms (indicated by alignment under the same UMLS Concept). Our team was able to reproduce positive results akin to to the results seen by the authors of the Surfcon Paper. We reproduced two of the expirements done by the authors, with a slight variation of datasets. The character based pre-trained embeddings utilized by the original authors were unavailable, and therefore we identified and utilized a different subword embeddings instead.

## 2. Description & Context
While the Authors did make code available for the models and training, modifications were required for the following key elements:

#### 1. Data loading
#### 2. Data pre-processing
#### 3. Transform data structure for inputs
#### 4. Implement PPMI algorithm to convert frequency to PPMI
#### 5. Implement subsampling algorithm
#### 6. Map & Create synonym graph for labels
#### 7. Update outdated packages
Additionally, as the character based (surface form) pre-trained embedding was not available, we worked with the authors to find a suitable alternative (see below), and wrote code to pre-process this data as the structure of the datasets differed.

Therefore, the referenced code includes a combination of the authors original code, modifications made by our team, and new code generated by our team.


## 3. Testing

Refer to the notebook cs598_execution.ipynb under src folder which has detailed instructions to pre-process data, train and test the models.
To test/run this code, note the instructions about data and compute requirements as mentioned in the cs598_execution.ipynb. 



