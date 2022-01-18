# Deep Learning Approaches for Predicting Glaucoma Progression Using Electronic Health Records and Natural Language Processing


Housing code and project files related to predicting whether a patient will need glaucoma surgery or not, using data from electronic health records, including both structured data and text data (clinical notes). 
Includes code for cutting cohort, feature engineering various sets of features, modeling, and plotting results. 

What's in the ipynb notebooks: 
- predictglaucsurg-cutcohort4.ipynb: creation of cohort and most feature engineering 
- predictglaucsurg-populationcharacteristics.ipynb:  analysis for table 1 of manuscript, for population characteristics 
- predictglaucsurg-curvesfortextandstructuredmodels.ipynb:  creation of ROC and PR curves for main and supplementary figures as well as Table 2 and supplementary table
- predictglaucomasurgery-explainability.ipynb:  explainability studies on structured data/model using LIME 


What's in the modeling folder: 

Descriptions of key files
Key dependencies:
- Python 3.4+
- Tensorflow 2.x with Keras (GPU support)
- Tensorflow Datasets 
- Numpy 1.x 

extract-extrinsic.py
- takes raw input for glaucoma training task (predictglaucomasurgerycombinedprocessedwolaserfilteredbalanced.csv)
- shuffles & removes stopwords while balancing classes across all sets and outputs (extrinsicdata.csv)

KerasTransformer.py
- Library based on Keras docs (https://keras.io/examples/nlp/text_classification_with_transformer/) to use their Transformer 

structured-extrinsic.py
- uses tf.data to pipeline from data (extrinsicdata.csv) 
- trains a model using only structured fields

pubmed-unstructured-extrinsic.py 
- uses tf.data to pipeline from data (extrinsicdata.csv) 
- trains a (TextCNN) model using embeddings from pubmed (pubmed_cbow_embeddings.h5) using only unstructured fields

pubmed-combined-extrinsic.py
- uses tf.data to pipeline from data (extrinsicdata.csv) 
- trains a model using embeddings from pubmed (pubmed_cbow_embeddings.h5) using both unstructured and structured fields

f1optimizer.py
- need to copy & paste code from chosen model (and sub out any TokenAndPositionPreTrainedEmbedding layers)
- tests saved model weights for decision threshold with best F1 score, and then evaluates it on the holdout set

glaucoma-mlstructured.py
- penalized regression and gradient boosted trees models using structured data, for supplementary analyses
