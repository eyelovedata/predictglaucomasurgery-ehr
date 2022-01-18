import csv, pickle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import KerasTransformer

model_weights_path = 'pubmed_unstructured_extrinsic' # select correct model weights path
''' copy all code from model up until model.summary()
replace TokenAndPositionPreTrainedEmbedding layers with TokenAndPositionEmbedding
'''

csv.field_size_limit(1310720) # avoids a buffer / memory size issue that comes up 
embedding_dimension = 300
width = 1000
extrinsic_file = 'extrinsicdata.csv'
model_path = 'pubmed_cbow_embeddings.h5'
model_vocabulary = 'pubmed_cbow_vocabulary.txt'

# grab vocabulary to load embeddings
print('loading vocabulary from', model_vocabulary)
with open(model_vocabulary, 'r', errors='surrogateescape') as g:
    vocabulary = []
    for word in g:
        vocabulary.append(word.strip())
    vocabulary_size = len(vocabulary)+2

# use encoder / tokenizer that comes with TF for simplicity sake
tokenizer = tfds.features.text.Tokenizer()
encoder = tfds.features.text.TokenTextEncoder(vocabulary, tokenizer=tokenizer)

# extract unstructured fields from file
print('loading CSV from', extrinsic_file)
with open(extrinsic_file, 'r') as f:
    r = csv.reader(f)
    i = 0
    outputArray = []
    tokenArray = []
    for row in r:
        i += 1
        if '' in row: # extra check to avoid blank / malformed rows
            continue 
        output = np.array([int(row[-1])])
        
        tokens = encoder.encode(row[2])
        tokens = tokens[0:width]
        if len(tokens) < width:
            tokens = tokens + [0 for i in range(width-len(tokens))]
        
        outputArray.append(output)
        tokenArray.append(tokens)

outputArray = np.array(outputArray)
tokenArray = np.array(tokenArray)
print('Token Array:', tokenArray.shape)
print('Output Array:', outputArray.shape)

# use tf.data to load numpy arrays and split them into different sets
print('loading TF Dataset')
total_output = tf.data.Dataset.from_tensor_slices(outputArray)
total_input = tf.data.Dataset.from_tensor_slices(tokenArray)
total_dataset = tf.data.Dataset.zip((total_input, total_output))
train_dataset = total_dataset.skip(600).shuffle(1000).batch(15)
validation_dataset = total_dataset.skip(300).take(300).batch(15)
holdout_dataset = total_dataset.take(300).batch(15)

# load embeddings
print('loading Pubmed CBOW vectors')
model = tf.keras.models.load_model(model_path)
e = model.layers[1]
embedding_matrix = e.get_weights()[0]
print(embedding_matrix.shape)
del model # free up memory

# set up model, adjust Keras code below to try different combinations
input1 = tf.keras.Input(shape=(width,))

''' swap out KerasTransformer.TokenAndPositionPreTrainedEmbedding with KerasTransformer.TokenAndPositionEmbedding '''
nl = KerasTransformer.TokenAndPositionEmbedding(width, vocabulary_size, embedding_dimension)(input1)
nl = KerasTransformer.TransformerBlock(embedding_dimension, 10, embedding_dimension)(nl)
nl = tf.keras.layers.GlobalAveragePooling1D()(nl)

nl = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())(nl)
nl = tf.keras.layers.Dropout(0.5)(nl)
nl = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())(nl)
output = tf.keras.layers.Dense(1, activation='sigmoid')(nl)

model = tf.keras.Model(inputs = input1, outputs = output)
model.summary()

# load weights
model.load_weights(model_weights_path)

# run model, adjust metrics / callbacks / optimizers below
callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=4, verbose=1, restore_best_weights=True, min_delta=0.0001), 
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=2, verbose=1)
]

# tries out different thresholds for decision criteria
thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
maxf1threshold = 0.0
maxf1 = 0.0
maxResults = []

for threshold in thresholds:
    print('\nThreshold:', threshold)
    metrics = [
        tf.keras.metrics.TruePositives(name='tp', thresholds=threshold), 
        tf.keras.metrics.FalsePositives(name='fp', thresholds=threshold), 
        tf.keras.metrics.TrueNegatives(name='tn', thresholds=threshold), 
        tf.keras.metrics.FalseNegatives(name='fn', thresholds=threshold), 
        tf.keras.metrics.Recall(name='sen', thresholds=threshold), 
        tf.keras.metrics.Precision(name='prc', thresholds=threshold), 
        tf.keras.metrics.AUC(name='auc', curve='ROC'), 
        tf.keras.metrics.AUC(name='auprc', curve='PR'), 
    ]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    
    loss, tp, fp, tn, fn, sens, prec, auroc, auprc = model.evaluate(validation_dataset)
    if sens * prec == 0.0:
        f1 = 0.0
    else:
        f1 = 2/(1/sens + 1/prec)
    
    if f1 > maxf1:
        maxf1threshold = threshold
        maxf1 = f1
        maxResults = [tp, fp, tn, fn, sens, prec]

print('\nBest threshold:', maxf1threshold)
print('Best F1:', maxf1*100)
print('True Positives:', maxResults[0])
print('False Positives:', maxResults[1])
print('True Negatives:', maxResults[2])
print('False Negatives:', maxResults[3])
print('Sensitivity/Recall:', maxResults[4]*100)
print('Precision:', maxResults[5]*100)

# use threshold with highest F1 to run on test set
metrics = [
    tf.keras.metrics.TruePositives(name='tp', thresholds=maxf1threshold), 
    tf.keras.metrics.FalsePositives(name='fp', thresholds=maxf1threshold), 
    tf.keras.metrics.TrueNegatives(name='tn', thresholds=maxf1threshold), 
    tf.keras.metrics.FalseNegatives(name='fn', thresholds=maxf1threshold), 
    tf.keras.metrics.Recall(name='sen', thresholds=maxf1threshold), 
    tf.keras.metrics.Precision(name='prc', thresholds=maxf1threshold), 
    tf.keras.metrics.AUC(name='auc', curve='ROC'), 
    tf.keras.metrics.AUC(name='auprc', curve='PR'), 
]
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
loss, tp, fp, tn, fn, sens, prec, auroc, auprc = model.evaluate(holdout_dataset)
print('AUROC:', auroc*100)
print('AUPRC:', auroc*100)
print('Sensitivity/Recall:', sens*100)
print('Precision:', prec*100)
if sens * prec == 0.0:
    f1 = 0.0
    print('F1 not applicable')
else:
    f1 = 2/(1/sens + 1/prec)
    print('F1:', f1*100)
print('True Positives', tp)
print('False Positives', fp)
print('True Negatives', tn)
print('False Negatives', fn)
