import csv
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import KerasTransformer

# if uncommented, sets a specific seed to get to better uniformity 
#seed = 2020
#tf.random.set_seed(seed)

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

# use Embedding layer based on pre-trained embeddings and disable further training
nl = tf.keras.layers.Embedding(vocabulary_size, embedding_dimension, embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix), input_length = width, trainable = False)(input1)
nl = tf.keras.layers.Dense(512, activation='relu')(nl)
nl = tf.keras.layers.Dropout(0.50)(nl)

# implement TextCNN 
kernels = [3, 5, 7, 10]
pooled = []
for kernel_size in kernels:
    mini_layer = tf.keras.layers.Conv1D(256, kernel_size, activation='relu')(nl)
    mini_pooled = tf.keras.layers.MaxPooling1D(width - kernel_size + 1)(mini_layer)
    pooled.append(mini_pooled)
nl = tf.keras.layers.Concatenate(axis=1)(pooled)
nl = tf.keras.layers.Flatten()(nl)
nl = tf.keras.layers.Dropout(0.50)(nl)

nl = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())(nl)
nl = tf.keras.layers.Dropout(0.5)(nl)
nl = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())(nl)
output = tf.keras.layers.Dense(1, activation='sigmoid')(nl)

model = tf.keras.Model(inputs = input1, outputs = output)
model.summary()

# run model, adjust metrics / callbacks / optimizers below
metrics = [
    tf.keras.metrics.TruePositives(name='tp'), 
    tf.keras.metrics.FalsePositives(name='fp'), 
    tf.keras.metrics.TrueNegatives(name='tn'), 
    tf.keras.metrics.FalseNegatives(name='fn'), 
    tf.keras.metrics.Recall(name='sen'), 
    tf.keras.metrics.Precision(name='prc'), 
    tf.keras.metrics.AUC(name='auroc'), 
    tf.keras.metrics.BinaryAccuracy(name='acc')
]
callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=4, verbose=1, restore_best_weights=True, min_delta=0.0001), 
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=2, verbose=1)
]
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='binary_crossentropy', metrics=metrics)
model.fit(train_dataset, epochs=160, validation_data=validation_dataset, verbose=1, callbacks=callbacks)
model.evaluate(validation_dataset)

# save model weights (weights rather than model due to size limitations)
# for more, refer to Tensorflow documentation: https://www.tensorflow.org/tutorials/keras/save_and_load#manually_save_weights
model.save_weights('pubmed_unstructured_extrinsic')
