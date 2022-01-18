import csv
import numpy as np
import tensorflow as tf

# if uncommented, sets a specific seed to get to better uniformity 
#seed = 2020
#tf.random.set_seed(seed)

csv.field_size_limit(1310720) # avoids a buffer / memory size issue that comes up 
embedding_dimension = 300
width = 1000
extrinsic_file = 'extrinsicdata.csv'

# extract structured fields from file
print('loading CSV from', extrinsic_file)
with open(extrinsic_file, 'r') as f:
    r = csv.reader(f)
    i = 0
    structuredArray = []
    outputArray = []
    for row in r:
        i += 1
        if '' in row:
            continue 
        structuredinputs = np.array([float(j) for j in row[3:-1]])
        output = np.array([int(row[-1])])
        
        structuredArray.append(structuredinputs)
        outputArray.append(output)

outputArray = np.array(outputArray)
structuredArray = np.array(structuredArray)
print('Structured Array:', structuredArray.shape)
print('Output Array:', outputArray.shape)

# use tf.data to load numpy arrays and split them into different sets
print('loading TF Dataset')
total_output = tf.data.Dataset.from_tensor_slices(outputArray)
total_input = tf.data.Dataset.from_tensor_slices(structuredArray)
total_dataset = tf.data.Dataset.zip((total_input, total_output))
train_dataset = total_dataset.skip(600).shuffle(1000).batch(15)
validation_dataset = total_dataset.skip(300).take(300).batch(15)
holdout_dataset = total_dataset.take(300).batch(15)

# set up model, adjust Keras code below to try different combinations
input2 = tf.keras.Input(shape=(structuredArray.shape[1],))
sl = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())(input2)
sl = tf.keras.layers.Dropout(0.5)(sl)
sl = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())(sl)
output = tf.keras.layers.Dense(1, activation='sigmoid')(sl)

model = tf.keras.Model(inputs = input2, outputs = output)
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

# save model weights (weights rather than model to match unstructured/combo models)
# for more, refer to Tensorflow documentation: https://www.tensorflow.org/tutorials/keras/save_and_load#manually_save_weights
model.save_weights('structured_extrinsic') 