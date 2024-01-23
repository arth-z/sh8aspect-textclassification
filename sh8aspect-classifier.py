import os
import tensorflow as tf

# casts a pair of (label, number) into (label, tf.int64) since tensorflow is like that
def labeler(example, index):
    return example, tf.cast(index, tf.int64)

ASPECT_NAMES = ["lantern", "forge", "edge", "winter", "heart", "grail", "moth", "knock"]

initial_labeled_ds = []

# pull data from text files, label them using ASPECT_NAMES and labeler function, and put into tf dataset
for i in range(len(ASPECT_NAMES)):
    lines_ds = tf.data.TextLineDataset(os.getcwd() + "/sh8aspect-data/train/" + str(i) + "-" + ASPECT_NAMES[i] + "Quotes.txt")
    labeled_ds = lines_ds.map(lambda ex: labeler(ex, i))
    initial_labeled_ds.append(labeled_ds)

# join the tf datasets together (previous approach only made a list of individual datasets)
all_labeled_ds = initial_labeled_ds[0]
for ds in initial_labeled_ds[1:]:
    all_labeled_ds = all_labeled_ds.concatenate(ds)

# scramble
all_labeled_ds = all_labeled_ds.shuffle(50000, reshuffle_each_iteration=False)

'''
for text, label in all_labeled_ds:
  print("Sentence: ", text.numpy())
  print("Label:", label.numpy())
'''

# using built-in text tf Keras vectorisation
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=250
    )

# training text is a dataset with the labels taken out, hence the lambda function
# feed it into the vectorisation layer
training_text = all_labeled_ds.map(lambda text, labels: text)
vectorize_layer.adapt(training_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(10000, 16),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8)])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])  
  
    return model

def export_from_model(model):
    export_model = tf.keras.Sequential([
        vectorize_layer,
        model,
        tf.keras.layers.Activation('sigmoid')])

    export_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )

    return export_model
   
model = create_model()
model.summary()

# organise data into training and validation
train_ds = all_labeled_ds.skip(32).shuffle(500000).padded_batch(4).map(vectorize_text)
val_ds = all_labeled_ds.take(32).shuffle(500000).padded_batch(4).map(vectorize_text)

# optimise
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# train - criminally small dataset, I had to make it myself since this is so niche, so that's why there's so many epochs
# probably overfitted...
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=175)

# export model so it works on raw text
export_model = export_from_model(model)

# user input model
while True:

    test_input = [input("Enter an artsy quote here (enter 'exit' to exit): ")]

    if test_input[0] == 'exit':
        break

    prediction_scores = export_model.predict(test_input)
    prediction_labels1 = tf.math.argmax(prediction_scores, axis=1)

    for sentence, label1 in zip(test_input, prediction_labels1):
        print("Sentence:", sentence)
        print("Predicted aspect:", ASPECT_NAMES[label1.numpy()])

