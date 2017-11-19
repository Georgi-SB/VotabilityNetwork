#   architecture:
#   1. input: sequence of characters: learn character embeddings
#   1a. possibly apply convolution to char embeddings
#   2. concatanate char embeddings features into words
#   3. sentences: represented as 2 channels: - word2vec embeddings (not trainable), - feature maps from characters from previous step
#   4. apply convolutions: inception layer 1. 1-2-3-4-5-gram convolutions (prior to them some 1x1 conv to reduce dimensionality)
#   5. apply LSTM: several groups - one group per each inception tower
#   6. the LSTM output goes into 2-3 fully connected layers



import tensorflow as tf
import numpy as np
import os


VALIDATION_SPLIT = 0.1
GLOVE_DIR = "/home/joro/Documents/gitRepos/VotabilityNetwork/glove.6B/"
EMBEDDING_DIM = 100
CHAR_EMBEDDING_DIM = 30
MAX_SENTENCE_LENGTH = 20
CHAR_MAX_SENTENCE_LENGTH = 200
VOCABULARY_SIZE = 100000
CHARACTER_VOCABULARY_SIZE = 100

# define documents
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better. and really start to dow so please otherwise you are really really screwed']
# define class labels
labels = [1,1,1,1,1,0,0,0,0,0]
labels = np.asarray(labels)


#############################
# prepare word level data
# create and fit the word tokenizer
word_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCABULARY_SIZE)
word_tokenizer.fit_on_texts(docs)

#word data: dimensions are nb_of_sentences x max_sentence_length. [i,j] is the j+1-st word in i+1st sentence
word_sequences = word_tokenizer.texts_to_sequences(docs)
word_data = tf.keras.preprocessing.sequence.pad_sequences(word_sequences, maxlen=MAX_SENTENCE_LENGTH)
print("Shape of word data", word_data.shape)


# word_index contains a word to int key value pairs. _rev is the reversed index
word_index = word_tokenizer.word_index
word_index_rev = dict((value,key) for key,value in word_index.items() )


#############################
# prepare sentence-character level data
char_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words= CHARACTER_VOCABULARY_SIZE,  char_level= True)
char_tokenizer.fit_on_texts(word_index.keys() + docs)

#create char embedding matrix. nb_of_sentences x max_sentence length in chars
char_sequences = char_tokenizer.texts_to_sequences(docs)
char_data = tf.keras.preprocessing.sequence.pad_sequences(char_sequences,
                                                          padding='post', truncating='post',
                                                          maxlen =  CHAR_MAX_SENTENCE_LENGTH)
char_index = char_tokenizer.word_index


#############################
#create word-char input data: nb_of_sentences , max_nb_words per sentence, max nb of chars per word
max_word_length = max(len(s) for s in word_index.keys())
word_char_input = np.zeros([len(docs), word_data.shape[1], max_word_length])

for idx_sentence in len(docs): #loop over sentences
    for idx_word in word_data.shape[1]: #loop over words in a sentence
        if word_data[idx_sentence,  idx_word] != 0:
            tmp =  char_tokenizer.texts_to_sequences(word_data[idx_sentence,  idx_word] )
            word_char_input[idx_sentence,idx_word, len(tmp)] = tmp

word_char_input_flat = np.reshape(word_char_input[ word_char_input[0], word_char_input[1]*word_char_input[2]])

#############################

#split into test and train
indices = np.arange(word_data.shape[0]) #sentence indices
np.random.shuffle(indices)
word_data = word_data[indices]
labels = labels[indices]

nb_validation_samples = int(VALIDATION_SPLIT * word_data.shape[0])

x_train = word_data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = word_data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


#############################

#preparing the char conv net branch
char_embedding_layer = tf.keras.layers.Embedding(input_dim=len(char_index) + 1,
                            output_dim = CHAR_EMBEDDING_DIM,
                            input_length=char_data.shape[1],
                            trainable=True)

char_sequence_input = tf.keras.layers.Input(shape=(CHAR_MAX_SENTENCE_LENGTH,), dtype='int32', name ="char_input")
embedded_char_sequence = char_embedding_layer(char_sequence_input)
x_char = tf.keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'valid', activation='relu')(embedded_char_sequence)
x_char = tf.keras.layers.MaxPooling1D(pool_size=3)(x_char)
x_char = tf.keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'valid', activation='relu')(x_char)
x_char = tf.keras.layers.MaxPooling1D(pool_size=3)(x_char)
x_char = tf.keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'valid', activation='relu')(x_char)
# x = tf.keras.layers.MaxPooling1D(pool_size=35)(x)  # global max pooling
x_char_out = tf.keras.layers.Flatten()(x_char)


#############################


#prepare the word-char branch
word_char_sequence_input = tf.keras.layers.Input(shape=(word_data.shape[1] * max_word_length,),
                                                 dtype='int32', name ="word_char_input")
embedded_word_char_sequence = char_embedding_layer(word_char_sequence_input)
#apply 1d conv to aggregate along words. stride is precisely the max word length in chars
x_word_char = tf.keras.layers.Conv1D(filters=EMBEDDING_DIM, kernel_size = max_word_length, strides= max_word_length, activation = 'relu')
#output should be max_sentence_length x EMBEDDING_DIM where each row represents the char level representation of the words
# this represents a new char level word embedding of the same size as the one used via glove or word2vec


#############################


#prepare the word input branch
#prepare the embeddings matrix


# #import glove embedings into a dictionary
# embeddings_index = {}
# f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()

#print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = np.random.rand(EMBEDDING_DIM)     # embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector



#############################

#DEFINE THE NETWORK


# define the word input:

word_embedding_layer_glove = tf.keras.layers.Embedding(input_dim=len(word_index) + 1,
                            output_dim = EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=word_data.shape[1],
                            trainable=False)

word_embedding_layer_trainable = tf.keras.layers.Embedding(input_dim=len(word_index) + 1,
                            output_dim = EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=word_data.shape[1],
                            trainable=True)

word_sequence_input = tf.keras.layers.Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32', name ="word_input")
#add non trainable word embedding sequence
word_embedded_sequence_glove = word_embedding_layer_glove(word_sequence_input)



#add a third word level learnable channel
word_embedded_sequence_trainable = word_embedding_layer_trainable(word_sequence_input)


#stack the glove/word2vec representation with the char level one
#the shape resembles three-channel word embeddings - one for glove, one for trainable word representation one for trainable char level representation
x_word = tf.keras.backend.stack((word_embedded_sequence_glove, word_embedded_sequence_trainable, x_word_char), axis=-1 ) #see numpy.stack

#do some convolutions
#todo - implement inception layer here to look at 1-2-3-4-grams

x_word = tf.keras.layers.Conv2D(filters = 64, kernel_size = (2,2), padding = 'valid', activation='relu')(x_word)

x_word_out = tf.keras.layers.Flatten()(x_word)

# Stack LSTM level
#todo

# merge with the sentence char branch


x = tf.keras.layers.concatenate([x_char_out, x_word_out])



#preparing a conv net


x = tf.keras.layers.Conv1D(filters = 128, kernel_size = 5, padding = 'same', activation='relu')(x)
x = tf.keras.layers.MaxPooling1D(pool_size=5, strides = 1)(x)
x = tf.keras.layers.Conv1D(filters =128, kernel_size = 5, padding = 'same', activation='relu')(x)
x = tf.keras.layers.MaxPooling1D(pool_size=5, strides = 1)(x)
x = tf.keras.layers.Conv1D(filters =128, kernel_size = 5, padding = 'same', activation='relu')(x)
# x = tf.keras.layers.MaxPooling1D(pool_size=35)(x)  # global max pooling
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
preds = tf.keras.layers.Dense(1, activation='sigmoid')(x)
#preds = tf.keras.layers.Dense(len(labels_index), activation='softmax')(x)
model = tf.keras.models.Model(inputs=[char_sequence_input, word_sequence_input], outputs=[preds])

print(model.summary())
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# loss='categorical_crossentropy'


# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=2, batch_size=128,verbose=1)



# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))



print("OLD STUFF from here on")

# integer encode the documents
vocab_size = 50
encoded_docs = [tf.keras.preprocessing.text.one_hot(d, vocab_size,
                                                    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                                    lower=True,
                                                    split=" ") for d in docs]

print(encoded_docs)

#split each doc into a sequence of words

words = [tf.keras.preprocessing.text.text_to_word_sequence(d,
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ") for d in docs]


print(words)




# pad documents to a max length of 4 words
max_length = 4
padded_docs = tf.keras.preprocessing.sequence.pad_sequences(sequences = encoded_docs,
                                                            maxlen=max_length, padding='post')
print(padded_docs)

embeddings_size = 8

inputs = tf.keras.Input(shape=(4,))
word_embeddings = tf.keras.layers.Embedding(input_dim=vocab_size,
                                            output_dim = 8,
                                            input_length = max_length )(inputs)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 8, input_length=max_length))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())

# fit the model
model.fit(padded_docs, labels, epochs=80, verbose=1)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))


