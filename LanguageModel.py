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



##################################################################################################################
##################### DEFINE SETTINGS/CONSTANTS###################################################################
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

##################################################################################################################
##################################################################################################################


##################################################################################################################
############################### PREPARE INPUT DATA ###############################################################

#############################
# prepare word level data
# create and fit the word tokenizer
#############################
word_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCABULARY_SIZE)
word_tokenizer.fit_on_texts(docs)

#word data: dimensions are nb_of_sentences x max_sentence_length. [i,j] is the j+1-st word in i+1st sentence
word_sequences = word_tokenizer.texts_to_sequences(docs)
word_data = tf.keras.preprocessing.sequence.pad_sequences(word_sequences, maxlen=MAX_SENTENCE_LENGTH, padding='post')
print("Shape of word data", word_data.shape)


# word_index contains a word to int key value pairs. _rev is the reversed index
word_index = word_tokenizer.word_index
word_index_rev = dict((value,key) for key,value in word_index.items() )


#############################
# prepare sentence-character level data
#############################
char_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words= CHARACTER_VOCABULARY_SIZE,  char_level= True)
char_tokenizer.fit_on_texts(list(word_index.keys()) + docs)

#create char embedding matrix. nb_of_sentences x max_sentence length in chars
char_sequences = char_tokenizer.texts_to_sequences(docs)
char_data = tf.keras.preprocessing.sequence.pad_sequences(char_sequences,
                                                          padding='post', truncating='post',
                                                          maxlen =  CHAR_MAX_SENTENCE_LENGTH)
char_index = char_tokenizer.word_index


#############################


#############################
# create word-char input data: nb_of_sentences , max_nb_words per sentence, max nb of chars per word
#############################

max_word_length = max(len(s) for s in word_index.keys())
word_char_input = np.zeros([len(docs), word_data.shape[1], max_word_length])

for idx_sentence in range(len(docs)): #loop over sentences
    for idx_word in range(word_data.shape[1]): #loop over words in a sentence
        if word_data[idx_sentence,  idx_word] != 0:
            word_string = str(word_index_rev[word_data[idx_sentence,  idx_word]] )
            tmp = np.ndarray.flatten(np.asarray(char_tokenizer.texts_to_sequences(word_string)))
            word_char_input[idx_sentence, idx_word, 0:len(tmp)] = tmp

word_char_input_flat = np.reshape(word_char_input,
                                  [ word_char_input.shape[0], word_char_input.shape[1]*word_char_input.shape[2]])


#############################
#prepare the embeddings matrix
#############################

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

##################################################################################################################
##################################################################################################################





##################################################################################################################
################################ SPLIT INTO TEST AND TRAIN #######################################################

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


##################################################################################################################
##################################################################################################################


##################################################################################################################
#################################preparing the sentence-character level  branch###################################


char_embedding_layer = tf.keras.layers.Embedding(input_dim=len(char_index) + 1,
                            output_dim = CHAR_EMBEDDING_DIM,
                            input_length=char_data.shape[1],
                            trainable=True)

sentence_char_input_tensor = tf.keras.layers.Input(shape=(CHAR_MAX_SENTENCE_LENGTH,), dtype='int32', name ="char_input")
embedded_sentence_char_sequence = char_embedding_layer(sentence_char_input_tensor)
x_char = tf.keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'valid', activation='relu')(embedded_sentence_char_sequence)
x_char = tf.keras.layers.MaxPooling1D(pool_size=3)(x_char)
x_char = tf.keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'valid', activation='relu')(x_char)
x_char = tf.keras.layers.MaxPooling1D(pool_size=3)(x_char)
x_char = tf.keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'valid', activation='relu')(x_char)
# x = tf.keras.layers.MaxPooling1D(pool_size=35)(x)  # global max pooling
x_char_out_tensor = tf.keras.layers.Flatten()(x_char)




##################################################################################################################
##################################################################################################################


##################################################################################################################
#################################preparing the word-character level  branch###################################


#prepare the word-char branch
word_char_input_tensor = tf.keras.layers.Input(shape=(word_data.shape[1] * max_word_length,),
                                                 dtype='int32', name ="word_char_input")
embedded_word_char_sequence = char_embedding_layer(word_char_input_tensor)
#apply 1d conv to aggregate along words. stride is precisely the max word length in chars
x_word_char = tf.keras.layers.Conv1D(filters=EMBEDDING_DIM, kernel_size = max_word_length, strides= max_word_length, activation = 'relu')(embedded_word_char_sequence)
#output should be max_sentence_length x EMBEDDING_DIM where each row represents the char level representation of the words
# this represents a new char level word embedding of the same size as the one used via glove or word2vec


##################################################################################################################
##################################################################################################################


##################################################################################################################
#################################preparing the word level  branch###################################

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

word__input_tensor = tf.keras.layers.Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32', name ="word_input")
#add non trainable word embedding sequence
word_embedded__glove = word_embedding_layer_glove(word__input_tensor)
word_embedded__trainable = word_embedding_layer_trainable(word__input_tensor)


##################################################################################################################
##################################################################################################################


##################################################################################################################
################################# merge the 3 word level inputs                 ###################################
x_word1 = tf.keras.layers.add( [word_embedded__glove, word_embedded__trainable, x_word_char] )
#x_word = tf.concat( [word_embedded__glove, word_embedded__trainable, x_word_char], axis=-1 )
x_word = tf.keras.layers.Concatenate(axis=-1)( [word_embedded__glove, word_embedded__trainable, x_word_char])
input_shape = x_word.shape
output_shape = [int(input_shape[1]),  3, int(input_shape[2]//3)]
x_word = tf.keras.layers.Reshape(target_shape=output_shape)(x_word)
x_word = tf.keras.layers.Permute( dims = [1,3,2])(x_word)
#this actually should be equivalent to
#need the work around since tf.keras.backend.stack has an issue and disrupts the graph
#x_word_test = tf.keras.backend.stack( [word_embedded__glove, word_embedded__trainable, x_word_char], axis=-1)
#assert x_word==x_word_test

#setattr(x_word, '_keras_history', getattr(x_word1, '_keras_history'))
#setattr(x_word, '_keras_history', getattr(word_embedded__trainable, '_keras_history'))
#setattr(x_word, '_keras_history', getattr(x_word_char, '_keras_history'))
#setattr(x_word, '_keras_shape', getattr(right_branch.output, '_keras_shape'))
#setattr(x_word, '_uses_learning_phase', getattr(x_word1, '_uses_learning_phase'))






#do some convolutions
#todo - implement inception layer here to look at 1-2-3-4-grams

x_word = tf.keras.layers.Conv2D(filters = 64, kernel_size = (2,100), padding = 'valid', activation='relu')(x_word)

x_word_out = tf.keras.layers.Flatten()(x_word)


##################################################################################################################
##################################################################################################################


##################################################################################################################
################################# merge the word and char  level inputs        ###################################




x = tf.keras.layers.concatenate([x_char_out_tensor, x_word_out])


x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
preds_x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
#preds = tf.keras.layers.Dense(len(labels_index), activation='softmax')(x)   char_sequence_input,
model = tf.keras.models.Model(inputs=[ sentence_char_input_tensor, word_char_input_tensor, word__input_tensor],
                              outputs=[preds_x])



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



