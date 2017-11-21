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
def conv1d_relu(input, kernel_shape, bias_shape, stride, padding , name = 'name'):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.truncated_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv1d(value = input, filters = weights, stride = stride, name = name, padding = padding)
    return tf.nn.relu(conv + biases)




##################################################################################################################
#################################preparing the sentence-character level  branch###################################


###### create relevant placeholders ####
# sentence-char data
size_of_input_sentence_char_data = char_data.shape[1]
sentence_char_data_placeholder = tf.placeholder(dtype=tf.float32, shape= [None,size_of_input_sentence_char_data ])
# sentence-word data
size_of_input_sentence_word_data = word_data.shape[1]
word_data_placeholder = tf.placeholder(dtype=tf.float32, shape= [None,size_of_input_sentence_word_data ])
# char-word data
size_of_input_char_word_data = word_char_input_flat.shape[1]
char_word_data_placeholder = tf.placeholder(dtype=tf.float32, shape= [None,size_of_input_char_word_data ])
######

# create embedding layers

# character embeddings variable. size is: char_vocabulaty , char embedding dimensions
embedding_char_weights = tf.get_variable("embedding_char_weights",shape = [len(char_index) + 1, CHAR_EMBEDDING_DIM],
        initializer=tf.truncated_normal_initializer(), name = 'embedding_char_weights')

embedded_sentence_char_sequence  = tf.nn.embedding_lookup(params = embedding_char_weights,
                                                          ids = sentence_char_data_placeholder,


x_char = tf.keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'valid', activation='relu')(embedded_sentence_char_sequence)
x_char = tf.keras.layers.MaxPooling1D(pool_size=3)(x_char)
x_char = tf.keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'valid', activation='relu')(x_char)
x_char = tf.keras.layers.MaxPooling1D(pool_size=3)(x_char)
x_char = tf.keras.layers.Conv1D(filters = 64, kernel_size = 5, padding = 'valid', activation='relu')(x_char)
# x = tf.keras.layers.MaxPooling1D(pool_size=35)(x)  # global max pooling
x_char_out_tensor = tf.keras.layers.Flatten()(x_char)

                                                          name = 'embedded_sentence_char_sequence' )
#apply some convs on the sentence char data
x_char = tf.layers.conv1d(inputs = embedded_sentence_char_sequence, filters= 64,
                          kernel_size= 5, padding='valid', activation='relu' )
x_char = tf.layers.max_pooling1d( pool_size= 3, strides = 1, padding = 'valid')

x_char = tf.layers.conv1d(inputs = embedded_sentence_char_sequence, filters= 64,
                          kernel_size= 5, padding='valid', activation='relu' )
x_char = tf.layers.max_pooling1d( pool_size= 3, strides = 3, padding = 'valid')
x_char = tf.layers.conv1d(inputs = embedded_sentence_char_sequence, filters= 64,
                          kernel_size= 5, padding='valid', activation='relu' )
x_char = tf.layers.max_pooling1d( pool_size= 3, strides = 3, padding = 'valid')
x_char_out_tensor = tf.layers.Flatten(x_char)

#word-char-embeddings
embedded_word_char_sequence1  = tf.nn.embedding_lookup(params = embedding_char_weights,
                                                          ids = char_word_data_placeholder,
                                                          name = 'embedded_word_char_sequence1' )
#apply conv1d to get into right shape to stack on the other word embeddings


nb_filters = EMBEDDING_DIM
filter_dim1 = max_word_length
filter_dim2 = CHAR_EMBEDDING_DIM
#embedded_word_char_sequence = conv1d_relu(input = embedded_word_char_sequence1,
#                                          kernel_shape= [max_word_length, CHAR_EMBEDDING_DIM, EMBEDDING_DIM],
#                                          bias_shape = [EMBEDDING_DIM], stride = max_word_length,
#                                          name = 'embedded_word_char_sequence', padding = 'valid')

embedded_word_char_sequence = tf.layers.conv1d(inputs= embedded_word_char_sequence1, filters= max_word_length,
                                               kernel_size = max_word_length, strides= max_word_length,
                                               activation='relu')

#test: should have the same dimension as:
embedded_word_char_sequence_test =   tf.keras.layers.Conv1D(filters=EMBEDDING_DIM, kernel_size = max_word_length, strides= max_word_length, activation = 'relu')(embedded_word_char_sequence)





# word embeddings: one trainable and one fixed - say glove
embedding_word_weights_trainable = tf.get_variable("embedding_word_weights_trainable",
                                                    shape = [len(word_index) + 1, EMBEDDING_DIM],
                                                    initializer=tf.truncated_normal_initializer(),
                                                   name='embedding_word_weights_trainable')


embedding_word_weights_glove = tf.get_variable("embedding_word_weights_glove",
                                                shape = [len(word_index) + 1, EMBEDDING_DIM],
                                                initializer=tf.constant_initializer(embedding_matrix),
                                                trainable=False, name='embedding_word_weights_glove')


embedded_word_sequence_glove  = tf.nn.embedding_lookup(params = embedding_word_weights_glove,
                                                          ids = word_data_placeholder,
                                                          name = 'embedded_word_sequence_glove' )

embedded_word_sequence_trainable  = tf.nn.embedding_lookup(params = embedding_word_weights_trainable,
                                                          ids = word_data_placeholder,
                                                          name = 'embedded_word_sequence_trainable' )


# stack the three  word embeddings into three channels to use convolutions on them

embedded_word_tensor = tf.stack(values=[embedded_word_sequence_glove, embedded_word_sequence_trainable, embedded_word_char_sequence], axis = -1, name = 'embedded_word_tensor'  )


def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

#do some convolutions
#todo - implement inception layer here to look at 1-2-3-4-grams

x_word =  tf.layers.conv2d(inputs = embedded_word_tensor, filters = 64, kernel_size = (2,100), padding = 'valid', activation='relu' )

x_word_out_tensor = tf.layers.Flatten()(x_word)

x = tf.concat([x_word_out_tensor, x_char_out_tensor ])



x = tf.layers.Dense(128, activation='relu')(x)
x = tf.layers.Dense(32, activation='relu')(x)
preds_x = tf.layers.Dense(1, activation='sigmoid')(x)
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



