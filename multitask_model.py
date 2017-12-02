import numpy as np
import os
import tensorflow as tf
import keras as keras
import math


from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}

    def create_functional_model(self):
        """Get a multitask functional model"""

        batch_size = 32
        max_sentence_len_in_batch = 20 # 20  is for testing purposes. replace with None in prod
        max_word_len_in_batch = 15 # again replace with None in prod
        CHAR_EMBEDDING_DIM = 50
        CHAR_VOCAB_SIZE = 80
        CHAR_EMBEDDINGS_biLSTM = True
        CHAR_LSTM_SIZE = 60

        WORD_VOCAB_SIZE = 10000
        WORD_EMBEDDING_DIM = 300

        USE_CONVNET = True
        USE_CHAR_DIM_REDUCTION = True
        CHAR_DIM_REDUCED = 20
        NB_CONV_FILTERS_WORD_BOTTLENECK = 16
        NB_CONV_FILTERS_CHAR_BOTTLENECK = 16
        NB_CONV_FILTERS_WORD = 16
        NB_CONV_FILTERS_CHAR = 16

        DROPAUT_W_merged = 0.25
        DROPAUT_U_merged = 0.25

        MERGED_LSTM_SIZE = 100

        NB_LABEL_KEYS = 5


        # WORDS EMBEDDINGS
        embedding_matrix = np.zeros((WORD_VOCAB_SIZE , WORD_EMBEDDING_DIM))
        for  i in range(WORD_VOCAB_SIZE): # for word, i in word_index.items():
            embedding_vector = np.random.rand(WORD_EMBEDDING_DIM)  # embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector


        #shape: max sentence length_in_batch
        word_inputs = tf.keras.Input(shape=(max_sentence_len_in_batch, ) )

        #########################################################################
        # output shape: batch_size , sentence_length, WORD_EMBEDDING_DIM
        word_embeddings = tf.keras.layers.Embedding (
            input_dim=WORD_VOCAB_SIZE, output_dim=WORD_EMBEDDING_DIM,
            weights=[embedding_matrix], trainable=False, input_length = max_sentence_len_in_batch)(word_inputs)
        ########################################################################


        # CHAR EMBEDDINGS
        char_inputs = tf.keras.Input(shape = (max_sentence_len_in_batch, max_word_len_in_batch) )
        #embed chars
        ###########################################################################
        # output shape: batch_size , sentence_length, word_length, CHAR_EMBEDDING_DIM
        u_limit = math.sqrt(3.0 / CHAR_EMBEDDING_DIM)
        char_embeddings = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Embedding(input_dim= CHAR_VOCAB_SIZE, output_dim=CHAR_EMBEDDING_DIM,input_length= max_word_len_in_batch,
                                      embeddings_initializer= tf.keras.initializers.RandomUniform(minval=-u_limit, maxval=u_limit),
                                      embeddings_regularizer=None, activity_regularizer=None) )(char_inputs)
        ############################################################################




        if USE_CHAR_DIM_REDUCTION:
            # reduce char embeddings dimension
            # output shape:   batch_size , sentence_length, CHAR_DIM_REDUCED
            char_embeddings = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D( filters=CHAR_DIM_REDUCED,
                                                   kernel_size=2, strides=1,
                                                   padding='same'))(char_embeddings)

        ############################################################################
        # flatten last two dimensions
        # output shape: batch_size , sentence_length, word_length x CHAR_EMBEDDING_DIM
         # in keras you do not specify the batch dimension
        #tmp_shape = tf.shape(char_embeddings)
        #char_embeddings = keras.layers.Reshape(
        #    target_shape=[max_sentence_len_in_batch, max_word_len_in_batch * CHAR_DIM_REDUCED])(char_embeddings)
        ############################################################################


        ###########################################################################
        # input into the LSTM: word_length  CHAR_EMBEDDING_DIM
        # output shape: batch_size , sentence_length,  CHAR_LSTM_SIZE
        # THIS RUNS INTO PROBLEM - bug in tf integration into tf
        #if CHAR_EMBEDDINGS_biLSTM:
        #    char_embeddings_layer = tf.keras.layers.TimeDistributed(keras.layers.Bidirectional(
        #        tf.keras.layers.LSTM(units = CHAR_LSTM_SIZE, return_sequences=False )) )
        #    char_embeddings = char_embeddings_layer(char_embeddings)
        ###########################################################################
        #REPLACE WITH
        if CHAR_EMBEDDINGS_biLSTM:
            s=tf.shape(char_embeddings)
            char_embeddings = tf.keras.backend.reshape(char_embeddings, (s[0]*max_sentence_len_in_batch,max_word_len_in_batch, CHAR_DIM_REDUCED))
            char_embeddings_layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units = CHAR_LSTM_SIZE, return_sequences=False ))
            char_embeddings = char_embeddings_layer(char_embeddings)
            char_embeddings = tf.keras.backend.reshape(char_embeddings, (s[0] , max_sentence_len_in_batch, 2*CHAR_LSTM_SIZE))




        #CONV NET. todo: add regularization and dropout
        if USE_CONVNET:
            #INCEPTION CONV NET ON WORDS:
            # output shape: batch_size , sentence_length, NB_CONV_FILTERS_WORD
            tower1w = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_WORD_BOTTLENECK,
                                                   kernel_size=1, strides=1,
                                                   padding='same')(word_embeddings)
            tower1w = tf.keras.layers.Conv1D(filters = NB_CONV_FILTERS_WORD,
                                             kernel_size = 2,
                                             strides = 1, activation='relu',
                                             padding = 'same')(tower1w)

            # output shape: batch_size , sentence_length, NB_CONV_FILTERS_WORD
            tower2w = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_WORD_BOTTLENECK,
                                                    kernel_size=1, strides=1,
                                                    padding='same')(word_embeddings)
            tower2w = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_WORD,
                                             kernel_size=3,
                                             strides=1, activation='relu',
                                             padding='same')(tower2w)
            # output shape: batch_size , sentence_length, NB_CONV_FILTERS_WORD
            tower3w = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_WORD_BOTTLENECK,
                                                    kernel_size=1, strides=1,
                                                    padding='same')(word_embeddings)
            tower3w = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_WORD,
                                             kernel_size=4,
                                             strides=1, activation='relu',
                                             padding='same')(tower3w)
            # output shape: batch_size , sentence_length, NB_CONV_FILTERS_WORD
            tower4w = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_WORD_BOTTLENECK,
                                                    kernel_size=1, strides=1,
                                                    padding='same')(word_embeddings)
            tower4w = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_WORD,
                                             kernel_size=5,
                                             strides=1, activation='relu',
                                             padding='same')(tower4w)

            # output shape: batch_size , sentence_length, NB_CONV_FILTERS_WORD
            tower5w = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_WORD,
                                             kernel_size=1,
                                             strides=1, activation='relu',
                                             padding='same')(word_embeddings)

            # output shape: batch_size , sentence_length, 5 x NB_CONV_FILTERS_WORD
            word_inception_out = tf.keras.layers.concatenate([tower1w, tower2w, tower3w, tower4w, tower5w], axis=-1)



            # INCEPTION CONV NET ON CHARS:
            # output shape: batch_size , sentence_length, NB_CONV_FILTERS_CHAR
            tower1c = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_CHAR_BOTTLENECK,
                                      kernel_size=1, strides=1,
                                      padding='same')(char_embeddings)
            tower1c = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_CHAR,
                                      kernel_size=2,
                                      strides=1, activation='relu',
                                      padding='same')(tower1c)

            # output shape: batch_size , sentence_length, NB_CONV_FILTERS_CHAR
            tower2c = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_CHAR_BOTTLENECK,
                                      kernel_size=1, strides=1,
                                      padding='same')(char_embeddings)
            tower2c = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_CHAR,
                                      kernel_size=3,
                                      strides=1, activation='relu',
                                      padding='same')(tower2c)
            # output shape: batch_size , sentence_length, NB_CONV_FILTERS_CHAR
            tower3c = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_CHAR_BOTTLENECK,
                                      kernel_size=1, strides=1,
                                      padding='same')(char_embeddings)
            tower3c = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_CHAR,
                                      kernel_size=4,
                                      strides=1, activation='relu',
                                      padding='same')(tower3c)
            # output shape: batch_size , sentence_length, NB_CONV_FILTERS_CHAR
            tower4c = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_CHAR_BOTTLENECK,
                                      kernel_size=1, strides=1,
                                      padding='same')(char_embeddings)
            tower4c = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_CHAR,
                                      kernel_size=5,
                                      strides=1, activation='relu',
                                      padding='same')(tower4c)

            # output shape: batch_size , sentence_length, NB_CONV_FILTERS_CHAR
            tower5c = tf.keras.layers.Conv1D(filters=NB_CONV_FILTERS_CHAR,
                                      kernel_size=1,
                                      strides=1, activation='relu',
                                      padding='same')(char_embeddings)

            # output shape: batch_size , sentence_length, 5 x NB_CONV_FILTERS_CHAR
            char_inception_out = tf.keras.layers.concatenate([tower1c, tower2c, tower3c, tower4c, tower5c], axis=-1)

            # output shape: batch_size , sentence_length, 5 x NB_CONV_FILTERS_WORD + 5 x NB_CONV_FILTERS_CHAR
            merged_embedding = tf.keras.layers.concatenate( [word_inception_out, char_inception_out] , axis = -1)

        else:
            # output shape: batch_size , sentence_length, WORD_EMBEDDING_DIM + CHAR_EMBEDDING_DIM /  CHAR_LSTM_SIZE
            merged_embedding = tf.keras.layers.Concatenate([word_embeddings, char_embeddings], axis = -1 )


        # add a biLSTM  layer
        # output shape: batch_size , sentence_length, 2xMERGED_LSTM_SIZE_
        x = tf.keras.layers.Bidirectional( tf.keras.layers.LSTM (units = MERGED_LSTM_SIZE, dropout= DROPAUT_W_merged,
                                                     recurrent_dropout=DROPAUT_U_merged, return_sequences=True ) )(merged_embedding)


        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense (units =  MERGED_LSTM_SIZE//2,
                                                       activation='relu' )) (x)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=NB_LABEL_KEYS))(x)

        DECODER = ('LSTM', (0.25,0.25)) # 'tanh_crf'  'crf' 'softmax'
        if DECODER == 'softmax':
            output = tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense( units= NB_LABEL_KEYS, activation='softmax'))(x)

            lossFct = 'sparse_categorical_crossentropy'
        elif DECODER == 'crf':
            output = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense( units= NB_LABEL_KEYS, activation=None))(x)
            crf = ChainCRF('_CRF')
            output = crf(output)
            lossFct = crf.sparse_loss
        elif DECODER == 'tanh_crf':
            output = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=NB_LABEL_KEYS, activation='tanh'))(x)
            crf = ChainCRF('tanh_CRF')
            output = crf(output)
            lossFct = crf.sparse_loss
        #elif DECODER == 'tf_crf':
        #    output = tf.keras.layers.TimeDistributed(
        #        tf.keras.layers.Dense(units=NB_LABEL_KEYS, activation=None))(x)
        #    log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
        #        output, batch_labels, batch_sequence_lengths)
        #    self.trans_params = trans_params  # need to evaluate it for decoding
        #    self.loss = tf.reduce_mean(-log_likelihood)

        elif DECODER[0] == 'LSTM':
            size = DECODER[1]
            output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(size, return_sequences=True,
                                                                dropout=DECODER[1][0],
                                                                recurrent_dropout=DECODER[1][1]))(x)




    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                #print("char lstm output 1", tf.shape(_output))
                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)
                #print("char lstm output 2", tf.shape(output))
                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                #print("char lstm output 3", tf.shape(output))
                #print("word embeddings shape", tf.shape(word_embeddings))
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])


    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)


    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]


    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1}


    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
