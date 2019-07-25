from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D, Conv2D, GlobalAveragePooling2D, Embedding, LSTM, Bidirectional, Reshape, GlobalMaxPooling2D, MaxPooling1D, concatenate, Flatten
from keras.initializers import TruncatedNormal
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2

# generic model design
def model_fn(actions):
    '''
    # unpack the actions from the list
    kernel_1, filters_1, kernel_2, filters_2, kernel_3, filters_3, kernel_4, filters_4 = actions

    ip = Input(shape=(32, 32, 3))
    x = Conv2D(filters_1, (kernel_1, kernel_1), strides=(2, 2), padding='same', activation='relu')(ip)
    x = Conv2D(filters_2, (kernel_2, kernel_2), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters_3, (kernel_3, kernel_3), strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(filters_4, (kernel_4, kernel_4), strides=(1, 1), padding='same', activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(model)
    '''

    emb_1, bidirect_lstm_1, filter_1, kernel_1, emb_2, bidirect_lstm_2, filter_2, kernel_2 = actions
    model = Sequential()
    model.add(Embedding(6863, emb_1, input_length=30, name='emb'))
    # model.add(LSTM(bidirect_lstm_1))

    # sequential network
    #model.add(Bidirectional(LSTM(bidirect_lstm_1), name='bilstm1'))

    # convolutional network
    '''
    model.add(Reshape((12, -1, 1)))
    model.add(Conv2D(filter_1, [kernel_1, kernel_1], strides=(1, 1), padding='same', activation='relu', name='conv'))
    model.add(GlobalAveragePooling2D())
    '''

    # stack bilstm + convolutional network
    model.add(LSTM(bidirect_lstm_1, return_sequences=True, name='lstm1'))

#    model.add(Bidirectional(LSTM(bidirect_lstm_1, return_sequences=True), name='bilstm1'))
#    model.add(Bidirectional(LSTM(bidirect_lstm_2), name='lstm2'))
#    model.add(Reshape((-1, 1, 1)))
#    model.add(Conv2D(filter_1, [kernel_1, kernel_1], strides=(1, 1), padding='same', activation='relu', name='conv'))
#    model.add(GlobalAveragePooling2D())
#    model.add(GlobalMaxPooling2D())
    model.add(Dense(22, activation='sigmoid', name='dense'))
    for layer in model.layers:
        print(layer.output_shape)

    return model

def model_fn_new(actions):
    # unpack actions
    emb_size_1, lstm_size_1, filter_num_1, kernel_height_1, pool_weight_1, fc_size_1, vocab_size_1, max_seq_len_1, label_num_1, type_1, \
        emb_size_2, lstm_size_2, filter_num_2, kernel_height_2, pool_weight_2, fc_size_1, vocab_size_2, max_seq_len_2, label_num_2, type_2, \
        emb_size_3, lstm_size_3, filter_num_3, kernel_height_3, pool_weight_3, fc_size_1, vocab_size_3, max_seq_len_3, label_num_3, type_3 = actions

    print("model_type:{}, seq_len:{}, emb_size:{}, label_num:{}, vocab_size:{}".format(type_1, max_seq_len_1, emb_size_1, label_num_1, vocab_size_1))
    if type_1 == "textcnn":
        # embedding layer
        m_inputs = Input(shape=(max_seq_len_1, ), name="embedding_input")
        print("m_inputs: {}".format(m_inputs.shape))
        embedder = Embedding(vocab_size_1, emb_size_1, input_length=max_seq_len_1, name="embedding")
        embed_inputs = embedder(m_inputs)
        print("embed_inputs: {}".format(embed_inputs.shape))

        # converlution layer, kernel=[kernel_height, kernel_height+1, kernel_height+2]
        cnn_1 = Conv1D(filter_num_1, kernel_height_1, strides=1, padding="valid", activation="relu", kernel_initializer=TruncatedNormal(), use_bias=True, name="cnn_1")(embed_inputs)
        print("cnn_1 result: {}".format(cnn_1.shape))
        cnn_1 = MaxPooling1D(pool_size=(max_seq_len_1-kernel_height_1+1))(cnn_1)
        print("pooling_1 result: {}".format(cnn_1.shape))
        cnn_2 = Conv1D(filter_num_2, kernel_height_1+1, strides=1, padding="valid", activation="relu", kernel_initializer=TruncatedNormal(), use_bias=True, name="cnn_2")(embed_inputs)
        print("cnn_2 result: {}".format(cnn_2.shape))
        cnn_2 = MaxPooling1D(pool_size=(max_seq_len_1-kernel_height_1))(cnn_2)
        print("pooling_2 result: {}".format(cnn_2.shape))
        cnn_3 = Conv1D(filter_num_3, kernel_height_1+2, strides=1, padding="valid", activation="relu", kernel_initializer=TruncatedNormal(), use_bias=True, name="cnn_3")(embed_inputs)
        print("cnn_3 result: {}".format(cnn_3.shape))
        cnn_3 = MaxPooling1D(pool_size=(max_seq_len_1-kernel_height_1-1))(cnn_3)
        print("pooling_3 result: {}".format(cnn_3.shape))

        # concat pooling result
        cnn = concatenate([cnn_1, cnn_2, cnn_3], axis=-1)
        print("cnn result: {}".format(cnn.shape))
        # flat = Reshape((-1, cnn.shape[-1]))(cnn)
        flat = Flatten()(cnn)
        print("flat result: {}".format(flat.shape))
        m_outputs = Dense(label_num_1, activation="softmax", name="y_prob")(flat)
        model = Model(m_inputs, m_outputs)
    else:
        model = Sequential()
        # embedding layer
        model.add(Embedding(vocab_size_1, emb_size_1, input_length=max_seq_len_1, name="embedding"))

        if type_1 == "lenet":
            # cnn layers
            model.add(Conv1D(filter_num_1, kernel_height_1, strides=1, padding="same", activation="relu", kernel_initializer=TruncatedNormal(), kernel_regularizer=l2(0.01), use_bias=True, bias_initializer=TruncatedNormal(), name="cnn_1"))
            model.add(MaxPooling1D(pool_size=pool_weight_1, strides=1, padding="same"))
            model.add(Conv1D(filter_num_2, kernel_height_2, strides=1, padding="same", activation="relu", kernel_initializer=TruncatedNormal(), kernel_regularizer=l2(0.01), use_bias=True, bias_initializer=TruncatedNormal(),  name="cnn_2"))
            model.add(MaxPooling1D(pool_size=pool_weight_2, strides=1, padding="same"))
            model.add(Conv1D(filter_num_3, kernel_height_3, strides=1, padding="same", activation="relu", kernel_initializer=TruncatedNormal(), kernel_regularizer=l2(0.01), use_bias=True, bias_initializer=TruncatedNormal(), name="cnn_3"))
            # flatten
            model.add(Flatten())
            # bn
            model.add(BatchNormalization())
            # full-connect
            model.add(Dense(fc_size_1, kernel_initializer=TruncatedNormal(), kernel_regularizer=l2(0.01), use_bias=True, bias_initializer=TruncatedNormal(), activation="relu"))
        elif type_1 == "lstm":
            model.add(LSTM(lstm_size_1, name='lstm'))
        elif type_1 == "bilstm":
            model.add(Bidirectional(LSTM(lstm_size_1), name='bilstm'))
        elif type_1 == "lstm+bilstm":
            model.add(LSTM(lstm_size_1, return_sequences=True, name='lstm1'))
            model.add(Bidirectional(LSTM(lstm_size_2), name='bilstm1'))
        # output
        model.add(Dense(label_num_1, activation="softmax", name="y_prob"))

    for layer in model.layers:
        print(layer)
        print(layer.output_shape)
    print(model.input.op.name)
    print(model.output.op.name)
    return model

def model_fn_nlp(actions):
    input = Input(shape=())

