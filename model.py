from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv1D, Conv2D, GlobalAveragePooling2D, Embedding, LSTM, Bidirectional, Reshape

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
    model.add(Embedding(3715, emb_1, input_length=12, name='emb'))
    model.add(Bidirectional(LSTM(bidirect_lstm_1), name='bilstm'))
    model.add(Reshape((-1, 1, 1)))
    model.add(Conv2D(filter_1, [kernel_1, kernel_1], strides=(1, 1), padding='same', activation='relu', name='conv'))
    model.add(GlobalAveragePooling2D())
    #model.add(Bidirectional(LSTM(bidirect_lstm_2), name='bilstm2'))
    #model.add(Reshape((-1, 1, 1)))
    #model.add(Conv2D(filter_2, [kernel_2, kernel_2], strides=(1, 1), padding='same', activation='relu', name='conv2'))
    #model.add(GlobalAveragePooling2D())
    model.add(Dense(22, activation='sigmoid', name='dense'))
    for layer in model.layers:
        print(layer.output_shape)

    return model



def model_fn_nlp(actions):
    input = Input(shape=())

