##ImageEmbedding##
model = Sequential()
model.add(Conv2D(32, (5, 5), padding='same',
                 input_shape=trainImages.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (5, 5),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(numClass))
model.add(Activation('softmax'))
##TextEmbedding##
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(5001, embedding_vecor_length, input_length=max_caption_length, mask_zero=True,name = 'embedding'))
model.add(LSTM(100,name = 'LSTM'))
model.add(Dense(90, activation='softmax',name = 'Dense'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

