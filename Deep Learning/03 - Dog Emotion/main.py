from keras import models
from keras import layers
from keras import regularizers
from keras import activations
from keras import initializers
from keras import optimizers
from keras import losses
from keras import metrics
from keras import callbacks
from keras.preprocessing import image


model = models.Sequential()

model.add(layers.Conv2D(
    32, (5,5),
    input_shape = (64, 64, 3),
    activation='relu'
))
model.add(layers.MaxPooling2D(
    pool_size=(2,2)
))
model.add(layers.Conv2D(
    16, (5,5),
    input_shape = (30, 30, 3),
    activation='relu'
))
model.add(layers.MaxPooling2D(
    pool_size=(2,2)
))

model.add(layers.Conv2D(
    8, (5,5),
    input_shape = (13, 13, 3),
    activation='relu'
))
model.add(layers.MaxPooling2D(
    pool_size=(2,2)
))

model.add(layers.Conv2D(
    4, (3,3),
    input_shape = (5, 5, 3),
    activation='relu'
))
model.add(layers.MaxPooling2D(
    pool_size=(2,2)
))

model.add(layers.Flatten())

model.add(layers.Dense(
    256,
    kernel_regularizer = regularizers.L2(1e-4),
    kernel_initializer=initializers.RandomNormal(stddev=1),
    bias_initializer=initializers.Zeros()
))
model.add(layers.Activation(activations.relu))

model.add(layers.Dense(
    64,
    kernel_regularizer = regularizers.L2(1e-4),
    kernel_initializer=initializers.RandomNormal(stddev=1),
    bias_initializer=initializers.Zeros()
))
model.add(layers.Activation(activations.relu))
model.add(layers.Dense(
    128,
    kernel_regularizer = regularizers.L2(1e-4),
    kernel_initializer=initializers.RandomNormal(stddev=1),
    bias_initializer=initializers.Zeros()
))
model.add(layers.Activation(activations.relu))

model.add(layers.Dense(
    64,
    kernel_regularizer = regularizers.L2(1e-4),
    kernel_initializer=initializers.RandomNormal(stddev=1),
    bias_initializer=initializers.Zeros()
))
model.add(layers.Activation(activations.relu))

model.add(layers.Dense(
    32,
    kernel_initializer=initializers.RandomNormal(stddev=1),
    bias_initializer=initializers.Zeros()
))
model.add(layers.Activation(activations.relu))

model.add(layers.Dense(
    16,
    kernel_initializer=initializers.RandomNormal(stddev=1),
    bias_initializer=initializers.Zeros()
))
model.add(layers.Activation(activations.relu))

model.add(layers.Dense(
    4,
    kernel_initializer=initializers.RandomNormal(stddev=1),
    bias_initializer=initializers.Zeros()
))
model.add(layers.Activation(activations.softmax))

model.compile(
    optimizer=optimizers.SGD(learning_rate=0.003),
    loss=losses.BinaryCrossentropy(),
    metrics=["acc"]
)

data_gen = image.ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.2
)

X_train = data_gen.flow_from_directory(
    "photos",
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

X_tests = data_gen.flow_from_directory(
    "photos",
    target_size=(64, 64),
    batch_size=32, 
    class_mode="categorical",
    subset="validation"
)

# 3200
# 800
model.fit(
    X_train, 
    steps_per_epoch=100,
    epochs=80, 
    validation_steps=10,
    callbacks=[
        callbacks.ModelCheckpoint(
            filepath='./models/model.{epoch:02d}.h5'
        )
    ]
)