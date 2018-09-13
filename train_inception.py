from keras import optimizers
from keras.layers import Dense
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from utils_data import GetGens

img_size = 299

train_data_dir = 'Data\\train'
validation_data_dir = 'Data\\validation'
epochs = 1
batch_size = 32

def GetModel():
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    x = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model, base_model
    

model, base_model = GetModel()
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.03, momentum=0.9),
              metrics=['accuracy'])



gen_train, gen_val = GetGens(size=img_size)


train_generator = gen_train.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = gen_val.flow_from_directory(
    validation_data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical')


# fine-tune the model
model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator)
