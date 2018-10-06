import keras

from dataset import load_data
from utils import preprocess_input
from models import DeepYeast
import os
import argparse
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from guassian_loss import build_gmd_log_likelihood
parser = argparse.ArgumentParser()
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--epoch", type=int, default=300)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--opt", type=str, default='adam')
parser.add_argument("--model", type=str, default='deepyeast')

args = parser.parse_args()
batch_size = args.batchsize
epoch = args.epoch
opt = args.opt
lr = args.lr
model_type = args.model

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('examples'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)
TENSORBOARD_DIR = os.path.join(ROOT_DIR, 'tensorboard_logs')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoint')
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'model_weights')
JSON_DIR = os.path.join(ROOT_DIR, 'json_files')

# set up data
x_val, y_val = load_data("val")
x_train, y_train = load_data("train")

num_classes = 12
y_val = keras.utils.to_categorical(y_val, num_classes)
y_train = keras.utils.to_categorical(y_train, num_classes)

x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# set up model
if model_type == 'deepyeast':
    model = DeepYeast()
if opt == 'adam':
    optimizer = keras.optimizers.Adam()
elif opt == 'sgd':
    optimizer = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)

model.compile(loss=build_gmd_log_likelihood(),
              optimizer=optimizer,
              metrics=['accuracy'])

filepath= os.path.join(CHECKPOINT_DIR, model_type + '_' + opt +'_gloss'+ "_weights-{epoch:02d}-{val_acc:.3f}.h5")
earlystop_callback = EarlyStopping(monitor='val_loss',
                                       patience=5,
                                       min_delta=0.01)
tensorboard_callback = TensorBoard(os.path.join(TENSORBOARD_DIR, model_type + '_' + opt +'_gloss'+'_tb_logs'))
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5, cooldown=0, min_lr=1e-5)
callbacks_list = [checkpoint, reduce_lr, tensorboard_callback]


# training loop
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epoch,
          validation_data=(x_val, y_val),
          callbacks=callbacks_list)


model_json = model.to_json()
with open(os.path.join(JSON_DIR, model_type + '_' + opt +'_gloss'+'.json'), 'w') as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(WEIGHTS_DIR, model_type + '_' + opt +'_gloss'+ '.h5'))
print('deepyeast model has been saved')
