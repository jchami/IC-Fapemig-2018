# PIBIC-FAPEMIG
# Copyright (C) 2018/19  Universidade Federal de Uberlândia
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from scipy.misc     import imsave
from keras          import metrics
from PIL            import Image

from keras.models                      import Model
from keras.models                      import model_from_json
from keras.utils.np_utils              import to_categorical

import keras.backend     as K
import numpy             as np
import matplotlib.pyplot as plt

import warnings
from show_images import show_images

warnings.filterwarnings('ignore')

def limit_mem():
    cfg                          = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config = cfg))

limit_mem()

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# Load model from JSON
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# Load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

filename = 'gatto.jpg'
original_pic   = Image.open(filename).resize((32, 32))
original_array = np.expand_dims(np.array(original_pic), 0)

plt.imshow(original_array[0])
original_array = np.swapaxes(original_array, 2, 3)
original_array = np.swapaxes(original_array, 1, 2)

pred = model.predict(original_array)
preds = [(class_names[i], pred[0][i]) for i in range(10)]

print(preds)

target_idx      = model.predict(original_array).argmax()
target          = to_categorical(target_idx, 10)
target_variable = K.variable(target)
loss            = metrics.categorical_crossentropy(model.output, target_variable)
gradients       = K.gradients(loss, model.input)
get_grad_values = K.function([model.input], gradients)
grad_values     = get_grad_values([original_array])[0]
grad_signs      = np.sign(grad_values)

epsilon         = 10000
perturbation    = grad_signs * epsilon
modified_array  = original_array + perturbation
modified_array  = modified_array.astype(np.uint8)
modified_pred   = model.predict(modified_array)
modified_preds  = [(class_names[i], modified_pred[0][i]) for i in range(10)]
print(preds)
print(modified_preds)

original_array = np.swapaxes(original_array, 1, 2)
original_array = np.swapaxes(original_array, 2, 3)
perturbation = np.swapaxes(perturbation, 1, 2)
perturbation = np.swapaxes(perturbation, 2, 3)
perturbation = perturbation.astype(np.uint8)
modified_array = np.swapaxes(modified_array, 1, 2)
modified_array = np.swapaxes(modified_array, 2, 3)
show_images([original_array[0], perturbation[0], modified_array[0]], 2)