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

from keras import metrics
from PIL import Image

from keras.models import model_from_json
from keras.utils.np_utils import to_categorical

import keras.backend as K
import numpy as np
# import matplotlib.pyplot as plt
import seaborn
from show_images import show_images

import warnings

warnings.filterwarnings('ignore')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def load_model(model_name='model.json', h5_name='model.h5'):
    # Load model from JSON
    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # Load weights into new model
    model.load_weights(h5_name)
    print("Loaded model from disk")
    return model


def process_pic(filename):
    original_pic = Image.open(filename).resize((32, 32))
    original_array = np.expand_dims(np.array(original_pic), 0)

    original_array = np.swapaxes(original_array, 2, 3)
    original_array = np.swapaxes(original_array, 1, 2)
    return original_array


def deprocess_array(array):
    array = np.swapaxes(array, 1, 2)
    array = np.swapaxes(array, 2, 3)
    # array = array.astype(np.uint8)

    return array


def generate_example(model, eps, original_array):
    target_idx = model.predict(original_array).argmax()
    target = to_categorical(target_idx, 10)
    target_variable = K.variable(target)

    loss = metrics.categorical_crossentropy(model.output, target_variable)

    gradients = K.gradients(loss, model.input)
    get_grad_values = K.function([model.input], gradients)
    grad_values = get_grad_values([original_array])[0]
    grad_signs = np.sign(grad_values)

    perturbation = grad_signs * eps
    print(perturbation[0])
    # perturbation = perturbation.astype(np.uint8)
    modified_array = original_array + perturbation
    # modified_array = modified_array.astype(np.uint8)    
    return perturbation, modified_array


def make_pred(model, array_input):
    pred = model.predict(array_input)
    preds = [(class_names[i], pred[0][i]) for i in range(10)]
    return preds


def compare_visualize(model, original, perturbation, modified):
    original_pred = make_pred(model, original)
    perturbation_pred = make_pred(model, perturbation)
    modified_pred = make_pred(model, modified)

    original = deprocess_array(original)
    # plt.imshow(original[0])
    print('Predictions for original image:\n {}'.format(original_pred))

    perturbation = deprocess_array(perturbation)
    perturbation = np.clip(255 - perturbation, 0., 255.)
    # plt.imshow(perturbation[0])
    print('Predictions for perturbation:\n {}'.format(perturbation_pred))

    modified = deprocess_array(modified)
    modified = 255 - np.clip(modified, 0., 255.)
    
    # plt.imshow(modified[0])
    print('Predictions for modified image:\n {}'.format(modified_pred))

    show_images([original[0], perturbation[0], modified[0]], 3)


model = load_model()
original = process_pic('../images/plane.jpg')
perturbation, modified = generate_example(model, 0.5, original)
compare_visualize(model, original, perturbation, modified)
