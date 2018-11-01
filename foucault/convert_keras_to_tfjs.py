import sys
import json

import tensorflowjs as tfjs
import keras

model_json_fpath = sys.argv[1]
weights_fpath = sys.argv[2]
target_dir = sys.argv[3]

if __name__ == '__main__':
    with open(model_json_fpath, 'r') as modelfile:
        model_json = json.dumps(json.load(modelfile))

    model = keras.models.model_from_json(model_json)

    model.load_weights(weights_fpath)

    tfjs.converters.save_keras_model(model, target_dir)
