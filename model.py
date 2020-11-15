from tensorflow.keras.applications import ResNet50
import tensorflow as tf
from tensorflow.keras import Model
class Model_Resnet:
    def pretrained_model(self,path):
        pretrained_model = ResNet50(include_top=False,input_shape=(150,150,3),pooling='max',weights=None)
        pretrained_model.load_weights(path)
        for layers in pretrained_model.layers:
            layers.trainable = False
        # If you want Transfer learning then we can set layers.trainable=False
        # If you want Finetune then we can set layes.trainable=True

        last_layer = pretrained_model.get_layer('max_pool')
        last_output = last_layer.output

        x = tf.keras.layers.Dense(3,activation='softmax')(last_output)
        model = Model(pretrained_model.input,x)
        return model
