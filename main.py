
from data import DataGenerator
from model import Model_Resnet
from callbacks import myCallBack


def main():
    #Getting the data
    data = DataGenerator()
    train_gen = data.train_data('<path to the train directory')
    valid_gen = data.test_data('<path to the test directory')

    #Getting the model
    pre_model = Model_Resnet()
    model = pre_model.pretrained_model('<path to the preloaded model')

    #Getting the callbacks
    callbacks_custom = myCallBack()
    callbacks = callbacks_custom.custom_callbacks()

    #Model training
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=["accuracy"])
    model.fit_generator(train_gen,validation_data=valid_gen,epochs=2,verbose=2,callbacks=callbacks)




if __name__ == '__main__':
    main()


