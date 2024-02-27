from keras import applications
from keras.layers import Conv2D,Conv2DTranspose,Input,Cropping2D,add,Dropout,Reshape,Activation
from keras.models import Model,Sequential
def FCN32(n_Classes,input_height,input_width):
    assert input_height % 32==0
    assert input_width % 32==0 #断言是否能被32整除，如果不能则会报错
    img_input=Input(shape=(input_height,input_width,3))

    model=applications.vgg16.VGG16(
        include_top=False,
        weights="imagenet",input_tensor=img_input,
        pooling=None,
        classes=1000
    )
    assert isinstance(model,Model)
    o=Conv2D(filters=4096,kernel_size=(7,7),padding="same",activation="relu",name="fc6")(model.output)
    o=Dropout(0.5)(o)
    o=Conv2D(filters=4096,kernel_size=(1,1),padding="same",activation="relu",name="fc7")(o)
    o=Dropout(0.5)(o)
    o=Conv2D(filters=n_Classes,kernel_size=(1,1),padding="same",activation="relu",
             kernel_initializer="he_normal",name="score_fr")(o)
   # s k 相等的话就是放大了多少倍
    o=Conv2DTranspose(n_Classes,kernel_size=32,strides=32,padding="valid",activation=None,name="upsample")(o)
    o=Reshape((-1,n_Classes))(o)
    o=Activation("softmax")(o)
    mymodel=Model(inputs=img_input,outputs=o)
    return mymodel

if __name__ =="__main__":
    m=FCN32(15,320,320)
    from keras.utils import plot_model
    plot_model(m,show_shapes=True,to_file="model_fcn32.png")
    print(len(m.layers))