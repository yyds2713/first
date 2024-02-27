from keras import applications
from keras.layers import Conv2D,Conv2DTranspose,Input,Cropping2D,add,Dropout,Reshape,Activation
from keras.models import Model,Sequential
def FCN16_helper(n_Classes,input_height,input_width):
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
    o=Conv2DTranspose(filters=n_Classes,kernel_size=(2,2),strides=(2,2),padding="valid",activation=None,
                      name="score2")(o)    # s k 相等的话就是放大了多少倍
    fcn16=Model(inputs=img_input,outputs=o)
    return fcn16



def FCN16(n_classes,input_height,input_width):
    fcn16=FCN16_helper(n_classes,input_height,input_width)
    skip_con1=Conv2D(n_classes,kernel_size=(1,1),padding="same",activation=None,kernel_initializer="he_normal",
                     name="score_pool4")(fcn16.get_layer("block4_pool").output)
    Summed=add(inputs=[skip_con1,fcn16.output])
    Up=Conv2DTranspose(n_classes,kernel_size=16,strides=16,padding="valid",activation=None,name="upsample")(Summed)
    Up=Reshape((-1,n_classes))(Up)
    Up=Activation("softmax")(Up)
    mymodel=Model(inputs=fcn16.input,outputs=Up)
    return mymodel
if __name__ =="__main__":
    m=FCN16(15,320,320)
    from keras.utils import plot_model
    plot_model(m,show_shapes=True,to_file="model_fcn16.png")
    print(len(m.layers))