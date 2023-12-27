import cv2
from kivy.app import App
import numpy as np
from keras.models import load_model
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView

model = load_model('Trained_model.h5')

Emotion_Labels={0:'angry',1:'Disgust',2:'Fear',3:'Happy',4:'Netrual',5:'sad',6:'Suprise'}


class Home(Screen):
    pass


class VideoRecgntion(Screen):

    def build(self):
        self.face_cascaed = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
        self.cap = cv2.VideoCapture(0)
        Clock.schedule_interval(self.Video_load, 1.0/30)


    def Video_load(self,*args):
        if self.ids.btn3.text=="Stop Detction":
            ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascaed.detectMultiScale(gray, 1.045, 4)
            for x, y, w, h in faces:
                self.face = cv2.rectangle(gray, (x - 25, y + 25), (x + w, y + h), (255, 255, 255), 5)
                roi_gray = gray[y:y + h, x:x + w]
                resize = cv2.resize(roi_gray, (48, 48))
                normlise = resize / 255.0
                reshape = np.reshape(normlise, (1, 48, 48, 1))
                result = model.predict(reshape)
                label = np.argmax(result, axis=1)[0]
                cv2.putText(gray, Emotion_Labels[label], (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 5)

                buffer = cv2.flip(gray, 0).tostring()
                texture = Texture.create(size=(gray.shape[1], frame.shape[0]), colorfmt='luminance')
                texture.blit_buffer(buffer, colorfmt='luminance', bufferfmt='ubyte')
                self.ids.webcam.texture = texture
        else:
            Clock.unschedule(self)
            self.ids.webcam.texture=None




class ImageRecgntion(Screen):
    def build (self):
        p=Pop_window()
        p.open()


    def clear_input(self):
        self.ids.img.source = ""


    def extract_emotion(self,sent_pic):

        image_chosen = cv2.imread(sent_pic)
        gray = cv2.cvtColor(image_chosen, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (48, 48))
        normlise = resize / 255.0
        reshape = np.reshape(normlise, (1, 48, 48, 1))
        result = model.predict(reshape)
        StateOfEmotion = np.argmax(result, axis=1)[0]
        self.update_label(Emotion_Labels[StateOfEmotion])

    def update_label(self, result):
            ImageRecgntion_screen = App.get_running_app().root.get_screen('ImageRecgntion')
            ImageRecgntion_screen.ids.result.text = result



class Pop_window(Popup):
    def __init__(self,**kwargs):
        super(Pop_window,self).__init__(**kwargs)
        self.image=""
        self.file_choser=FileChooserIconView()
        self.file_choser.bind(on_submit=self.grap_image)
        self.content=self.file_choser


    def grap_image(self,instanse,sslect,value):
        selected = instanse.selection
        if selected:

            self.image = selected[0]
            ImageRecgntion_screen = App.get_running_app().root.get_screen('ImageRecgntion')
            ImageRecgntion_screen.ids.img.source = self.image
            temp=ImageRecgntion()
            temp.extract_emotion(self.image)

            self.dismiss()



class Manger(ScreenManager):
    pass



kv = Builder.load_string("""
Manger:
    Home:
    VideoRecgntion:
    ImageRecgntion:


<Home>:
    name:"Home"
    GridLayout:
        cols:1
        Button:
            text:"Live detction"
            on_release:
                app.root.current="VideoRecgntion"
        Button:
            text:"Upload Image"
            on_release:
                app.root.current="ImageRecgntion"


<VideoRecgntion>:
    name:"VideoRecgntion"
    GridLayout:
        cols:1
        row:2
        Image:
            id:webcam
            source:""
        GridLayout:
            cols:1
            ToggleButton:
                id:btn3
                text:"Start Detction" if btn3.state == "normal" else "Stop Detction"
                on_release:
                    root.build()

            Button:
                text:"Home"

                on_release:
                    app.root.current="Home"


<ImageRecgntion>:
    name:"ImageRecgntion"
    orientation:"vertical"
    GridLayout:
        cols:1
        Image:
            id:img
            source:""

        Label:
            id:result
            text:""
            font_size: 50
            halign: 'center'
            valign: 'middle'
        GridLayout:
            cols:1
            Button:
                text:"Home"
                on_release:
                    app.root.current="Home"
            Button:
                text:"Upload Image"
                on_release:
                    root.clear_input()
                    root.build()

""")

class MyMain(App):
    def build(self):
        return kv


if __name__ == '__main__':
    MyMain().run()

