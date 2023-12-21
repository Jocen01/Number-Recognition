from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty
from kivy.graphics.texture import Texture
from kivy.graphics import Line
from kivy.graphics import Color
from PIL import Image
import numpy as np
from scipy.ndimage import zoom
from keras.datasets import mnist
from src.Model.NerualNetwork import NeuralNetwork



class Main(BoxLayout):
    guess_str = StringProperty("-")
    def __init__(self, **kwargs):
        super(Main, self).__init__(**kwargs)
        self.nn: NeuralNetwork = NeuralNetwork.load(f"./saved_networks/pre_loaded.pkl")
        (self.train_x, self.train_y), (self.test_x, self.test_y) = mnist.load_data()
        self.textures_gen = self.generate_img(self.test_x, self.test_y)
        self.show_next_test()

    def train(self, epochs):
        try:
            epochs = int(epochs)
        except ValueError:
            self.guess_str = "Try input a number"
            return
        self.nn.train(self.train_x, self.train_y,int(input("How many epochs: ")))

    def show_score_test(self):
        res = self.nn.test(self.test_x[:10000],self.test_y[:10000])
        self.guess_str = f"final score is {res[0]}  {round(res[1],5)}"

    def load(self, name):
        self.nn: NeuralNetwork = NeuralNetwork.load(f"./saved_networks/{name}.pkl")

    def save(self, name):
        NeuralNetwork.save(f"./saved_networks/{name}.pkl", self.nn)

    def guess(self, img: np.array):
        self.guess_str = str(self.nn.guess(img/img.max()))

    def show_next_test(self):
        texture, y = next(self.textures_gen)
        self.guess_str = str(y)
        self.ids["img"].texture=texture


    def generate_img(self, data_x: np.array, data_y: np.array):
        for X,y in zip(data_x,data_y):
            texture = Texture.create(size=(280, 280), colorfmt='luminance')
            texture.flip_vertical()
            texture.blit_buffer(zoom(X,10).tobytes(), colorfmt='luminance', bufferfmt='ubyte')
            yield texture, y



class DrawingWindow(BoxLayout):
    def __init__(self, **kwargs):
        super(DrawingWindow, self).__init__(**kwargs)

    def _export_image(self):
        # Render the image widget to get the texture
        self.ids.drawCanvas.export_to_png("exported_image.png")

        # Open the saved image and convert it to a NumPy array
        pil_image = Image.open("exported_image.png")
        image_array = np.array(pil_image)

        # Extract the RGB values from the array
        rgb_values = image_array[:, :, :3]
        black_white = rgb_values.mean(axis=2)
        
        return black_white
    
    def get_image_array(self, shape: (int,int) = None) -> np.array:
        img = self._export_image()
        org_size = img.shape
        z = zoom(self.trim(img,delta=2), shape[0]/org_size[0],prefilter=False)
        return z
    
    def trim(self, img: np.array, delta: int = 1) -> np.array:
        return img[delta:-delta, delta:-delta]

        

class DrawingWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.line_width = 12

        with self.canvas:
            Color(1,0,0,0.5,"rgba")

    def clear(self):
        instructions_to_remove = [instr for instr in self.canvas.children[1:] if isinstance(instr, Line)]
        for instr in instructions_to_remove:
            self.canvas.remove(instr)
        self.outline()

    def outline(self):
        with self.canvas:
            Line(
                width=2,
                points=(
                    self.pos[0],                self.pos[1],
                    self.pos[0]+self.size[0],   self.pos[1],
                    self.pos[0]+self.size[0],   self.pos[1]+self.size[1],
                    self.pos[0],                self.pos[1]+self.size[1],
                    self.pos[0],                self.pos[1])
            )

    def on_touch_down(self, touch):
        if self.collide_point(touch.x, touch.y):
            with self.canvas:
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=self.line_width)

    def on_touch_move(self, touch):
        if self.collide_point(touch.x,touch.y):
            if touch.ud:
                touch.ud['line'].points += [touch.x, touch.y]
            else:
                with self.canvas:
                    touch.ud['line'] = Line(points=(touch.x, touch.y), width=self.line_width)
        else:
            with self.canvas:
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=self.line_width)

class MainApp(App):
    def build(self):
        return Main()
