from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Line
from kivy.graphics import Color
from PIL import Image
import numpy as np
from scipy.ndimage import zoom

import matplotlib.pyplot as plt

from kivy.logger import Logger, LOG_LEVELS
import sys
sys.path.append('.')
from src.Model.utilities import printImage
from src.Model.NerualNetwork import NeuralNetwork




Logger.setLevel(LOG_LEVELS["warning"])



class Main(BoxLayout):
    def __init__(self, **kwargs):
        super(Main, self).__init__(**kwargs)
        file = input("Selcet a network to load: ")
        self.nn: NeuralNetwork = NeuralNetwork.load(f"./saved_networks/{file}.pkl")
        self.guess_str = "-"

    def train(self):
        pass

    def load(self):
        pass

    def guess(self, img: np.array):
        self.guess_str = self.nn.guess(img/img.max())
        print(self.guess_str)

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
        print("Image Matrix (RGB values):")
        print(rgb_values.shape)
        return black_white
    
    def get_image_array(self, shape: (int,int) = None) -> np.array:
        img = self._export_image()
        org_size = img.shape
        print("order",shape[0]/org_size[0])
        z = zoom(self.trim(img,delta=2), shape[0]/org_size[0],prefilter=False)
        print(type(z),z.shape)
        #fig = plt.figure()
        #ax1 = fig.add_subplot(121)  # left side
        #ax2 = fig.add_subplot(122)  # right side
        #ascent = img
        #result = z
        #ax1.imshow(ascent, vmin=0, vmax=255)
        #ax2.imshow(result, vmin=0, vmax=255)
        #plt.show()
        printImage(z)
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
