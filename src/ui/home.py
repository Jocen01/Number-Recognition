from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Line
from kivy.graphics import Color
from PIL import Image
import numpy as np


class Main(BoxLayout):
    pass

class DrawingWindow(BoxLayout):
    def __init__(self, **kwargs):
        super(DrawingWindow, self).__init__(**kwargs)

    def export_image(self):
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
        

class DrawingWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with self.canvas:
            Color(1,0,0,0.5,"rgba")

    def clear(self):
        pass

    def on_touch_down(self, touch):
        if self.collide_point(touch.x, touch.y):
            with self.canvas:
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=2)

    def on_touch_move(self, touch):
        if self.collide_point(touch.x,touch.y):
            if touch.ud:
                touch.ud['line'].points += [touch.x, touch.y]
            else:
                with self.canvas:
                    touch.ud['line'] = Line(points=(touch.x, touch.y), width=2)
        else:
            with self.canvas:
                touch.ud['line'] = Line(points=(touch.x, touch.y), width=2)

class MainApp(App):
    def build(self):
        return Main()


MainApp().run()
#TestApp().run()