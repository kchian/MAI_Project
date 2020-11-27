
try:
    from malmo import MalmoPython
except:
    import MalmoPython
from past.utils import old_div
from tkinter import *

from PIL import Image
from PIL import ImageTk
video_width = 800
video_height = 500
WIDTH = video_width
HEIGHT = video_height# + video_width

root = Tk()
root.wm_title("Depth and ColourMap Example")
root_frame = Frame(root)
canvas = Canvas(root_frame, borderwidth=0, highlightthickness=0, width=WIDTH, height=HEIGHT, bg="black")
canvas.config( width=WIDTH, height=HEIGHT )
canvas.pack(padx=5, pady=5)
root_frame.pack()

# https://github.com/microsoft/malmo/blob/c3d129721c5a2f7c0eac274836f113f4c7ae4205/Malmo/samples/Python_examples/radar_test.py
class draw_helper(object):
    def __init__(self):
        self._canvas = canvas
        self.reset()
        self._line_fade = 9
        self._blip_fade = 100

    def reset(self):
        self._canvas.delete("all")
        self._dots = []
        self._segments = []
        self._panorama_image = Image.new('RGB', (800, 500))
        self._panorama_photo = None
        self._image_handle = None
        self._current_frame = 0
        self._last_angle = 0


    def showFrame(self, frame):
        cmap = Image.frombytes('RGB', (video_width, video_height), bytes(frame.pixels))
        cmap.load()
        self._panorama_image.paste(cmap, (0, 0, video_width, video_height))
        self._panorama_photo = ImageTk.PhotoImage(self._panorama_image)
        # And update/create the canvas image:
        if self._image_handle is None:
            self._image_handle = canvas.create_image(old_div(WIDTH, 2), HEIGHT - (old_div(video_height, 2)), image=self._panorama_photo)
        else:
            canvas.itemconfig(self._image_handle, image=self._panorama_photo)
        root.update()