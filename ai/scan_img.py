import PIL
from PIL import Image
from PIL import UnidentifiedImageError
import glob

img_all = glob.glob("G:/GitHub/wsem-ai/ai/*/*/*.jpg")

for img in img_all:
    try:
        img = PIL.Image.open(img)
    except PIL.UnidentifiedImageError:
        print(img)  # print full path if error beim öffnen