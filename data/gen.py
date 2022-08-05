from PIL import Image, ImageDraw
import random as rd

rd.seed(0)


def draw_rectangle(draw, color):
    x = rd.randint(11, 21)
    y = rd.randint(11, 21)
    draw.rectangle((x - 5, y - 5, x + 5, y + 5), fill=color)


def generate_scene(path, idx):
    img = Image.new('RGB', (32, 32))
    draw = ImageDraw.Draw(img)

    draw_rectangle(draw, "blue")

    img.save(f"simple/images/{path}/{str(idx) + '.png'}")


for i in range(20000):
    generate_scene("train", i)

for i in range(5000):
    generate_scene("test", i)

for i in range(5000):
    generate_scene("val", i)