"""Draw a digit with the mouse and let the network guess it.

The preprocessing matters more than it looks.  MNIST digits were not simply
resized to 28x28: each digit was scaled to fit a 20x20 box preserving aspect
ratio, then placed in a 28x28 field so that its **centre of mass** sits at the
centre.  A network trained on that will do noticeably worse on drawings that
are merely cropped and resized, which is the usual reason a demo feels less
accurate than the reported test score.
"""

from __future__ import annotations

import numpy as np


def preprocess(image):
    """PIL grayscale image (white digit on black) -> (784, 1) array in [0, 1]."""
    from PIL import Image

    bbox = image.getbbox()
    if bbox is None:
        return None
    digit = image.crop(bbox)

    # Scale the long side to 20px, preserving aspect ratio.
    w, h = digit.size
    scale = 20.0 / max(w, h)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    digit = digit.resize(new_size, Image.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    canvas.paste(digit, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))

    # Shift so the centre of mass lands on the centre of the frame.
    pixels = np.asarray(canvas, dtype=np.float64)
    total = pixels.sum()
    if total > 0:
        rows, cols = np.indices((28, 28))
        cy = (pixels * rows).sum() / total
        cx = (pixels * cols).sum() / total
        shifted = np.zeros_like(pixels)
        dy, dx = int(round(13.5 - cy)), int(round(13.5 - cx))
        ysrc = slice(max(0, -dy), 28 - max(0, dy))
        ydst = slice(max(0, dy), 28 - max(0, -dy))
        xsrc = slice(max(0, -dx), 28 - max(0, dx))
        xdst = slice(max(0, dx), 28 - max(0, -dx))
        shifted[ydst, xdst] = pixels[ysrc, xsrc]
        pixels = shifted

    return (pixels.reshape(784, 1) / 255.0)


def run_paint_demo(net, canvas_size=280, brush_radius=9):
    """Open a canvas, preprocess whatever is drawn, and predict."""
    import tkinter as tk
    from PIL import Image, ImageDraw

    root = tk.Tk()
    root.title("Draw a digit (0-9)")

    canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="black")
    canvas.pack()

    state = {"image": Image.new("L", (canvas_size, canvas_size), color=0)}
    state["draw"] = ImageDraw.Draw(state["image"])

    def paint(event):
        x, y = event.x, event.y
        box = [x - brush_radius, y - brush_radius, x + brush_radius, y + brush_radius]
        canvas.create_oval(*box, fill="white", outline="white")
        state["draw"].ellipse(box, fill=255)

    canvas.bind("<B1-Motion>", paint)

    label = tk.Label(root, text="Draw a digit, then click Predict", font=("Arial", 14))
    label.pack(pady=5)
    runners_up = tk.Label(root, text="", font=("Arial", 10), fg="#666")
    runners_up.pack()

    def predict():
        x = preprocess(state["image"])
        if x is None:
            label.config(text="Canvas is empty - draw a digit first")
            return
        scores = net.predict(x).ravel()
        scores = scores / scores.sum()  # sigmoid outputs are not normalised
        order = np.argsort(scores)[::-1]
        label.config(text=f"Prediction: {order[0]}   ({scores[order[0]] * 100:.1f}%)")
        runners_up.config(
            text="  ".join(f"{d}: {scores[d] * 100:.1f}%" for d in order[1:4])
        )

    def clear():
        canvas.delete("all")
        state["image"] = Image.new("L", (canvas_size, canvas_size), color=0)
        state["draw"] = ImageDraw.Draw(state["image"])
        label.config(text="Draw a digit, then click Predict")
        runners_up.config(text="")

    buttons = tk.Frame(root)
    buttons.pack(pady=5)
    tk.Button(buttons, text="Predict", command=predict).pack(side="left", padx=5)
    tk.Button(buttons, text="Clear", command=clear).pack(side="left", padx=5)

    root.mainloop()


if __name__ == "__main__":
    import argparse

    from network import Network

    p = argparse.ArgumentParser(description="Draw-and-predict demo")
    p.add_argument("--model", default="model.npz")
    run_paint_demo(Network.load(p.parse_args().model))
