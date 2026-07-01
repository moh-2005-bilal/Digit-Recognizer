# =============================================================================
#  INTERACTIVE HANDWRITTEN DIGIT RECOGNIZER WITH ONLINE LEARNING (v2)
#
#  What changed vs. the original version, and why:
#
#  1. MNIST-STYLE PREPROCESSING
#     MNIST digits aren't just "resized to 28x28" - they're bounding-box
#     cropped, scaled to fit inside a ~20x20 box, and centered by their
#     center of mass in the 28x28 frame. A plain resize of a big canvas
#     drawing (small digit in a big black square) looks nothing like that
#     to the network. This is the single biggest reason a homemade digit
#     recognizer "feels dumb" - so we replicate MNIST's own pipeline.
#
#  2. A SMALL CNN INSTEAD OF A DENSE NETWORK
#     Two conv+pool blocks generalize much better to messy hand-drawn
#     input than a single 128-unit dense layer, for a small compute cost.
#
#  3. SAFER ONLINE LEARNING (replay buffer + anchors + lower LR)
#     Calling train_on_batch on a single new sample can cause "catastrophic
#     forgetting" - the model overfits to that one correction and gets
#     worse at everything else. We fix this three ways:
#       - keep a replay buffer of past corrections and mix a few in
#       - mix in a handful of real MNIST samples ("anchors") every update,
#         so the model is reminded what it already knew
#       - use a much smaller learning rate for these live updates than
#         for the original training run
#
#  4. STATE IN A CLASS, NOT GLOBALS
#     Everything the app needs to remember (the drawing, brush size,
#     replay buffer, etc.) lives on one object instead of scattered
#     global variables. Same behavior, easier to read and extend.
#
#  5. VECTORIZED BRUSH + BLANK-CANVAS GUARD
#     The brush stamp uses NumPy boolean masking instead of a nested
#     Python loop, and "Predict" on an empty canvas is now a no-op
#     with a message, instead of a confident guess about nothing.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageFilter

MODEL_PATH = "digit_cnn_model.keras"   # new filename: architecture changed (CNN, not dense)
CANVAS_SIZE = 280                      # drawing area, pixels
INITIAL_BRUSH_RADIUS = 8
NUM_ANCHOR_SAMPLES = 500               # real MNIST samples kept in memory to fight forgetting
MAX_REPLAY = 300                       # how many past corrections we remember
REPLAY_SAMPLE_SIZE = 15                # how many replay samples to mix into each update
ANCHOR_SAMPLE_SIZE = 32                # how many anchor samples to mix into each update
ONLINE_LEARNING_RATE = 1e-4            # much smaller than the training-time LR


# -----------------------------------------------------------------------
# 1. MODEL: build, train (once), or load a saved one
# -----------------------------------------------------------------------
def build_model():
    """A small CNN: two conv+pool blocks, then a dense head."""
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, 3, activation="relu"),
        MaxPooling2D(),
        Conv2D(64, 3, activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def load_or_train_model():
    """Load a saved model if one exists and matches this architecture; otherwise train fresh."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train / 255.0).reshape(-1, 28, 28, 1).astype(np.float32)
    x_test = (x_test / 255.0).reshape(-1, 28, 28, 1).astype(np.float32)
    y_train_oh = to_categorical(y_train, 10)
    y_test_oh = to_categorical(y_test, 10)

    model = None
    if os.path.exists(MODEL_PATH):
        try:
            print("Loading saved model...")
            model = load_model(MODEL_PATH)
            test_loss, test_acc = model.evaluate(x_test, y_test_oh, verbose=0)
            print(f"Loaded model test accuracy: {test_acc:.2%}")
        except Exception as e:
            print(f"Couldn't load saved model ({e}); training a new one instead.")
            model = None

    if model is None:
        print("Training new model...")
        model = build_model()
        model.fit(x_train, y_train_oh, epochs=5, validation_data=(x_test, y_test_oh), verbose=1)
        test_loss, test_acc = model.evaluate(x_test, y_test_oh, verbose=0)
        print(f"\nTest accuracy: {test_acc:.2%}")
        model.save(MODEL_PATH)
        print(f"Model saved as '{MODEL_PATH}'")

    # Lower the learning rate now, permanently, for the live online-learning updates
    # that happen later via train_on_batch. This does not affect the training above.
    model.optimizer.learning_rate.assign(ONLINE_LEARNING_RATE)

    # Keep a small random slice of real MNIST test images in memory as "anchors" -
    # every online update will mix a few of these in, so the model keeps
    # remembering what real digits look like instead of drifting toward
    # whatever one sample it was just shown.
    rng = np.random.default_rng(0)
    anchor_idx = rng.choice(len(x_test), size=NUM_ANCHOR_SAMPLES, replace=False)
    anchors_x = x_test[anchor_idx]
    anchors_y = y_test_oh[anchor_idx]

    return model, anchors_x, anchors_y


# -----------------------------------------------------------------------
# 2. PREPROCESSING: turn a raw 280x280 canvas into an MNIST-style 28x28 image
# -----------------------------------------------------------------------
def preprocess_drawing(drawing):
    """
    Reproduce MNIST's own preprocessing so the network sees something
    close to what it was trained on:
      1. Crop to the bounding box of what was actually drawn.
      2. Resize so the longer side fits in a 20x20 box (aspect preserved).
      3. Slight blur, since MNIST strokes are antialiased, not hard-edged.
      4. Paste into a 28x28 frame, centered by center of mass (not just
         the geometric center - this matches how MNIST digits are framed).
    Returns a (28, 28) float32 array in [0, 1], or an all-zero array if
    the canvas is blank.
    """
    coords = np.argwhere(drawing > 20)
    if coords.size == 0:
        return np.zeros((28, 28), dtype=np.float32)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = drawing[y0:y1, x0:x1]

    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))

    small = Image.fromarray(cropped).resize((new_w, new_h), Image.Resampling.LANCZOS)
    small = small.filter(ImageFilter.GaussianBlur(radius=1))
    small_arr = np.array(small, dtype=np.float32)

    # Center of mass of the resized stroke, used to place it precisely
    ys, xs = np.nonzero(small_arr > 10)
    cy, cx = (ys.mean(), xs.mean()) if len(xs) else (new_h / 2, new_w / 2)

    canvas28 = np.zeros((28, 28), dtype=np.float32)
    top = round(14 - cy)
    left = round(14 - cx)

    # Paste with bounds-checking in case centering pushes it off the edge
    y_start, x_start = max(0, top), max(0, left)
    y_end, x_end = min(28, top + new_h), min(28, left + new_w)
    src_y, src_x = y_start - top, x_start - left
    canvas28[y_start:y_end, x_start:x_end] = small_arr[
        src_y:src_y + (y_end - y_start), src_x:src_x + (x_end - x_start)
    ]

    return canvas28 / 255.0


# -----------------------------------------------------------------------
# 3. THE APP: drawing canvas, prediction, live feedback, UI wiring
# -----------------------------------------------------------------------
class DigitRecognizerApp:
    def __init__(self, model, anchors_x, anchors_y):
        self.model = model
        self.anchors_x = anchors_x
        self.anchors_y = anchors_y

        self.drawing = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        self.brush_radius = INITIAL_BRUSH_RADIUS
        self.drawing_enabled = False

        self.pending_feedback = False
        self.feedback_image = None      # 28x28 float array awaiting a label
        self.replay_buffer = []         # list of (28x28 float array, true_label)

        self._build_figure()
        self._connect_events()

    # ---------------- UI construction ----------------
    def _build_figure(self):
        self.fig = plt.figure(figsize=(10, 6), facecolor="#f5f5f5")
        gs = GridSpec(2, 2, figure=self.fig,
                       width_ratios=[2, 1], height_ratios=[3, 1],
                       left=0.05, right=0.95, top=0.93, bottom=0.12,
                       wspace=0.15, hspace=0.25)

        # Drawing canvas
        self.ax_draw = self.fig.add_subplot(gs[:, 0])
        self.ax_draw.set_facecolor("#2d2d2d")
        self.ax_draw.set_xticks([]); self.ax_draw.set_yticks([])
        for spine in self.ax_draw.spines.values():
            spine.set_edgecolor("#777777")
            spine.set_linewidth(2)
        self.img_display = self.ax_draw.imshow(
            self.drawing, cmap="gray", vmin=0, vmax=255, origin="upper"
        )
        self._set_draw_title()

        # 28x28 preview
        self.ax_preview = self.fig.add_subplot(gs[0, 1])
        self.ax_preview.set_title("Model's view (28\u00d728)", fontsize=10,
                                   fontweight="bold", color="#555555")
        self.ax_preview.set_xticks([]); self.ax_preview.set_yticks([])
        self.preview_img = self.ax_preview.imshow(np.zeros((28, 28)), cmap="gray", vmin=0, vmax=1)

        # Confidence bar chart
        self.ax_bars = self.fig.add_subplot(gs[1, 1])
        self.ax_bars.set_title("Confidence per digit", fontsize=10, fontweight="bold", color="#555555")
        self.ax_bars.set_xlim(0, 1)
        self.ax_bars.set_ylim(-0.5, 9.5)
        self.ax_bars.set_yticks(range(10))
        self.ax_bars.set_yticklabels(range(10))
        self.ax_bars.invert_yaxis()
        self.ax_bars.set_xlabel("Probability")
        self.bars = self.ax_bars.barh(range(10), np.zeros(10), color="#cccccc")

        # Buttons
        btn_predict_ax = plt.axes([0.28, 0.03, 0.12, 0.06])
        btn_clear_ax = plt.axes([0.42, 0.03, 0.12, 0.06])
        self.btn_predict = Button(btn_predict_ax, "Predict", color="#4CAF50", hovercolor="#66BB6A")
        self.btn_clear = Button(btn_clear_ax, "Clear", color="#f44336", hovercolor="#EF5350")
        for btn in (self.btn_predict, self.btn_clear):
            btn.label.set_fontsize(11)
            btn.label.set_fontweight("bold")
            btn.label.set_color("white")
        self.btn_predict.on_clicked(lambda event: self.process_and_predict())
        self.btn_clear.on_clicked(lambda event: self.clear_canvas())

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    # ---------------- drawing ----------------
    def on_press(self, event):
        if event.inaxes != self.ax_draw:
            return
        self.drawing_enabled = True
        self._paint(event)

    def on_motion(self, event):
        if not self.drawing_enabled or event.inaxes != self.ax_draw:
            return
        self._paint(event)

    def on_release(self, event):
        self.drawing_enabled = False

    def _paint(self, event):
        """Stamp a filled circle onto self.drawing, vectorized with NumPy."""
        if event.xdata is None or event.ydata is None:
            return
        col, row, r = int(event.xdata), int(event.ydata), self.brush_radius

        y0, y1 = max(0, row - r), min(CANVAS_SIZE, row + r + 1)
        x0, x1 = max(0, col - r), min(CANVAS_SIZE, col + r + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = (yy - row) ** 2 + (xx - col) ** 2 <= r * r

        region = self.drawing[y0:y1, x0:x1]
        region[mask] = 255
        self.drawing[y0:y1, x0:x1] = region

        self.img_display.set_data(self.drawing)
        self.fig.canvas.draw_idle()

    # ---------------- prediction ----------------
    def process_and_predict(self):
        if self.drawing.max() == 0:
            self.ax_draw.set_title("Canvas is empty - draw a digit first!",
                                    fontsize=12, fontweight="bold", color="#c62828")
            self.fig.canvas.draw_idle()
            return

        img28 = preprocess_drawing(self.drawing)
        self.feedback_image = img28.copy()

        self.preview_img.set_data(img28)

        probs = self.model.predict(img28.reshape(1, 28, 28, 1), verbose=0)[0]
        pred_digit = int(np.argmax(probs))
        conf = float(probs[pred_digit])

        self._update_bars(probs, pred_digit)
        color = self._confidence_color(conf)
        self.ax_draw.set_title(
            f"Predicted: {pred_digit} ({conf:.1%})  |  Brush: {self.brush_radius} px\n"
            f"[ Press Y if correct, or digit 0-9 for true label ]",
            fontsize=12, fontweight="bold", color=color,
        )
        self.pending_feedback = True
        self.fig.canvas.draw_idle()
        print(f"Predicted: {pred_digit} (confidence: {conf:.2%}) - waiting for feedback...")

    def apply_feedback(self, true_label):
        """One live training step on the correction, cushioned by anchors + replay."""
        if self.feedback_image is None:
            return

        # Remember this correction for future updates
        self.replay_buffer.append((self.feedback_image.copy(), true_label))
        if len(self.replay_buffer) > MAX_REPLAY:
            self.replay_buffer.pop(0)

        batch_x = [self.feedback_image]
        batch_y = [to_categorical(true_label, num_classes=10)]

        # Mix in a handful of past corrections
        if len(self.replay_buffer) > 1:
            k = min(REPLAY_SAMPLE_SIZE, len(self.replay_buffer) - 1)
            idx = np.random.choice(len(self.replay_buffer) - 1, size=k, replace=False)
            for i in idx:
                img, label = self.replay_buffer[i]
                batch_x.append(img)
                batch_y.append(to_categorical(label, num_classes=10))

        # Mix in real MNIST anchors so the model doesn't drift
        anchor_idx = np.random.choice(len(self.anchors_x), size=ANCHOR_SAMPLE_SIZE, replace=False)
        batch_x = np.array(batch_x).reshape(-1, 28, 28, 1)
        batch_x = np.concatenate([batch_x, self.anchors_x[anchor_idx]], axis=0)
        batch_y = np.concatenate([np.array(batch_y), self.anchors_y[anchor_idx]], axis=0)

        self.model.train_on_batch(batch_x, batch_y)
        print(f"Model updated with label {true_label} "
              f"(batch: 1 new + {len(batch_x) - ANCHOR_SAMPLE_SIZE - 1} replay + {ANCHOR_SAMPLE_SIZE} anchors).")

        # Re-predict the same image to show the effect of the update
        probs = self.model.predict(self.feedback_image.reshape(1, 28, 28, 1), verbose=0)[0]
        new_pred = int(np.argmax(probs))
        new_conf = float(probs[new_pred])

        self._update_bars(probs, new_pred, prefix="Updated")
        color = self._confidence_color(new_conf)
        self.ax_draw.set_title(f"Updated prediction: {new_pred} ({new_conf:.1%})  |  Brush: {self.brush_radius} px",
                                fontsize=12, fontweight="bold", color=color)

        self.pending_feedback = False
        self.feedback_image = None
        self.fig.canvas.draw_idle()

    # ---------------- clearing ----------------
    def clear_canvas(self):
        self.drawing[:] = 0
        self.img_display.set_data(self.drawing)
        self._set_draw_title()

        self.preview_img.set_data(np.zeros((28, 28)))
        self.ax_preview.set_title("Model's view (28\u00d728)", fontsize=10,
                                   fontweight="bold", color="#555555")

        for bar in self.bars:
            bar.set_width(0)
            bar.set_color("#cccccc")
        self.ax_bars.set_title("Confidence per digit", fontsize=10, fontweight="bold", color="#555555")

        self.pending_feedback = False
        self.feedback_image = None

        self.fig.canvas.draw_idle()
        print("Canvas cleared.")

    # ---------------- keyboard ----------------
    def on_key(self, event):
        if self.pending_feedback:
            if event.key == "y":
                self.apply_feedback_from_last_prediction()
            elif event.key in [str(d) for d in range(10)]:
                self.apply_feedback(int(event.key))
            elif event.key == "c":
                self.clear_canvas()
            return  # ignore everything else while awaiting feedback

        if event.key == "p":
            self.process_and_predict()
        elif event.key == "c":
            self.clear_canvas()
        elif event.key in ["+", "="]:
            self.brush_radius = min(20, self.brush_radius + 2)
            self._set_draw_title()
            self.fig.canvas.draw_idle()
        elif event.key == "-":
            self.brush_radius = max(2, self.brush_radius - 2)
            self._set_draw_title()
            self.fig.canvas.draw_idle()

    def apply_feedback_from_last_prediction(self):
        probs = self.model.predict(self.feedback_image.reshape(1, 28, 28, 1), verbose=0)[0]
        self.apply_feedback(int(np.argmax(probs)))

    # ---------------- small shared helpers ----------------
    def _set_draw_title(self):
        self.ax_draw.set_title(f"Draw a digit (0-9)  |  Brush: {self.brush_radius} px",
                                fontsize=12, fontweight="bold", color="#333333")

    def _update_bars(self, probs, highlight_digit, prefix="Predicted"):
        for bar, val in zip(self.bars, probs):
            bar.set_width(val)
            bar.set_color("#4CAF50" if np.argmax(probs) == highlight_digit and val == probs[highlight_digit] else "#cccccc")
        self.ax_bars.set_title(f"{prefix}: {highlight_digit} ({probs[highlight_digit]:.1%})",
                                fontsize=10, fontweight="bold", color="#1a73e8")

    @staticmethod
    def _confidence_color(conf):
        if conf >= 0.9:
            return "#2e7d32"   # green
        if conf >= 0.7:
            return "#f57c00"   # orange
        return "#c62828"       # red

    def run(self):
        plt.show()


# -----------------------------------------------------------------------
# 4. ENTRY POINT
# -----------------------------------------------------------------------
if __name__ == "__main__":
    model, anchors_x, anchors_y = load_or_train_model()
    app = DigitRecognizerApp(model, anchors_x, anchors_y)
    app.run()