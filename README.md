# Interactive Handwritten Digit Recognizer ( 4TH Semester Intro to AI Project )

A desktop app that lets you draw digits by hand and watch a neural network
guess them in real time — and you can correct it on the spot, and it
learns from the correction immediately.

Built with TensorFlow/Keras and a Matplotlib GUI. Single file, no server,
no internet required after the first run.

## Features

- Draw digits (0–9) on a canvas with an adjustable brush
- Live 28×28 preview of exactly what the model sees
- Confidence bar chart across all ten digits
- One-click correction: if the model is wrong, tell it the right answer
  and it retrains on that example right away
- Model is cached to disk after the first run, so startup is instant
  after that

## How prediction works

Your drawing isn't just shrunk down to 28×28 — it goes through the same
kind of preprocessing MNIST digits themselves went through:

1. Crop to the bounding box of what you actually drew
2. Scale it to fit in a 20×20 box, keeping proportions
3. Soften the edges slightly (MNIST digits are antialiased, not hard-edged)
4. Center it in the 28×28 frame by center of mass — not just geometric
   center

This matters a lot for accuracy: without it, a small digit drawn in a
corner of the canvas looks nothing like what the model trained on.

## How the live learning works

When you correct a prediction, the app doesn't just train on that one
image. Doing that alone tends to make the model *worse* overall — it
overfits to the one correction and forgets things it already knew
("catastrophic forgetting"). Instead, each correction is combined into a
small training batch with:

- a few of your past corrections (a replay buffer)
- a handful of real MNIST images the app keeps in memory as "anchors"

...and trained with a much lower learning rate than the initial training
run. This keeps the model stable while still learning from you.

## Requirements

- Python 3.9–3.11 recommended (TensorFlow compatibility)
- pip

## Installation

```bash
pip install numpy matplotlib tensorflow pillow
```

Apple Silicon (M1/M2/M3) Macs, use this instead:

```bash
pip install numpy matplotlib tensorflow-macos pillow
```

Using a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install numpy matplotlib tensorflow pillow
```

## Running it

```bash
python3 digit_recognizer.py
```

**First run:** downloads MNIST (~11MB, one time), trains a small CNN for
5 epochs, then saves it as `digit_cnn_model.keras` in the same folder.
This takes a minute or two depending on your machine.

**Every run after that:** loads the saved model instantly, no retraining.

To force a full retrain from scratch, just delete `digit_cnn_model.keras`
and run the script again.

## Controls

| Action | How |
|---|---|
| Draw | Click and drag on the dark canvas |
| Predict | Click **Predict**, or press `P` |
| Clear canvas | Click **Clear**, or press `C` |
| Increase brush size | Press `+` or `=` |
| Decrease brush size | Press `-` |
| Confirm a correct prediction | Press `Y` (after predicting) |
| Correct a wrong prediction | Press the correct digit key `0`–`9` (after predicting) |

After you hit Predict, the app waits for your feedback (`Y` or a digit)
before accepting normal keyboard shortcuts again — this is intentional,
so a stray keypress doesn't skip past a correction you meant to give.

## Troubleshooting

**"No module named tensorflow" / pip install fails**
Make sure you're using a supported Python version (3.9–3.11). TensorFlow
does not yet support every new Python release on every platform.

**Matplotlib window doesn't open / backend error**
This usually happens in headless environments (WSL without an X server,
remote SSH sessions, some Docker setups). On Debian/Ubuntu, try:

```bash
sudo apt-get install python3-tk
```

**Predictions feel inaccurate even with preprocessing**
Try drawing digits a bit larger and more centered — very thin or tiny
strokes lose detail once cropped and scaled down. Increasing brush size
(`+`) often helps.

**The model seems to be forgetting things after many corrections**
This is mitigated by the anchor/replay system but not eliminated. If it
happens, delete `digit_cnn_model.keras` to start over from a freshly
trained model.

## File structure

```
digit_recognizer.py       # the whole app
digit_cnn_model.keras     # saved model (created after first run)
```

## Notes

- All learning from your corrections is lost if you delete
  `digit_cnn_model.keras` — the model file *is* the memory.
- This is a learning/demo project, not a production OCR tool. Accuracy on
  hand-drawn input will vary based on your drawing style and screen/mouse
  precision.