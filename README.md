# Camuse

Camera-based body tracking prototype. For now it opens your webcam, tracks hand
positions, and estimates mouth openness. Later this can become a music
playground driven by body movement.

## Requirements

- Windows camera permissions enabled for desktop apps.
- [uv](https://docs.astral.sh/uv/) installed.
- A webcam.

The project pins Python to `>=3.11,<3.13` because MediaPipe wheels are most
reliable there.

## Run

```powershell
uv run camuse
```

Useful options:

```powershell
uv run camuse --camera 1
uv run camuse --width 960 --height 540
uv run camuse --print-json
```

Controls:

- Press `q` or `Esc` to quit.

## What The App Tracks

- `hands`: each detected hand includes `wrist`, `index_tip`, and `palm_center`
  coordinates normalized from `0.0` to `1.0`.
- `mouth.openness`: distance between the upper and lower lip, normalized by
  mouth width. Bigger values mean a more open mouth.

When `--print-json` is enabled, each frame prints a compact JSON line with this
state.
