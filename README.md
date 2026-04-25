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

## Control Serum 2 Parameters

Camuse sends MIDI CC values that Serum 2 can learn on its knobs and sliders.
This keeps Serum running inside your DAW while the camera tracker acts like a
MIDI controller.

On Windows, create a virtual MIDI port with a tool such as loopMIDI, enable
that port as a MIDI input in your DAW, and route it to the track containing
Serum 2. Then list the output ports Camuse can see:

```powershell
uv run camuse --list-midi-ports
```

Start tracking with MIDI enabled:

```powershell
uv run camuse --midi-port "Camuse"
```

In Serum 2, right-click a knob or slider, choose MIDI Learn, then move the
camera control you want to assign. Serum 2 assigns the incoming CC to that
parameter. See Xfer's Serum 2 manual section on
[using knobs and sliders](https://xferrecords.com/web-manual/serum-2/using-knobs-and-sliders)
for the MIDI Learn behavior.

Default MIDI mappings:

| Camera input | MIDI CC | Notes |
| --- | ---: | --- |
| Palm X | 20 | Left to right maps 0 to 127 |
| Palm Y | 21 | Raising your palm increases the value |
| Index fingertip X | 22 | Left to right maps 0 to 127 |
| Index fingertip Y | 23 | Raising your fingertip increases the value |
| Mouth openness | 24 | Closed to open maps 0 to 127 |

Useful MIDI options:

```powershell
uv run camuse --midi-port "Camuse" --control-hand Right
uv run camuse --midi-port "Camuse" --cc-palm-x 74 --cc-mouth 1
uv run camuse --midi-port "Camuse" --cc-index-x -1 --cc-index-y -1
uv run camuse --midi-port "Camuse" --midi-smoothing 0.2 --midi-deadband 2
```

Use `-1` for any CC option to disable that camera input.

Controls:

- Use the tracker window's `Send Palm X`, `Send Palm Y`, `Send Index X`,
  `Send Index Y`, and `Send Mouth` sliders as on/off switches for MIDI output.
  Inputs configured with CC `-1` stay unmapped.
- Press number keys `1` through `5` to toggle those same MIDI inputs from the
  camera window.
- Press `q` or `Esc` to quit.

## What The App Tracks

- `hands`: each detected hand includes `wrist`, `index_tip`, and `palm_center`
  coordinates normalized from `0.0` to `1.0`.
- `mouth.openness`: distance between the upper and lower lip, normalized by
  mouth width. Bigger values mean a more open mouth.

When `--print-json` is enabled, each frame prints a compact JSON line with this
state.
