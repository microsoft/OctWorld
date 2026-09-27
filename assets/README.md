# Example Assets

This directory separates editable source material from generated model input.

```text
assets/
├── raw/
│   ├── DL3DVHotelRoom/
│   │   ├── image.png
│   │   └── camera_turns.txt
│   └── MinecraftCanyon/
│       ├── image.png
│       └── camera_turns.txt
└── wan_standard/
    ├── instances.json
    └── scenes/
        ├── scene_00000/  # DL3DV image + original DA3 trajectory
        ├── scene_00001/  # DL3DV image + custom trajectory
        └── scene_00002/  # Minecraft image + custom trajectory
```

The two directories under `raw/` are the only editable custom-case inputs. Each
contains exactly one image and one WorldScore-style camera-turn file. The DL3DV
GT case has no raw directory because it already includes a source trajectory.

`wan_standard/` is the complete three-case dataset consumed by the single-case
and batch world-generation examples. Every scene is limited to four chunks:
321 camera poses and four prompt lines.

Rebuild the two custom scenes from `raw/` with the one-shot builder documented
in [the customization guide](../example/customize_world_gen/README.md). The
optional preview command is useful while editing `camera_turns.txt`, but is not
required by the one-shot build path.