# Third-party attribution

`serve_three_arm.py` uses the length-prefixed TCP/pickle and model-serving pattern adapted from [Xiaomi-Robotics-1 deploy/server.py](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/0dd7aef8dc87296246aae812a1f59ccb708e5546/deploy/server.py).

Copyright (C) 2026 Xiaomi Corporation. Licensed under Apache License 2.0; full text is preserved in `LICENSE_XIAOMI.md`. Changes add visual latent steering, native/steering selection, loopback-only transport, bounded requests, explicit CLI and GPU selection.

The actor, critic/SAC experiment logic, and synthetic tests are project-written, not a copy or official reproduction of DSRL. The runtime adapter transforms an installed, reviewed forward function but does not distribute that function or any model weights. RoboCasa and robosuite are imported dependencies, not vendored sources. Their licenses and model-weight terms remain separate. This notice does not relicense the whole project.
