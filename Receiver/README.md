# GlassesData receiver

Receives the Meta glasses video (HEVC) and sensor data that the GlassesData iPhone app forwards over the GDL1 protocol, and on this computer (macOS; Ubuntu 24.04 see below):

- records the original HEVC into `video.mp4` (no re-encoding), plus CSVs of every sensor row;
- decodes the video and tracks hand skeletons (MediaPipe, 21 joints per hand, 2D and 3D);
- serves a local web page: the live video with the skeleton, the link and sensor status, a Record button and the list of recordings;
- feeds the video to Zoom, Meet and other apps as a webcam, through the OBS Virtual Camera (macOS) or v4l2loopback (Linux): see Virtual camera.

The receiver is a command-line process; it never opens a window. Open the page it prints (`WEB_URL=...`) in any browser.

## Setup

Python 3.11. Create the virtual environment **outside** any cloud-synced folder (OneDrive, iCloud Drive, Dropbox): syncing thousands of small files breaks or slows the environment.

```sh
python3.11 -m venv ~/.venvs/glassesdata-receiver
~/.venvs/glassesdata-receiver/bin/pip install -r requirements.txt        # requirements-dev.txt adds pytest
```

Run everything from this `Receiver/` folder with that interpreter (below, `python` means `~/.venvs/glassesdata-receiver/bin/python`). The first run downloads the hand model (see Folders).

### Folders

| | macOS | Linux |
|---|---|---|
| recordings (`--out`) | `~/Movies/GlassesData` | `~/Videos/GlassesData` |
| Wi-Fi identity (`--state-dir`) | `~/Library/Application Support/GlassesData` | `$XDG_STATE_HOME/glassesdata` (default `~/.local/state/glassesdata`) |
| hand model cache | `~/Library/Caches/GlassesData` | `$XDG_CACHE_HOME/glassesdata` (default `~/.cache/glassesdata`) |

## The web page

```sh
python -m glassesdata_receiver --usb            # prints WEB_URL=http://127.0.0.1:8765/
python -m glassesdata_receiver --usb --open     # also opens it in the default browser
```

A dark monitoring surface (it follows the browser's light or dark setting), made to stay open on a second screen. Left: the video at its own aspect (9:16 for the glasses), as tall as the window allows, with the hand skeleton drawn by the browser (crisp at any size; cyan L for the left hand, pink R for the right), and under it the transport bar: the round Record button, the timer, frames and folder while recording, the hand-tracking toggle, the virtual camera toggle and, with `--wifi`, Pair. When no iPhone is connected, the video area lists numbered steps for each enabled link (USB, Wi-Fi) with its live state. Right, scrolling on its own: the **signal chain** Glasses → iPhone → Computer → Decode → Hands, each hop with its frame rate against the configured one, a 20 s sparkline, and amber or red with the reason in words when it falls behind ("Phone dropped 12 frames in 10 s"); then Virtual camera (see below); then Link (phone → computer delay chart, round trip, each link's state), Hands (rate, time per frame, each hand's confidence, whether the skeleton belongs to the frame it is drawn on), Glasses (battery ring, worn / hinge / thermal / link, recent inputs), Motion (accel, gyro and magnetometer, x/y/z over 20 s) and the recordings, newest first. Keys: `R` record, `H` hand tracking, `V` virtual camera, `P` pairing. The page needs no internet connection and stacks vertically on a phone, with the transport bar pinned to the bottom.

| Flag | Meaning |
|---|---|
| `--web-port PORT` | port of the page, 8765 by default; `0` picks a free one (read it from `WEB_URL`) |
| `--web-host HOST` | address to listen on, `127.0.0.1` by default (this computer only). `0.0.0.0` lets other devices (a laptop, a tablet) open the page: see Access |
| `--open` | open the page in the default browser (never done otherwise) |
| `--headless` | no web page at all: recording and logs only (with `--record`, `--duration`, `--snapshot`) |
| `--snapshot PNG` | at exit, save the last video frame, full size, with the skeleton (a placeholder if there was no video) |

### Access

- On a loopback address (the default) there is no key. The server still refuses requests whose `Host` header is not `127.0.0.1`, `localhost` or `[::1]`, so a web site that points its own domain at 127.0.0.1 (DNS rebinding) cannot read the status or the pairing code.
- On any other address the receiver logs a one-line warning and creates a random access key at every start. `WEB_URL` carries it (`http://192.168.1.20:8765/?key=...`); open that URL once and the page stores the key in an HttpOnly, SameSite=Strict cookie and drops it from the address bar. Every request without the key gets 403: the page, the status, the history, both streams, the recordings, the pairing code and the actions. Treat the URL like a password; restart the receiver to change the key.
- Actions (POST) also need the header `X-GlassesData: 1`, which a page on another site cannot send without a CORS preflight (never granted), and a matching `Origin` when the browser sends one. So a malicious site cannot start or stop a recording through your browser.
- The traffic is plain HTTP: on a network you do not trust, keep the default `127.0.0.1` and use an SSH tunnel (`ssh -L 8765:127.0.0.1:8765 host`) instead of `--web-host`.

### Endpoints

| Path | Method | Answer |
|---|---|---|
| `/` | GET | the page (one file, inline CSS and JS) |
| `/api/status` | GET | JSON: `link` (connected, text, kind, device, per-source `sources` texts), `video` (format, has_frame, `glasses_fps`, `phone_sent_fps`, `received_fps`, `decoded_fps`, `phone_dropped_frames`, `phone_dropped_fps`, `delay_ms`, `rtt_ms`, `display_skipped`, `decode_errors`, `decode_backlog`), `hands` (available, enabled, error, fps, process_ms, count, detected), `motion` (accel, gyro, mag, source, samples_per_s), `glasses` (state, inputs), `recording` (active, name, seconds, frames, note, waiting_for_keyframe, last, folder), `wifi` (enabled, pairing_ready, pair_at_start), `vcam` (available: pyvirtualcam can be imported, running, device, backend, width, height, fps, mode, position, overlay, frames_sent, showing: `video` or `card`, check: the start-up check's result, last_error, hint: what to do, in words). The page polls it 4 times a second. |
| `/api/history` | GET | JSON `{"interval_s", "seconds", "fields", "rows"}`: the last 30 s of per-hop frame rates, counters (dropped frames, display skips, decode errors), delay and motion, one row every 0.1 s (values are null while nothing is connected); `?since=T` returns only rows after time `T`. The page's charts use it. |
| `/api/live.mjpg` | GET | MJPEG like `/api/stream.mjpg` but without the skeleton drawn in. Every part carries `X-Frame` (the frame index) and `X-Hands`, compact JSON `{"frame", "enabled", "lag", "hands": [{"side", "score", "points": [x0, y0, ... x20, y20]}]}` (normalised to the image), so the page draws each frame's skeleton from the same part as its picture. When the tracker keeps up, the encoder waits briefly (at most 60 ms) for the result of that very frame (`lag` 0); otherwise `lag` says how many frames older the landmarks are. |
| `/api/stream.mjpg` | GET | MJPEG (`multipart/x-mixed-replace; boundary=gdframe`): the decoded frame with the skeleton, at most 720 px tall, JPEG quality 80, at most the video's frame rate. Each viewer gets the newest frame and skips what it was too slow for; a stalled viewer never slows decoding or recording and is dropped after 10 s. The page falls back to it when the browser cannot read a streaming `fetch`. |
| `/api/recordings` | GET | JSON `{"folder", "recordings": [{"name", "started", "seconds", "frames", "bytes", "state"}]}`, newest first; `state` is `recording`, `complete` or `interrupted` (the receiver stopped without closing it) |
| `/api/pairing.png` | GET | the Wi-Fi pairing QR code; 404 without `--wifi` |
| `/api/record/start` | POST | start recording: `{"ok", "message", "name"}`; 409 with the reason when it cannot (no iPhone, no video yet) |
| `/api/record/stop` | POST | stop recording: `{"ok", "message", "name"}` |
| `/api/hands/toggle` | POST | hand tracking on/off: `{"ok", "message", "enabled"}`; 409 if unavailable |
| `/api/vcam/start` | POST | start the virtual camera: `{"ok", "message", "vcam"}` (`vcam` as in the status); 409 with the reason when the device cannot be opened |
| `/api/vcam/stop` | POST | stop it: `{"ok", "message", "vcam"}` |
| `/api/vcam/settings` | POST | JSON body with any of `"mode": "fit" \| "fill"`, `"position": 0..1` (0 top, 1 bottom), `"overlay": true \| false`; applies to the next frame sent. 400 for anything else (unknown keys, wrong types, out of range), and nothing changes |
| `/api/vcam/preview.jpg` | GET | the picture the virtual camera is sending right now (also the placeholder card), 480 px wide; 404 while it is off. The page's thumbnail |

For scripts: `curl -X POST -H 'X-GlassesData: 1' http://127.0.0.1:8765/api/record/start`, or `curl -X POST -H 'X-GlassesData: 1' -d '{"mode": "fill", "position": 0.8}' http://127.0.0.1:8765/api/vcam/settings` (add `?key=...` from `WEB_URL` on a non-loopback host).

When the page opens while video is flowing, the first picture appears within about 0.1 s. If the stream request hangs (for example behind the browser's limit of 6 connections per host, or in an embedded browser view), the page gives up after 1 s, then 2 s, 4 s, and reopens it. A tab in the background closes its stream after 5 s and reopens it the moment it is in front again. The stage says "waiting for the next keyframe" only when the receiver really has no decoded frame yet; otherwise it says the picture is loading, or that it is paused while the tab is in the background.

## Connecting the iPhone

**USB** (default): plug in the iPhone, trust the computer, open GlassesData and turn on Computer link (USB).

```sh
python -m glassesdata_receiver --usb
```

The receiver reaches the app through `usbmuxd` (part of macOS; the `usbmuxd` package on Linux); the cable and the iPhone's trust prompt are the security boundary, so there is no pairing and no encryption.

**Wi-Fi**: the iPhone connects to the computer over TLS 1.3. Pair once:

```sh
python -m glassesdata_receiver --wifi --pair
```

Scan the QR code (Pair over Wi-Fi on the web page, which `--pair` opens at start, and in the terminal) in the app: Computer card > Pair over Wi-Fi. It carries the computer's addresses (LAN ranges 192.168/16, 10/8, 172.16/12 first, carrier-grade NAT such as Tailscale's 100.64/10 last, because the phone tries them in order; container and VM bridges such as `docker0` are left out) and port, the SHA-256 fingerprint of the receiver's self-signed certificate (the phone pins it) and a random token (the receiver rejects any phone without it). Treat it like a password. Later runs only need `--wifi`; the phone finds the computer through Bonjour (`_glassesdata._tcp`) or the addresses in the code.

| Flag | Meaning |
|---|---|
| `--state-dir DIR` | where the identity lives: `cert.pem`, `key.pem`, `pairing.json` (default: see Folders) |
| `--reset-pairing` | new certificate and token; every paired phone must scan again |
| `--print-pairing` | with `--wifi`: print `PAIRING_URI=<uri>` on stdout once the listener accepts connections |
| `--no-bonjour` | do not advertise on the local network |
| `--port N` | link port, 7900 by default (the app's USB port, or the computer's Wi-Fi port) |

Other flags: `--out DIR` (recordings), `--record` (start recording as soon as video arrives, and again after a reconnect; a manual stop is respected until the next connection), `--no-hands`, `--max-hands N`, `--mirrored-input` (see Handedness), `--duration S`. See `--help`.

## Record from the iPhone (GDL1 remote control)

The phone can start and stop recording on the computer (the app's Record button; the iOS side is a later task). GDL1 message `CONTROL` (0x50, phone to computer) carries `{"t": <phone ns>, "cmd": "record_start" | "record_stop"}`; the receiver answers with `RECORDER_STATE` (0x51) `{"recording", "name", "seconds", "frames", "note"}`, which it also sends right after HELLO_ACK, on every change (start, stop, a new folder after a stream restart, a failure) and once a second while recording. A refused command (already recording, no video yet, not recording) does not end the connection: the answer's `note` says why. Details in `glassesdata_receiver/protocol.py`.

## Recordings

Each recording is a folder named after its start time, `YYYY-MM-DD_HHMMSS` (a suffix `_2`, `_3`... if two start in the same second). A recording also ends, and the next one starts in a new folder, when:

- the video format (size or HEVC parameters) changes;
- the glasses stream restarts with the same format: a keyframe arrives whose `pts_us` is not after the last recorded one. The new recording starts at that keyframe, so no video is lost; the log says why and the page shows it as a notice above the signal chain. (A non-keyframe with an old timestamp is dropped: MP4 timestamps must increase.)


```
video.mp4         original HEVC, fragmented MP4 (stays playable if the receiver is killed)
frames.csv        one row per frame in video.mp4
hands.csv         live hand skeletons (frames the tracker skipped are missing; see offline.py)
motion.csv        IMU samples
inputs.csv        button and touch events
device_state.csv  glasses state; the first row is the state known when the recording started
clock.csv         phone/computer clock samples (PING/PONG)
info.json         start/end times, video size, frame count, first/last pts, app, link, handedness convention
```

### Clocks

Every row keeps the clocks it was stamped with:

- `pts_us`: the glasses video timeline (microseconds), from the SDK's sample buffers.
- `phone_time_ns`: the iPhone's monotonic clock when the phone received the item.
- `mac_time_ns`: the receiving computer's monotonic clock (`time.monotonic_ns`) when the item arrived here. The name is historical; on Linux it is the Linux computer's clock.
- `glasses_time_ns` / `glasses_time_ms`: the glasses' own sensor and input clocks.

### Columns

| File | Columns |
|---|---|
| `frames.csv` | `frame_index` (frame number in video.mp4), `pts_us`, `phone_time_ns`, `mac_time_ns`, `keyframe` (1/0), `bytes` (sample size) |
| `motion.csv` | `phone_time_ns`, `glasses_time_ns`, `ax ay az` acceleration (m/s²), `gx gy gz` rotation rate (rad/s), `mx my mz` magnetic field (µT), `qw qx qy qz` orientation quaternion, `source`, `mac_time_ns`. Empty cells: the sensor did not report that value. |
| `inputs.csv` | `phone_time_ns`, `glasses_time_ms`, `source`, `type`, `detail`, `mac_time_ns` |
| `device_state.csv` | `phone_time_ns`, `battery` (%), `charging`, `don` (worn or not), `hinge`, `thermal`, `link`, `mac_time_ns`. One row per DEVICE_STATE message from the phone. |
| `hands.csv` | `pts_us`, `phone_time_ns` (of the frame), `hand_index`, `handedness`, `score`, then `x0 y0 z0 ... x20 y20 z20` (x, y normalised to the image, 0..1 from the top left; z depth relative to the wrist, smaller is nearer the camera), then `wx0 wy0 wz0 ... wz20` (metres, origin near the hand's centre). Joint numbering follows MediaPipe: 0 wrist, 4 thumb tip, 8 index tip, 12 middle tip, 16 ring tip, 20 little tip. Join with frames.csv on `pts_us`. |
| `clock.csv` | `mac_send_ns` (PING sent), `mac_receive_ns` (PONG received), `phone_time_ns` (phone clock when it answered), `rtt_ns`, `offset_ns` (phone minus Mac) |

### Handedness

`handedness` is anatomical: `Left` is the person's own left hand, in `hands.csv`, `hands_offline.csv` and the page's labels and colours (cyan L for left, pink R for right). This assumes a normal, non-mirrored camera such as the glasses'. MediaPipe's documentation says its labels assume a mirrored selfie image, but measured on a non-mirrored photo the Tasks model already names each person's own hands, and flipping the image swaps them; so the labels are kept as they are. For a mirrored source (a selfie-style recording), pass `--mirrored-input` to the receiver or to `offline.py`, which swaps them. `info.json` records the convention used: `handedness` for the live tracker, `handednessOffline` for `offline.py`.

### Phone time to computer time

The receiver sends a PING every second; the phone answers with its clock. Assuming the answer was stamped half-way through the round trip:

```
offset_ns = phone_time_ns - (mac_send_ns + mac_receive_ns) / 2
mac_time_ns ≈ phone_time_ns - offset_ns
```

Use the rows with the smallest `rtt_ns`: they carry the least queueing error. For a long recording, fit `offset_ns` against `mac_receive_ns` over the low-`rtt_ns` rows (a straight line absorbs clock drift) instead of using one constant. Computer time then lines up video frames (through `frames.csv`), hands, IMU and inputs on one axis.

## Offline hand tracking

The live tracker only works on the newest frame and skips frames when it falls behind. To track every frame of a recording afterwards:

```sh
python -m glassesdata_receiver.offline ~/Movies/GlassesData/2026-11-20_212051   # ~/Videos/GlassesData/... on Linux [--max-hands 2] [--mirrored-input]
```

It writes `hands_offline.csv` (same columns as `hands.csv`) into the folder, taking `pts_us` and `phone_time_ns` from `frames.csv` by frame index, and prints a one-line summary.

## Testing without the iPhone

`tools/fake_sender.py` plays an HEVC file (no B-frames) as if it were the app, with simulated IMU, device state, stats and a skewed phone clock. A demo of the web page without hardware:

```sh
python tools/fake_sender.py ../CameraAccess/TestResources/plant.mp4 --listen 7901 --loop
python -m glassesdata_receiver --tcp 127.0.0.1:7901       # then open the printed WEB_URL
```

Wi-Fi style (the receiver listens, the sender connects over TLS and checks the fingerprint):

```sh
python -m glassesdata_receiver --wifi --no-bonjour --print-pairing
python tools/fake_sender.py VIDEO.mp4 --connect 127.0.0.1:7900 --fingerprint FP --token TOKEN
```

`fp` and `token` are in the printed `PAIRING_URI`. Other options: `--seconds`, `--fps`, `--drop-every N`, `--clock-skew-ms`, `--listen 0` (any free port; the sender prints it), and `--record-after N --record-for M`, which press the phone's Record button: CONTROL `record_start` N seconds after connecting, `record_stop` M seconds later; the sender prints every `RECORDER_STATE` it receives. Unit tests: `python -m pytest -p no:cacheprovider` (no device needed; they use loopback ports only).

## Virtual camera

Zoom, Meet, Teams and other apps can use the glasses video as a webcam. The receiver composes each frame onto a **landscape** picture and sends it to a virtual camera device, on its own thread at a steady 30 fps (it repeats the last picture until the next frame arrives, because apps expect a constant rate). Recording and decoding never wait for it.

### Set up once

1. `~/.venvs/glassesdata-receiver/bin/pip install pyvirtualcam` (0.15 or later). Without it everything else works, and the page says it is missing.
2. **macOS:** install [OBS Studio](https://obsproject.com) (30 or later). Open it and click **Start Virtual Camera** once. macOS asks to allow the OBS camera extension: allow it in System Settings > General > Login Items & Extensions > Camera Extensions. Then click **Stop Virtual Camera** and quit OBS. macOS only lets a signed camera extension provide a system camera; OBS ships one, so the receiver does not need its own. OBS does not need to run afterwards.
3. **Linux (Ubuntu 24.04):** pyvirtualcam writes to a v4l2loopback device: `sudo apt install v4l2loopback-dkms`, then `sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label=GlassesData` (again after every reboot, or add it to `/etc/modules-load.d`). The same canvas, modes and controls apply. Untested on real hardware so far.
4. In the other app, choose the camera **OBS Virtual Camera** (macOS) or **GlassesData** (Linux).

### Use

```sh
python -m glassesdata_receiver --usb --virtual-camera            # starts with it on
python -m glassesdata_receiver --usb                              # off; start it on the web page (V)
```

| Flag | Meaning |
|---|---|
| `--virtual-camera` | start with the virtual camera on. Without it, it is off but can be started from the page at any time |
| `--virtual-camera-size 1920x1080\|1280x720` | the picture sent, 1920x1080 by default |
| `--virtual-camera-mode fit\|fill` | the starting mode (default `fit`) |
| `--overlay` | start with the hand skeleton drawn into the picture |

If pyvirtualcam is missing, or the device cannot be opened (OBS not installed, extension not allowed, v4l2loopback not loaded), the receiver keeps running; the log and the page say why and what to do.

**Two ways to fit a portrait video into a landscape picture:**

- **Fit** (default): the whole frame, at the full height of the picture, centred. The sides show a blurred, darkened, enlarged copy of the same frame (the usual treatment of portrait video on a landscape screen). Nothing of the frame is lost; it is small in a call.
- **Fill**: a landscape band of the frame fills the whole picture. **Band** moves it from the top (0) to the bottom (1) of the frame (default the middle): hands are often low in the glasses' view, so move the band down to keep them in the call.

A landscape source is fitted exactly (16:9) or with blurred bars, or cropped to cover in fill mode. With the **hand skeleton** on, it is drawn on the video part only, in the right place for the band, while hand tracking is on (`H`). Without video (no iPhone connected, no video yet, or no new frame for 3 s) the camera shows a calm card, "GlassesData · No video" and why, never a frozen frame or black.

**On the page:** the **Virtual camera** panel, under the signal chain, has the on/off switch with the device name, Fit / Fill, the band slider (fill only), the hand-skeleton switch and a live thumbnail of exactly what is being sent, so you can check the framing without opening Zoom. The camera button in the transport bar and the `V` key turn it on and off. The panel and the button follow the page's access rules (the same key, header and origin checks as Record).

### What the receiver can and cannot know

**Delivery cannot be confirmed by the sender.** The virtual camera API only takes frames; it never says whether an app receives them. And the OBS Virtual Camera on macOS only takes landscape 1920x1080 or 1280x720 frames: a frame of any other size (for example the glasses' own 720x1280) is dropped silently. The open succeeds, `send` raises nothing, and OBS only prints `Pixel buffer size mismatch` on standard error. This is why the receiver always sends a landscape canvas of exactly the configured size, never the video's own size.

At start the receiver logs a self-check: the device it opened, the backend, the size and the rate. On macOS it also watches its own standard error for 2 s for OBS's rejection message, and reports it in the log and on the page ("OBS rejects the frames") if it appears. A clean check means only that OBS did not complain. To see the picture arrive, open the camera in an app:

1. Start the receiver with `--virtual-camera` (or press `V` on the page).
2. Open QuickTime Player > File > New Movie Recording, and choose **OBS Virtual Camera** from the menu next to the record button.

## Ubuntu 24.04

**The Linux path is untested on real hardware.** It was written and unit-tested on a Mac (platform branches tested with a simulated `sys.platform`); nobody has yet run it on Ubuntu with an iPhone attached. Expect to report and fix small problems on the first run.

1. **Python 3.11.** Ubuntu 24.04 ships Python 3.12 and has no `python3.11` package in its own archive. Install 3.11 from the deadsnakes PPA (`sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.11 python3.11-venv`) or with uv (`uv python install 3.11`, then `uv venv --python 3.11 ~/.venvs/glassesdata-receiver`). The receiver is only tested with 3.11.
2. **Virtual environment**, as in Setup: `python3.11 -m venv ~/.venvs/glassesdata-receiver` and `pip install -r requirements.txt`. If `import cv2` then fails with `libGL.so.1` or `libgthread` missing (a minimal or server install), `sudo apt install libgl1 libglib2.0-0t64`.
3. **USB:** `sudo apt install usbmuxd` (and `libimobiledevice-utils` for `idevicepair`). udev starts usbmuxd when an iPhone is plugged in, so before that the page says to plug it in. Unlock the iPhone and tap Trust; `idevicepair validate` checks the pairing. The receiver talks to `/var/run/usbmuxd`, the same socket path as on macOS.
4. **Firewall.** Ubuntu's `ufw` is off by default. If it is on, `--wifi` needs `sudo ufw allow 7900/tcp` (the link) and `sudo ufw allow 5353/udp` (Bonjour), and `--web-host 0.0.0.0` needs the web port, best limited to your LAN: `sudo ufw allow from 192.168.1.0/24 to any port 8765 proto tcp`. Without `--web-host` the page is only reachable from the laptop itself and needs no rule.
5. **GPU.** The NVIDIA GPU is not used: PyAV decodes HEVC and MediaPipe's Python package runs the hand model on the CPU. One 720p stream at 24-30 fps should be well within a laptop CPU; the page's signal chain shows whether decoding or tracking falls behind.
6. Viewing from another device: start with `--web-host 0.0.0.0` and open the printed `WEB_URL` (with its key) on that device.
