# GlassesData receiver

Receives the Meta glasses video (HEVC) and sensor data that the GlassesData iPhone app forwards over the GDL1 protocol, and on this computer (macOS; Ubuntu 24.04 see below):

- records the original HEVC into `video.mp4` (no re-encoding), plus CSVs of every sensor row;
- decodes the video and tracks hand skeletons (MediaPipe, 21 joints per hand, 2D and 3D);
- serves a local web page: the live video with the skeleton, the link and sensor status, a Record button and the list of recordings;
- calibrates the glasses' camera with a ChArUco board shown on this computer's screen, and keeps the person's body measurements; every new recording gets a copy of both (`calibration.json`, `profile.json`, the formats of `Docs/PIPELINE.md` sections 3 and 4);
- feeds the video to Zoom, Meet and other apps as a webcam, through the OBS Virtual Camera (macOS) or v4l2loopback (Linux): see Virtual camera.

The receiver is a command-line process; it never opens a window. Open the page it prints (`WEB_URL=...`) in any browser.

## Setup

Python 3.11 on macOS; on Ubuntu 24.04 its own Python 3.12, set up by one script (see Ubuntu 24.04). Create the virtual environment **outside** any cloud-synced folder (OneDrive, iCloud Drive, Dropbox): syncing thousands of small files breaks or slows the environment.

```sh
python3.11 -m venv ~/.venvs/glassesdata-receiver
~/.venvs/glassesdata-receiver/bin/pip install -r requirements.txt        # requirements-dev.txt adds pytest
```

Run everything from this `Receiver/` folder with that interpreter (below, `python` means `~/.venvs/glassesdata-receiver/bin/python`). The first run downloads the hand model (see Folders).

### Folders

| | macOS | Linux |
|---|---|---|
| recordings (`--out`) | `~/Movies/GlassesData` | `GlassesData` in the desktop's Videos folder: `~/Videos/GlassesData`, or the folder's name in the session's language (`XDG_VIDEOS_DIR` in `~/.config/user-dirs.dirs`, such as `~/视频/GlassesData`) |
| state (`--state-dir`): Wi-Fi identity, `calibration/`, `profile.json` | `~/Library/Application Support/GlassesData` | `$XDG_STATE_HOME/glassesdata` (default `~/.local/state/glassesdata`) |
| hand model cache | `~/Library/Caches/GlassesData` | `$XDG_CACHE_HOME/glassesdata` (default `~/.cache/glassesdata`) |

## The web page

```sh
python -m glassesdata_receiver --usb            # prints WEB_URL=http://127.0.0.1:8765/
python -m glassesdata_receiver --usb --open     # also opens it in the default browser
```

A dark monitoring surface (it follows the browser's light or dark setting), made to stay open on a second screen. Left: the video at its own aspect (9:16 for the glasses), as tall as the window allows, with the hand skeleton drawn by the browser (crisp at any size; cyan L for the left hand, pink R for the right), and under it the transport bar: the round Record button, the timer, frames and folder while recording, the hand-tracking toggle, the virtual camera toggle and, with `--wifi`, Pair. When no iPhone is connected, the video area lists numbered steps for each enabled link (USB, Wi-Fi) with its live state. Right, scrolling on its own: the **signal chain** Glasses → iPhone → Computer → Decode → Hands, each hop with its frame rate against the configured one, a 20 s sparkline, and amber or red with the reason in words when it falls behind ("Phone dropped 12 frames in 10 s"); then Virtual camera (see below); then Link (phone → computer delay chart, round trip, how far the preview is behind, the phone's video queue, its drop episodes and whether it buffers or drops when the link is slow, each link's state), Hands (rate, time per frame, each hand's confidence, whether the skeleton belongs to the frame it is drawn on), Glasses (battery ring, worn / hinge / thermal / link, recent inputs), Camera calibration and Body measurements (see below), Motion (accel, gyro and magnetometer, x/y/z over 20 s) and the recordings, newest first. When the live picture is more than 0.5 s behind real time, an amber "Preview 1.3 s behind" appears in the top bar (red beyond 3 s) and in the signal chain's summary, never over the video; the iPhone hop says when frames queue on the phone. Keys: `R` record, `H` hand tracking, `V` virtual camera, `P` pairing. The page needs no internet connection and stacks vertically on a phone, with the transport bar pinned to the bottom.

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
- Actions (POST) also need the header `X-GlassesData: 1`, which a page on another site cannot send without a CORS preflight (never granted), and a matching `Origin` when the browser sends one. So a malicious site cannot start or stop a recording through your browser. The calibration and body-measurement actions follow the same rules.
- The traffic is plain HTTP: on a network you do not trust, keep the default `127.0.0.1` and use an SSH tunnel (`ssh -L 8765:127.0.0.1:8765 host`) instead of `--web-host`.

### Endpoints

| Path | Method | Answer |
|---|---|---|
| `/` | GET | the page (one file, inline CSS and JS) |
| `/api/status` | GET | JSON: `link` (connected, text, kind, device, per-source `sources` texts), `video` (format, has_frame, `glasses_fps`, `phone_sent_fps`, `received_fps`, `decoded_fps`, `phone_dropped_frames`, `phone_dropped_fps`, `delay_ms`, `rtt_ms`, `display_skipped`, `decode_errors`, `decode_backlog`, `preview_lag_s`: how far the newest decoded frame is behind real time, see Preview lag; from newer phones, else null: `phone_queue_frames` (video frames waiting on the phone), `phone_record_priority` (the phone buffers instead of dropping because this computer records), `phone_gaps` (drop episodes since connecting)), `hands` (available, enabled, error, fps, process_ms, count, detected), `motion` (accel, gyro, mag, source, samples_per_s), `glasses` (state, inputs), `recording` (active, name, seconds, frames, note, waiting_for_keyframe, last, folder), `wifi` (enabled, pairing_ready, pair_at_start), `vcam` (available: pyvirtualcam can be imported, running, device, device_label: the name apps show (Linux), requested_device: `--virtual-camera-device`, loopback: on Linux `{"module_loaded", "devices": [{"path", "label"}]}`, else null, backend, width, height, fps, mode, position, overlay, frames_sent, showing: `video` or `card`, check: the start-up check's result, last_error, hint: what to do, in words; on Linux also before any start when there is no v4l2loopback device). The page polls it 4 times a second. |
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
| `/api/calibration` | GET | JSON: `screen` (the saved screen scale), `current` (the calibration new recordings get, with `fx fy cx cy` and `fov_deg`, or null), `session` (null, or the capture: `capturing`, `board`, `video`, `samples`, `target`, `max`, `min_solve`, `coverage` (`cols`, `rows`, `cells`: kept views per cell, `fraction`), `distances`, `sides`, `last` (the last look: `corners`, `steady`, `accepted`, `reason`, `points`, `distance_m`, `tilt_deg`), `guidance`, `sequence`, `accepted_total`, `solving`, `solution`), `card_mm`, `storage` |
| `/api/calibration/board.png` | GET | the board of the running session, one image pixel per device pixel; 404 without a session |
| `/api/calibration/preview.jpg` | GET | the last frame looked at, 360 px tall, with the markers and corners found; 404 before the first look |
| `/api/calibration/undistorted.jpg` | GET | the solution's check picture: a kept view as captured and undistorted, side by side, board rows and columns joined; 404 before solving |
| `/api/calibration/screen` | POST | `{"css_px_per_mm", "device_pixel_ratio", "screen_css": [w, h]}`: save the screen scale measured with a card (`<state>/calibration/screen.json`) |
| `/api/calibration/start` | POST | `{"css_px_per_mm", "device_pixel_ratio", "area_css": [w, h]}`: design the board for that part of the screen and capture. The same board as the running session resumes it (`"kept": true`, its views stay); another board starts over. Answers `board` (squares, sizes in metres, `width_px`, `height_px`, `css_width`, `css_height`) |
| `/api/calibration/capture` | POST | `{"active": true \| false}`: resume or pause looking at frames |
| `/api/calibration/solve` | POST | fit both models to the kept views (a few seconds): `{"ok", "message", "solution"}`; 409 with fewer than 6 views or no video |
| `/api/calibration/save` | POST | write the solution as `calibration.json`: `<state>/calibration/calibration_<UTC>.json` and `current.json` |
| `/api/calibration/reset` | POST | drop the session and its views |
| `/api/profile` | GET | `{"profile", "fields", "file", "saved", "note"}`: the body measurements (PIPELINE.md section 4), the form's fields with their definitions, defaults and plausible ranges |
| `/api/profile` | POST | `{"hand_length_m": 0.19, "palm_width_m": null, ...}`: a number (metres) marks a value measured, null puts it back to its default, keys not named keep theirs. 400 with `errors` per field for anything out of range (nothing is saved) |

For scripts: `curl -X POST -H 'X-GlassesData: 1' http://127.0.0.1:8765/api/record/start`, or `curl -X POST -H 'X-GlassesData: 1' -d '{"mode": "fill", "position": 0.8}' http://127.0.0.1:8765/api/vcam/settings` (add `?key=...` from `WEB_URL` on a non-loopback host).

When the page opens while video is flowing, the first picture appears within about 0.1 s. If the stream request hangs (for example behind the browser's limit of 6 connections per host, or in an embedded browser view), the page gives up after 1 s, then 2 s, 4 s, and reopens it. A tab in the background closes its stream after 5 s and reopens it the moment it is in front again. The stage says "waiting for the next keyframe" only when the receiver really has no decoded frame yet; otherwise it says the picture is loading, or that it is paused while the tab is in the background.

## Camera calibration

The pipeline needs the camera's intrinsics (focal length, principal point, lens distortion). Without them it assumes a 65° horizontal field of view and no distortion. Calibrate once per glasses and video resolution, on the computer the receiver runs on, with the glasses on. The **Camera calibration** panel shows the current calibration (field of view, model and error, focal length, centre, video size, when it was made; amber when the stream has another size) and the **Calibrate** button, which opens three steps over the page:

1. **Screen scale.** Hold a bank card (any ISO/IEC 7810 ID-1 card: 85.60 × 53.98 mm) against the screen in the corner mark and drag the outline's handle, or press − and + (arrow keys; Shift is faster), until it matches the card. That gives CSS pixels per millimetre; the page also shows the screen size it implies, as a sanity check. The scale is kept in the browser (`localStorage`, as device pixels per mm, which page zoom does not change) and by the receiver (`screen.json`). If the device pixel ratio or the screen changes later (zoom, another monitor), the page says so and asks to measure again. The scale sets the board's real size, which the pipeline uses for distances; the lens calibration itself does not depend on it.
2. **Capture.** *Show the board* fills the screen (Fullscreen API; a window-sized overlay where that is refused) with a white page and the board, drawn by OpenCV (`cv2.aruco.CharucoBoard`, `DICT_5X5_100`) at exactly one image pixel per device pixel, at a whole-device-pixel position, so it is sharp and of known size: squares of 25 to 45 mm (about six along the short side, as many as fill the screen; smaller only on a screen too small for that), markers of whole modules at about 0.75 of the square. Beside it (below it on a portrait screen) a compact panel, readable through the glasses, shows the views kept against the target (25), a coverage map of the camera image with the last corners found, the corners, distance and tilt of the last look, and one line of guidance: *Look at the board*, *Move closer*, *Show the whole board*, *Hold still*, *Next: board in the top left of your view*, *Next: step back*, *Next: look at the board from the left*, *Enough views: press Solve*. `Esc` leaves the board (capture pauses, the views stay).
3. **Result.** *Solve* fits both OpenCV models: pinhole (`cv2.calibrateCamera`, k1 k2 p1 p2 k3) and fisheye (`cv2.fisheye.calibrate`, k1..k4, started from the pinhole focal length). Views with an error above 3× the median are dropped and the model fitted again. The model with the lower RMS error on views left out of the fit (5-fold) is kept; fisheye only when it is clearly lower (below 90 % of the pinhole one). The page shows the error, the field of view (horizontal, vertical, diagonal, distortion included), the parameters, both held-out errors, the error of every view (dropped ones in red) and a kept view as captured and undistorted, with the board's rows and columns joined: straight lines when the calibration is right. *Save* writes `calibration.json` exactly as PIPELINE.md section 3 describes it (`source: "charuco-web"`, `device: "Ray-Ban Meta"`, with the `board` and its `css_px_per_mm`) to `<state>/calibration/calibration_<UTC time>.json` and `current.json`.

While capturing, a thread looks at the receiver's newest decoded frame at most 5 times a second (it only waits on the decoder's notification, so decoding, recording and the stream never wait for it), detects the board (`CharucoDetector.detectBoard`) and keeps a view when it has at least 12 corners, has hardly moved since the previous look 0.2 s earlier (no motion blur), is not a near copy of a kept view, and adds something: a cell of the image (6 across the short side) no kept view covered, or a distance or tilt bin (near / mid / far; under 15°, 15–30°, over 30°, from the left, right, above or below) with fewer than two views. Distance and tilt come from the board's pose under a provisional camera: the current calibration when it has the same aspect ratio, else the 65° guess. It keeps at most 40 views; solving needs 6. While the calibration view is open, the page pauses its own live stream.

The code is `glassesdata_receiver/calibration.py`. Unit tests render the board through a known camera (`tests/charuco_synth.py`: every pixel traced back to the board plane through OpenCV's distortion model, supersampled, blurred, noisy) and check that fx, fy, cx and cy come back within 1 % at an RMS under 0.5 px, and that a fisheye lens is recognised as one. `tests/test_calibration.py::test_end_to_end_calibration_over_http_and_a_recording_that_carries_both_files` encodes such views as HEVC, streams them with `tools/fake_sender.py` into `python -m glassesdata_receiver --tcp`, drives the whole calibration through the HTTP API and checks the result against the true camera.

## Body measurements

The pipeline sizes the hand and arm model from six measurements (PIPELINE.md section 4); anything not measured is an adult average. The **Body measurements** panel lists them (measured values in white, averages in grey); *Edit measurements* opens a form in centimetres beside a drawing of an arm, the shoulders and a hand that highlights where each one goes: (1) hand length, wrist crease to middle fingertip; (2) palm length, wrist crease to the base crease of the middle finger; (3) palm width, across the knuckles of index to little finger; (4) forearm, wrist crease to the tip of the elbow; (5) upper arm, shoulder tip (acromion) to the tip of the elbow; (6) shoulder width, between the two shoulder tips. An empty field keeps the average (shown as the placeholder). Values outside plausible adult ranges (hand 13–25 cm, palm length 7.5–14, palm width 6–11.5, forearm 18–36, upper arm 22–42, shoulders 28–50; palm at least 2 cm shorter than the hand) are refused. They are saved in metres to `<state>/profile.json` with `measured` flags for all eight keys (the two eye-to-shoulder offsets always keep their defaults). The code is `glassesdata_receiver/bodyprofile.py`.

## Preview lag

`preview_lag_s` in the status, and the "Preview N.N s behind" pill: this computer's monotonic clock now, minus the newest decoded frame's `phone_time_ns` mapped to this clock with the PING/PONG offset (the sample with the smallest round trip, as in Phone time to computer time below). It grows when frames queue on the phone (the phone buffers rather than drops while this computer records), on the link, or in the decoder, and while no new frame arrives at all. The phone reports its side in STATS: `queue_frames` (video frames not sent yet), `record_priority` (true while it buffers for a recording instead of dropping) and `gaps` (drop episodes since connecting); older phones leave them out and the page shows nothing for them.

## Connecting the iPhone

**USB** (default): plug in the iPhone, trust the computer, open GlassesData and turn on Computer link (USB).

```sh
python -m glassesdata_receiver --usb
```

The receiver reaches the app through `usbmuxd` (part of macOS; the `usbmuxd` package on Linux); the cable and the iPhone's trust prompt are the security boundary, so there is no pairing step of the receiver's own and no encryption. On Linux, trust the computer once (Ubuntu 24.04 > USB).

**Wi-Fi**: the iPhone connects to the computer over TLS 1.3. Pair once:

```sh
python -m glassesdata_receiver --wifi --pair
```

Scan the QR code (Pair over Wi-Fi on the web page, which `--pair` opens at start, and in the terminal) in the app: Computer card > Pair over Wi-Fi. It carries the computer's addresses (LAN ranges 192.168/16, 10/8, 172.16/12 first, carrier-grade NAT such as Tailscale's 100.64/10 last, because the phone tries them in order; container and VM bridges such as `docker0` are left out) and port, the SHA-256 fingerprint of the receiver's self-signed certificate (the phone pins it) and a random token (the receiver rejects any phone without it). Treat it like a password. Later runs only need `--wifi`; the phone finds the computer through Bonjour (`_glassesdata._tcp`) or the addresses in the code.

| Flag | Meaning |
|---|---|
| `--state-dir DIR` | where the receiver keeps its state: the identity (`cert.pem`, `key.pem`, `pairing.json`), the camera calibration (`calibration/`) and the body measurements (`profile.json`) (default: see Folders) |
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
calibration.json  the camera calibration in use when the recording started (a copy of <state>/calibration/current.json)
profile.json      the body measurements in use when the recording started (a copy of <state>/profile.json)
info.json         start/end times, video size, frame count, first/last pts, app, link, handedness convention,
                  and what was copied: "calibration" and "profile"
```

`info.json` describes the two copies: `"calibration": {"file": "calibration.json", "model", "width", "height", "rms_px", "created", "source"}` and `"profile": {"file": "profile.json", "measured": [...]}`. When there is nothing to copy, `"file"` is null, nothing is written and a `note` says why (the pipeline then uses its default camera or the adult averages). A calibration made at another video size is still copied, with a `note`: the pipeline rescales K for the same aspect ratio and refuses a different one.

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
python -m glassesdata_receiver.offline ~/Movies/GlassesData/2026-11-20_212051   # on Linux the Videos folder, see Folders [--max-hands 2] [--mirrored-input]
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

`fp` and `token` are in the printed `PAIRING_URI`. Other options: `--seconds`, `--fps`, `--drop-every N`, `--clock-skew-ms`, `--lag-ms MS` (stamp every frame MS earlier, as if it had queued on the phone: the page shows the preview that far behind), `--listen 0` (any free port; the sender prints it), and `--record-after N --record-for M`, which press the phone's Record button: CONTROL `record_start` N seconds after connecting, `record_stop` M seconds later; the sender prints every `RECORDER_STATE` it receives. Unit tests: `python -m pytest -p no:cacheprovider` (no device needed; they use loopback ports only).

## Virtual camera

Zoom, Meet, Teams and other apps can use the glasses video as a webcam. The receiver composes each frame onto a **landscape** picture and sends it to a virtual camera device, on its own thread at a steady 30 fps (it repeats the last picture until the next frame arrives, because apps expect a constant rate). Recording and decoding never wait for it.

### Set up once

1. `~/.venvs/glassesdata-receiver/bin/pip install pyvirtualcam` (0.15 or later). Without it everything else works, and the page says it is missing.
2. **macOS:** install [OBS Studio](https://obsproject.com) (30 or later). Open it and click **Start Virtual Camera** once. macOS asks to allow the OBS camera extension: allow it in System Settings > General > Login Items & Extensions > Camera Extensions. Then click **Stop Virtual Camera** and quit OBS. macOS only lets a signed camera extension provide a system camera; OBS ships one, so the receiver does not need its own. OBS does not need to run afterwards.
3. **Linux (Ubuntu 24.04):** pyvirtualcam writes to a v4l2loopback device, `/dev/video10` labelled GlassesData: `scripts/setup_ubuntu.sh` sets it up, by hand see Ubuntu 24.04 > Virtual camera. The same canvas, modes and controls apply.
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
| `--virtual-camera-device /dev/videoN` | Linux: the v4l2loopback device to write to. Default: the one labelled GlassesData, else the first v4l2loopback device. Ignored (with a warning) on macOS |

If pyvirtualcam is missing, or the device cannot be opened (OBS not installed, extension not allowed, v4l2loopback not loaded, no permission, another program writing to it), the receiver keeps running; the log and the page say why and what to do.

**Two ways to fit a portrait video into a landscape picture:**

- **Fit** (default): the whole frame, at the full height of the picture, centred. The sides show a blurred, darkened, enlarged copy of the same frame (the usual treatment of portrait video on a landscape screen). Nothing of the frame is lost; it is small in a call.
- **Fill**: a landscape band of the frame fills the whole picture. **Band** moves it from the top (0) to the bottom (1) of the frame (default the middle): hands are often low in the glasses' view, so move the band down to keep them in the call.

A landscape source is fitted exactly (16:9) or with blurred bars, or cropped to cover in fill mode. With the **hand skeleton** on, it is drawn on the video part only, in the right place for the band, while hand tracking is on (`H`). Without video (no iPhone connected, no video yet, or no new frame for 3 s) the camera shows a calm card, "GlassesData · No video" and why, never a frozen frame or black.

**On the page:** the **Virtual camera** panel, under the signal chain, has the on/off switch with the device name, Fit / Fill, the band slider (fill only), the hand-skeleton switch and a live thumbnail of exactly what is being sent, so you can check the framing without opening Zoom. The camera button in the transport bar and the `V` key turn it on and off. The panel and the button follow the page's access rules (the same key, header and origin checks as Record).

### What the receiver can and cannot know

**Delivery cannot be confirmed by the sender.** The virtual camera API only takes frames; it never says whether an app receives them. And the OBS Virtual Camera on macOS only takes landscape 1920x1080 or 1280x720 frames: a frame of any other size (for example the glasses' own 720x1280) is dropped silently. The open succeeds, `send` raises nothing, and OBS only prints `Pixel buffer size mismatch` on standard error. This is why the receiver always sends a landscape canvas of exactly the configured size, never the video's own size.

At start the receiver logs a self-check: the device it opened, the backend, the size and the rate. On macOS it also watches its own standard error for 2 s for OBS's rejection message, and reports it in the log and on the page ("OBS rejects the frames") if it appears. A clean check means only that OBS did not complain. On Linux the check says that v4l2loopback accepted the format and took the frames of the first 2 s without an error; no app needs to be reading. To see the picture arrive, open the camera in an app (on Linux: `ffplay /dev/video10`, or the camera list of Zoom, Meet or OBS):

1. Start the receiver with `--virtual-camera` (or press `V` on the page).
2. Open QuickTime Player > File > New Movie Recording, and choose **OBS Virtual Camera** from the menu next to the record button.

## Ubuntu 24.04

**Not yet run on Ubuntu.** The Linux code paths were written and unit-tested on a Mac (`tests/test_linux.py` and `tests/test_paths.py` simulate `sys.platform`, sysfs, usbmuxd and pyvirtualcam), and the two scripts below were only checked with `bash -n` and read through. Nobody has run them, or the receiver, on Ubuntu with an iPhone attached. On the first run keep the output of `scripts/setup_ubuntu.sh` and `scripts/check_linux.sh`; expect small fixes.

Everything else in this README applies unchanged: the page, recordings, calibration, Wi-Fi pairing, the virtual camera's canvas and controls. `python` below means `~/.venvs/glassesdata-receiver/bin/python`, run from this `Receiver/` folder.

### Install

Put the repository on the laptop (a `git clone`, outside any synced folder), then run as yourself, not with sudo:

```sh
Receiver/scripts/setup_ubuntu.sh          # asks before every sudo step; --yes answers yes, --help lists the options
```

It is idempotent (run it again at any time; it only does what is missing) and prints every command before running it:

1. checks the system (Ubuntu 24.04, x86_64, Secure Boot state; it refuses to run as root);
2. installs with apt what is missing: `python3-venv python3-dev usbmuxd libimobiledevice-utils libgl1 libglib2.0-0t64`, and `ffmpeg` if you want it (`ffplay` to look at the virtual camera, `ffprobe` for recordings);
3. creates `~/.venvs/glassesdata-receiver` with Ubuntu's Python 3.12 and installs `requirements-dev.txt` and `pyvirtualcam`, pinned to the versions the tests passed with on macOS (`--latest` for the newest allowed). Every one of them has a manylinux x86_64 wheel usable by CPython 3.12 (checked on PyPI: av 18.1.0 and cryptography 50.0.2 as `cp311-abi3`, opencv-contrib-python 5.1.0.95 as `cp37-abi3`, mediapipe 1.2.3 as `py3-none-manylinux_2_28`, numpy 2.4.6, zeroconf 0.154.0 and pyvirtualcam 0.15.0 as `cp312`; qrcode, ifaddr and pytest are pure Python), so nothing is compiled; then it downloads the hand model;
4. optionally sets up the virtual camera (below);
5. checks the iPhone's pairing if one is plugged in, and offers `idevicepair pair`;
6. if `ufw` is on, offers to open the Wi-Fi ports;
7. runs `python -m glassesdata_receiver --help` and the test suite (`--no-tests` skips it). The tests have so far only run on Python 3.11 on macOS, so this is their first run on 3.12 and on Linux.

By hand, the same without the virtual camera:

```sh
sudo apt install python3-venv python3-dev usbmuxd libimobiledevice-utils
python3 -m venv ~/.venvs/glassesdata-receiver
~/.venvs/glassesdata-receiver/bin/pip install -r requirements-dev.txt pyvirtualcam
```

`scripts/check_linux.sh` is the read-only diagnosis (no sudo): Python packages, folders, usbmuxd and the socket, the iPhone on the bus and its pairing, recent usbmuxd crashes, the v4l2loopback module, its signer and devices, who has the camera device open, ports 8765 and 7900, the firewall. Each problem line says what to do; it exits with 1 when something fails.

The NVIDIA GPU is not used: PyAV decodes HEVC and MediaPipe's Python package runs the hand model on the CPU. One 720p stream at 24-30 fps is light for a laptop CPU; the page's signal chain shows whether decoding or tracking falls behind.

### USB

```sh
python -m glassesdata_receiver --usb --open
```

- **usbmuxd** comes from the `usbmuxd` package. udev starts it when an iPhone is plugged in and stops it when the last one is unplugged, so before that the page says to plug the iPhone in. Its socket is `/var/run/usbmuxd`, the same path as on macOS, and usbmuxd makes it world-writable (`chmod(..., 0666)` in its `main.c`), so no group membership is needed.
- **Trust, once.** Plug in the iPhone and unlock it. Linux usbmuxd asks the phone itself to show "Trust This Computer?"; tap Trust and enter the passcode. usbmuxd keeps the pairing record in `/var/lib/lockdown`. Check with `idevice_id -l` (lists the iPhone's UDID) and `idevicepair validate` ("SUCCESS: Validated pairing"). If the prompt was missed or dismissed: unlock the phone and run `idevicepair pair`, tap Trust, run it again.
- **Does the link need that pairing?** usbmuxd itself does not ask for it: its `Connect` only looks the device up and sends a TCP SYN to the app's port (`device_start_connect` in usbmuxd's `src/device.c`), and an iPhone that was never paired is listed anyway (`src/preflight.c`, "make device visible anyways"); libusbmuxd's `iproxy(1)` asks for pairing only for network connections. Whether iOS accepts a connection to an app's port from a computer it has never trusted is not documented by Apple and has not been tried here. So trust the computer once in any case. When the app's port refuses the connection and usbmuxd has no pairing record for the phone (asked over the socket with `ReadPairRecord`, no root needed), the page says "this computer is not trusted yet" instead of "open GlassesData".
- The page's USB line tells the cases apart: usbmuxd not installed; the iPhone on the USB bus but usbmuxd not running (`sudo systemctl restart usbmuxd`); no iPhone; not trusted yet; the app's Computer link off.

### Wi-Fi

As on macOS: `python -m glassesdata_receiver --wifi --pair` once, then `--wifi`. The identity (`cert.pem`, `key.pem`, `pairing.json`, mode 0600 in a 0700 folder) lives in `~/.local/state/glassesdata` (`$XDG_STATE_HOME`).

- **Bonjour.** The receiver advertises `_glassesdata._tcp` with its own responder (python-zeroconf). It binds UDP 5353 with address reuse, as Ubuntu's `avahi-daemon` does, so both run side by side; nothing needs to be stopped (expected, not yet seen on the laptop). The pairing code also carries the addresses, so the phone finds the computer without Bonjour too.
- **Firewall.** Ubuntu's `ufw` is off by default. If it is on (`sudo ufw status`): `sudo ufw allow 7900/tcp` (the link) and `sudo ufw allow 5353/udp` (Bonjour). With `--web-host 0.0.0.0` also the web port, best limited to your LAN: `sudo ufw allow from 192.168.1.0/24 to any port 8765 proto tcp`. USB mode and the page on `127.0.0.1` need no rule.
- **Networks that isolate clients** (eduroam and many campus or hotel networks) do not let the phone reach the laptop at all. Use USB, or a hotspot or home network.
- Viewing from another device: start with `--web-host 0.0.0.0` and open the printed `WEB_URL` (with its key) there.

### Virtual camera

pyvirtualcam writes to a v4l2loopback device: a video device that apps read like a webcam. The receiver's is `/dev/video10`, labelled **GlassesData**.

1. **The module.** Ubuntu's own kernels ship it, built and signed by Canonical, so Secure Boot accepts it as it is: for the 6.8, 6.11, 6.14 and 6.17 kernels of 24.04 it is `/lib/modules/<version>/kernel/v4l2loopback/v4l2loopback.ko.zst` in the kernel's `linux-modules-<version>` package; the newest 7.0 kernels moved it to `linux-main-modules-v4l2loopback-<version>` (`sudo apt install linux-main-modules-v4l2loopback-$(uname -r)`). `modinfo v4l2loopback` shows whether this kernel has it and who signed it. (Seen in Ubuntu's package index in November 2026.)
2. **Only for a kernel without it**: build it with DKMS, `sudo apt install v4l2loopback-dkms v4l2loopback-utils`. With Secure Boot on, a DKMS module is signed with this computer's own key (MOK), which the firmware must trust once, otherwise `modprobe` says `Key was rejected by service`. The installer may ask for a one-time password to enrol the key at the next reboot; if it did not, or that was skipped: `sudo mokutil --import /var/lib/shim-signed/mok/MOK.der` (choose a one-time password), reboot, and in the blue "Perform MOK management" screen choose Enroll MOK > Continue > Yes, type the password, Reboot. `mokutil --test-key /var/lib/shim-signed/mok/MOK.der` then says "is already enrolled".
3. **Load it:**

   ```sh
   sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="GlassesData" exclusive_caps=1
   ```

   `video_nr=10` keeps it clear of the built-in webcam's numbers, `card_label` is the name apps list, and `exclusive_caps=1` makes the device announce itself as a capture device once something writes to it, which Chrome (Meet) and Zoom need to list it.
4. **At every start** (what `setup_ubuntu.sh` writes): `/etc/modules-load.d/glassesdata-v4l2loopback.conf` containing `v4l2loopback`, and `/etc/modprobe.d/glassesdata-v4l2loopback.conf` containing `options v4l2loopback devices=1 video_nr=10 card_label=GlassesData exclusive_caps=1`.
5. **Start it**: `python -m glassesdata_receiver --usb --virtual-camera` (or `V` on the page), and choose **GlassesData** in Zoom, Meet or Teams; `ffplay /dev/video10` shows it too. `--virtual-camera-device /dev/videoN` picks another v4l2loopback device; without it the receiver takes the one labelled GlassesData, else the first one.

The page's Virtual camera panel shows the device it writes to (or would write to, with the v4l2loopback devices found as its tooltip) and, when there is none, the steps above. Pixel format: the receiver hands pyvirtualcam BGR frames; pyvirtualcam converts them with libyuv to I420 and sets the device to `V4L2_PIX_FMT_YUV420`, the format WebRTC (Chrome, Meet), Zoom, OBS and ffmpeg read. A desktop login may write to `/dev/video*` (systemd gives the logged-in user access); over SSH add yourself to the `video` group (`sudo usermod -aG video $USER`, log in again).

**OBS for Linux** can show the same device as a source (Sources > Video Capture Device (V4L2) > GlassesData). OBS's own Start Virtual Camera also writes to a v4l2loopback device (it loads the module itself when it is not loaded). Only one program can write to a device at a time, so to use both, give each its own: `sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback devices=2 video_nr=10,11 card_label=GlassesData,OBS exclusive_caps=1,1` (and the same `options` line in the modprobe.d file).

### Troubleshooting

Run `scripts/check_linux.sh` first; it covers most of this.

| Symptom | What to do |
|---|---|
| USB: "usbmuxd is not installed" | `sudo apt install usbmuxd libimobiledevice-utils` |
| USB: "plug in the iPhone" although it is plugged in | Another cable or port (a charge-only cable does not work); `lsusb` should list an Apple device; unlock the phone |
| USB: "the iPhone is plugged in but usbmuxd is not running" | `sudo systemctl restart usbmuxd`; `journalctl -u usbmuxd -n 30` says why it stopped |
| The iPhone connects and disconnects every few seconds, `journalctl -u usbmuxd` shows `status=11/SEGV` | A known crash of usbmuxd 1.1.1 on Ubuntu 24.04 with newer iPhones (libimobiledevice/usbmuxd issue #246, open); a USB-A to USB-C cable was reported to avoid it |
| USB: "this computer is not trusted yet" | Unlock the phone, tap Trust; or `idevicepair pair`. If the phone never asks: Settings > General > Transfer or Reset iPhone > Reset > Reset Location & Privacy, then plug in again |
| USB: "iPhone found - open GlassesData and turn on Computer link (USB)" | The phone is trusted but the app is not listening: open it and turn the link on |
| Virtual camera: "the v4l2loopback module is not loaded" | `sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=GlassesData exclusive_caps=1`, or run `setup_ubuntu.sh` |
| `modprobe: ERROR: could not insert 'v4l2loopback': Key was rejected by service` | A DKMS module under Secure Boot: enrol its key (Virtual camera, step 2), or use Ubuntu's signed module |
| Virtual camera: "missing permissions" | `sudo usermod -aG video $USER`, log out and in |
| Virtual camera: "already in use", "busy" | Another program writes to the device (OBS's virtual camera): stop it, or `devices=2` as above |
| Zoom or Meet does not list GlassesData | The module needs `exclusive_caps=1`; reload it with that, then restart the app (apps list cameras when they start) |
| `--open` does nothing | No graphical session (SSH), or run as root: open the printed `WEB_URL` yourself |
| `import cv2` fails with `libGL.so.1` or `libgthread` | `sudo apt install libgl1 libglib2.0-0t64` (a minimal install) |
| Recordings are not in `~/Videos` | They follow the desktop's Videos folder (its name in the session's language); `--out DIR` sets another one |
| A long recording stops when the laptop goes to sleep | Ubuntu suspends an idle laptop on battery after 15 minutes. Plug it in, or start the receiver under `systemd-inhibit --what=sleep:idle --why="GlassesData" ~/.venvs/glassesdata-receiver/bin/python -m glassesdata_receiver ...` |
| Closing the terminal | The receiver stops cleanly on SIGHUP (the recording is closed), as on Ctrl-C. To keep it running, start it under `nohup` or in `tmux` |
