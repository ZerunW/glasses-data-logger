# Security and privacy review

Review date: 2026-11-21. Scope: the GDL1 link between the iPhone app and the receiver (USB and
Wi-Fi), Wi-Fi pairing, the receiver's web page, files on disk on both sides, and network access
at runtime. Method: reading the source (`CameraAccess/Streaming/*`, `CameraAccess/Recording/*`,
`CameraAccess/ViewModels/CameraViewModel.swift`, `Receiver/glassesdata_receiver/{wifi,web,session,link,usbmux,protocol,paths,hands,bodyprofile,calibfile,recorder}.py`,
`Receiver/glassesdata_receiver/page.html`) and checking file modes on the development Mac. This is
not a penetration test. The link code ran end to end only between processes on the development
Mac (the app's own link sources compiled for macOS against the receiver, and the app on the iOS
simulator); nothing below has been tried with a real iPhone, real glasses or the Ubuntu laptop.

The earlier review (TetherCam camera extension, IUCM stream, PSK pairing, Mac relay app) no longer
applies: all of that code has been removed.

## System in one paragraph

The iPhone app receives HEVC video, IMU, input events and device state from the Ray-Ban Meta
glasses through Meta's Wearables DAT SDK and forwards them to a computer over GDL1 (length-prefixed
binary messages, `Receiver/glassesdata_receiver/protocol.py`, `CameraAccess/Streaming/LinkProtocol.swift`).
Over **USB** the app listens on `127.0.0.1:7900` on the phone and the receiver connects to it through
usbmuxd; there is no authentication and no encryption. Over **Wi-Fi** the app connects out to the
receiver with TLS 1.3, pins the receiver's certificate fingerprint and presents a token; both come
from a QR code (`gdpair://v1?...`) the receiver shows once. The receiver records to disk and serves
a web page on `127.0.0.1:8765`. The offline pipeline reads recordings from disk and has no network
listener.

## Assets

| Asset | Where | Why it matters |
|---|---|---|
| Live video and recordings | link, receiver memory, `video.mp4` / phone `video.mov`, pipeline `derived/` (masks, meshes, replays) | First-person video shows everything the wearer looks at: other people, rooms, screens, documents |
| Phone-side audio | phone `video.mov` only, when the glasses microphone (HFP) is enabled | Conversations. The link carries no audio; the receiver never records any |
| Sensor and state data | link, `motion.csv`, `inputs.csv`, `device_state.csv` | Head motion, wear state, battery; low sensitivity on its own |
| Body measurements | receiver `profile.json` (state dir and every recording) | Personal body dimensions |
| Device names | GDL1 HELLO / HELLO_ACK, `info.json`, Bonjour | Identify the phone and the computer |
| Receiver identity | `cert.pem`, `key.pem`, `pairing.json` (token) in the state dir | The key lets someone impersonate the receiver; the token lets someone connect as the phone |
| Phone pairing | iPhone Keychain | The pinned fingerprint and token of the paired computer |
| Web access key | receiver memory, `WEB_URL`, browser cookie | Full control of the web page when it listens beyond loopback |

## Attackers considered

1. Someone on the same Wi-Fi network (passive or active).
2. Another process or user account on the receiving computer.
3. Another app on the iPhone.
4. A web site open in the user's browser (cross-site requests, DNS rebinding).
5. Someone who sees or photographs the pairing QR code, or the `WEB_URL` with its key.
6. A computer other than the user's that the phone is plugged into.
7. Someone with access to backups, synced folders or a lost device.

Out of scope: a compromised operating system, the Meta SDK and Meta AI app internals (binary
dependencies, not audited), and a malicious receiver run by the user.

## USB link

**What the code does.**

- `ComputerLink` (iOS) starts an `NWListener` bound to `127.0.0.1:7900` whenever the link mode is
  USB. The mode is persisted and **USB is the default**; the listener starts when the app launches,
  independently of the glasses session, and stays up while the app runs (the app declares the
  `audio`, `bluetooth-*`, `external-accessory` and `processing` background modes, so it may keep
  running in the background). It accepts one connection at a time.
- Whoever connects gets the phone's HELLO (app version, device model, device name, no token over
  USB). Once that peer answers with a `HELLO_ACK` whose `ok` is true, the phone streams video,
  IMU, inputs, device state and STATS, and accepts `RECORDER_STATE` messages, which drive the
  Record button and status line. The phone does not authenticate the peer.
- The receiver (`link.py`, `usbmux.py`) talks to usbmuxd over its Unix socket `/var/run/usbmuxd`,
  asks it for the first USB device and connects to the app's port. It does not authenticate the
  phone either: whatever listens on that port is taken to be GlassesData.
- The usbmuxd socket is world-writable: `srw-rw-rw- root daemon` on the development Mac (checked),
  and created with mode 0666 by Linux usbmuxd (from its source, as cited in `Receiver/README.md`;
  not checked on the Omen).

**The trade-off.** The cable and the iPhone's "Trust This Computer" pairing were chosen as the
trust boundary, so USB needs no pairing step and has no encryption. The consequence, precisely:

- **Any process on the computer, under any user account,** that can open `/var/run/usbmuxd` can,
  while the iPhone is attached by cable and the app is listening, connect to port 7900 on the phone
  and receive the complete live stream (video, IMU, inputs, device state, the phone's name). It can
  also send `RECORDER_STATE` messages that the phone shows as the computer's recording state.
  Because the phone accepts one connection at a time, a process that connects first also keeps
  the real receiver out until it disconnects.
- **Not reachable from the network.** usbmuxd is a local Unix socket and the phone's listener is
  bound to the loopback address, so nothing on the Wi-Fi can reach the USB path.
- **Other apps on the iPhone.** iOS does not document whether one app can connect to another app's
  listener on `127.0.0.1`. If it can, an app running at the same time could receive the stream the
  same way. Not verified yet: needs a second app on a real iPhone.
- **Other computers.** Whether iOS lets a computer the phone has never trusted reach an app's port
  through usbmuxd is not documented and has not been tried (`Receiver/README.md`, Ubuntu > USB).
  Until it is, assume any computer the phone is plugged into while the app listens can receive the
  stream.
- **Spoofing the phone.** Another app on the phone that listens on 7900 while GlassesData does not
  could feed the receiver invented data. Low impact (it ends up in a recording).

**How to limit it today.**

- Set the Computer card's link to **Off** when you are not streaming; the setting persists across
  launches.
- Plug the phone only into your own computer. On a computer shared with other people or running
  untrusted software, use Wi-Fi mode (token plus TLS) instead of USB.
- On Linux the socket's mode is a usbmuxd property; restricting it to a group is possible but is a
  system change that affects every tool that uses usbmuxd (not done or tested here).
- A real fix is an authenticated USB link (see Open issues).

## Wi-Fi link

**Receiver** (`wifi.py`, `session.py`):

- `--wifi` listens on `0.0.0.0:7900` (all interfaces; `--port` changes it). TLS only:
  `ssl.SSLContext(PROTOCOL_TLS_SERVER)` with `minimum_version = TLSv1_3`.
- Identity: a self-signed ECDSA P-256 certificate (valid 10 years), its private key and a token
  from `secrets.token_urlsafe(32)` (256 bits). `key.pem` and `pairing.json` are written with mode
  0600 (`os.open` + `fchmod`, which also tightens an existing file) in a state directory set to
  0700. `--reset-pairing` creates a new certificate and token, which unpairs every phone.
- The TLS handshake must finish within 5 s and HELLO must arrive within 5 s. The token in HELLO
  (sent inside TLS) is compared with `hmac.compare_digest`; a wrong token gets `HELLO_ACK`
  `{"ok": false, "code": "pairing"}` and the connection is closed. Only after the token check
  does the session claim the receiver; a second phone gets `code: "busy"`.
- Bonjour: advertises `_glassesdata._tcp` with the computer's name, the certificate fingerprint and
  its IPv4 addresses (`--no-bonjour` turns it off).

**Phone** (`LinkTLS.swift`, `ComputerLink.swift`, `LinkPairingStore.swift`):

- `NWConnection` with TLS, minimum TLS 1.3. The verify block accepts the server only when the
  SHA-256 of the leaf certificate's DER encoding equals the pinned fingerprint. There is no system
  trust evaluation and no host-name check, which is intended for a self-signed certificate.
- Discovery: Bonjour results whose TXT `fp` matches the pinned fingerprint, else the addresses in
  the pairing, in order. The TXT record is only a hint; the TLS pin is what authenticates.
- The pairing (name, hosts, port, fingerprint, token) is a generic-password Keychain item, not
  synchronised, `kSecAttrAccessibleWhenUnlockedThisDeviceOnly`. One paired computer;
  **Forget this computer** deletes it.

**What this gives.** Against someone on the network: confidentiality and integrity of the stream
(TLS 1.3 with ephemeral key exchange, so a later leak of `key.pem` does not decrypt recorded
traffic); the phone only sends to the paired receiver; the receiver only accepts a phone that has
the token.

**What it does not give.**

- The port is open to the whole network. Anyone can complete a TLS handshake; without the token
  they are refused after HELLO. There is no limit on concurrent connections: each one gets a
  thread for up to about 10 s (handshake plus HELLO timeouts), so a host on the network can tie
  up threads (denial of service).
- Bonjour announces the computer's name and the fact that a GlassesData receiver runs on it.
- Someone who has the token and can reach the port can connect as the phone while no phone is
  connected: inject invented video and sensor data, and start or stop recordings with `CONTROL`
  (filling the disk). They cannot receive the real phone's data, because the phone only connects
  to the pinned certificate.
- Someone who can read `key.pem` can impersonate the receiver: by advertising the same fingerprint
  over Bonjour, or answering on one of the paired addresses, they receive the phone's stream. The
  state directory is the thing to protect.
- Networks that isolate clients (eduroam, many hotel networks) block the link entirely; that is an
  availability limit, not a weakness.

## Pairing QR code

- Contents: `gdpair://v1?name=&hosts=&port=&fp=&token=`. It is a credential: treat it like a
  password. The terminal says so when `--pair` prints it.
- Where it appears: the web page (`/api/pairing.png`, under the page's access rules below), the
  terminal with `--pair` (ASCII QR) and stdout with `--print-pairing` (`PAIRING_URI=...`, which
  then sits in terminal scrollback or any log the output is redirected to).
- Phone side: the code only enters through the in-app scanner (`DataScannerViewController`);
  `gdpair` is not a registered URL scheme, so links in other apps cannot inject a pairing.
  `LinkPairing(uri:)` validates strictly: scheme `gdpair`, host `v1`, no repeated keys, 1-8 hosts
  of at most 64 characters without whitespace or control characters, a numeric port, a 64-digit hex
  fingerprint, a 16-128 character token of `[A-Za-z0-9_-]`, a name of at most 64 characters.
- A valid code is saved at once, **replaces the existing pairing without asking** and switches the
  link to Wi-Fi. Someone who gets the user to scan their code while the scanner is open would
  receive the stream from then on. See Open issues.

## Receiver web page

Facts from `web.py`:

- Default bind `127.0.0.1:8765`. On a loopback bind there is **no key**, but every request's `Host`
  header must name a loopback address (`127.0.0.0/8`, `::1`, `localhost`); this blocks DNS
  rebinding, where a web site points its own name at 127.0.0.1 to read the status or the pairing
  code.
- Any other bind (`--web-host 0.0.0.0`, a LAN address) logs a warning and creates a random key per
  start (`secrets.token_urlsafe(24)`). `WEB_URL` carries `?key=...`; the first request with it sets
  an `HttpOnly; SameSite=Strict` cookie named `gd_key_<port>` and the page redirects to drop the key
  from the address bar. Every route needs the key (page, status, history, both MJPEG streams,
  recordings list, pairing code, calibration, profile, all actions). Comparisons use
  `hmac.compare_digest`. The query string is never logged.
- Every POST needs the header `X-GlassesData: 1` (a cross-site page cannot send it without a CORS
  preflight, which the server never answers), an `Origin`, when present, equal to `http://<Host>`,
  and `Sec-Fetch-Site`, when present, of `same-origin` or `none`. This blocks cross-site request
  forgery against Record, the virtual camera, calibration and body measurements. POST bodies over
  64 KiB are not read. The calibration board request is bounded (at most 25 device pixels per mm,
  16384 pixels a side, 40 million pixels in all), so a request cannot make the receiver allocate
  an arbitrarily large image. An unexpected error answers a JSON 500 and is logged.
- Headers on every response: `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`,
  `X-Frame-Options: DENY`, `Cache-Control: no-store`. The page has a CSP: `default-src 'none'`,
  inline script and style only, images from itself and `data:`, `connect-src 'self'`,
  `form-action 'none'`, `frame-ancestors 'none'`.
- The page writes values that come from the phone or the receiver with `textContent`; its two
  `innerHTML` uses insert constant strings of the page itself.
- The recordings list shows names and sizes; the server never serves recording files.

Limits:

- With the default loopback bind, any process or user account on the computer can watch the
  stream, read the pairing QR code (and so the token), start and stop recordings, and change the
  calibration and body measurements. This is consistent with the USB trust model above.
- Beyond loopback the traffic is plain HTTP: the key, the cookie and the video cross the network
  unencrypted. On a network you do not trust, keep the loopback bind and use an SSH tunnel
  (`ssh -L 8765:127.0.0.1:8765 host`), as `Receiver/README.md` says.
- The key is printed on stdout and is in the browser history of the first visit.

## Files on disk

**Receiver.**

- State directory: `local/state` in the repository (macOS and Linux; the project keeps nothing
  outside its folder). If the repository sits in a synced folder such as OneDrive, the Wi-Fi
  identity's private key and pairing token are uploaded with it. It is set to 0700
  only when the Wi-Fi identity is loaded (`--wifi` or `--reset-pairing`); a receiver that never
  ran with `--wifi` leaves the directory at the umask default. Files in it: the identity (above),
  `calibration/calibration_<UTC>.json`, `calibration/current.json`, `calibration/screen.json` and
  `profile.json`, all written through `tempfile.mkstemp` (mode 0600) and an atomic rename.
- Recordings: `local/recordings` in the repository (synced along with it when the repository is
  in a synced folder). `calibration.json`,
  `profile.json` and `info.json` are written through `tempfile.mkstemp` and an atomic rename, so
  they are 0600; `video.mp4`, the CSVs and the folders follow the umask and the parent folders'
  permissions. Nothing is encrypted by the receiver; rely on FileVault or LUKS full-disk
  encryption.
- Each recording holds the video, the sensor CSVs, `hands.csv`, `info.json` (including the
  phone's name), `calibration.json` and `profile.json` (body measurements). The last two are
  written from validated content, not copied byte for byte: a calibration that fails the check in
  `calibfile.py` (shape and finiteness of K, positive focal lengths, sizes, distortion length, a
  known model) is left out with a note in `info.json`, and an implausible stored measurement falls
  back to its average. The pipeline adds `derived/` inside the recording: object masks, meshes, a
  replay video and figures made from the same video, with the same sensitivity.
- `--out` pointing into a synced folder (OneDrive, iCloud Drive, Dropbox) uploads the recordings.
  This repository lives in OneDrive; `.gitignore` excludes `Recordings/`, and recordings should
  stay outside the repository.
- The virtual camera makes the live video available to any app that opens that camera: on macOS
  every app with camera permission sees "OBS Virtual Camera"; on Linux `/dev/video10` is readable
  by the logged-in user and the `video` group.

**iPhone.**

- **Save on this iPhone** is off by default. When on, recordings (`video.mov` with optional HFP
  audio, CSVs, `info.json`) go to `Documents/Recordings`, which is visible in Files
  (On My iPhone > GlassesData) and in Finder file sharing (`UIFileSharingEnabled`,
  `LSSupportsOpeningDocumentsInPlace`) and is included in device backups (nothing is excluded).
  The app sets no file protection class of its own, so iOS's default applies.
- A folder picked in Files is remembered as a bookmark in `UserDefaults`
  (`recordingFolderBookmark`); it lets the app open that folder and nothing else. Recordings
  bound for it are written to the app's temporary folder (`tmp/StagedRecordings`) and moved into
  it with file coordination when they end; if the move fails they go to `Documents/Recordings`,
  and leftovers from a crash are moved there at the next launch. A folder in iCloud Drive uploads
  the recordings to iCloud; a USB drive is unencrypted removable media. Neither has been tried on
  a device (see `Docs/VALIDATION.md`).
- The Wi-Fi pairing is in the Keychain (above). Link mode, resolution and the save setting are in
  `UserDefaults`.

## Meta SDK and third parties

- The app uses Meta's Wearables DAT SDK 1.0.1 (binary XCFrameworks, pinned by revision) and needs
  the Meta AI app for registration. It sets `MWDAT.CrashReporting.OptOut = true`, which only opts
  out of the SDK's crash reporting; registration, permissions and device connection still involve
  Meta's software and may exchange data. `MWDAT.MetaAppID` is `"0"` (Developer Mode). Not audited.
- **No component calls ChatGPT, OpenAI or any other LLM or cloud API at runtime.** ChatGPT was
  used during development to draft some code (for example `Pipeline/glassesdata_pipeline/core/hull.py`
  says so); nothing of it runs as a service.
- Network access at runtime on the computer, apart from the GDL1 link and Bonjour, is downloads
  only:
  - receiver and pipeline `hands2d`: the MediaPipe hand model from `storage.googleapis.com`
    (HTTPS, the `latest` path, no checksum), once, into the cache folder;
  - pipeline heavy backends on the Linux machine: Depth Anything V2 through Hugging Face
    `from_pretrained` (model name, no pinned revision), SAM 2 `from_pretrained`, HaMeR's detector
    weights, and the WiLoR / Video Depth Anything / MANO files that are downloaded by hand.
    `Pipeline/scripts/setup_ubuntu.sh --models` fetches the Depth Anything V2 and SAM 2.1 weights
    and the MediaPipe model ahead of time (about 2.8 GB into the repository's `local/cache`). Most checkpoints are
    pickles (`.pt`, `.ckpt`, `.pkl`), which can run code when loaded: download them only from the
    official sources;
  - installation (not runtime): `Pipeline/scripts/setup_ubuntu.sh` installs PyPI packages pinned in
    `requirements.txt`, PyTorch from the PyTorch wheel index, SAM 2 from its GitHub repository
    (default branch, not pinned) and clones the MuJoCo Menagerie; every sudo step asks first;
  - the receiver's web page loads nothing from the internet.
- The receiver decodes HEVC from the phone with FFmpeg (PyAV) and OpenCV; a malicious sender on
  an accepted link could target those decoders. GDL1 frames are capped at 4 MiB on both sides.

## Open issues

1. **USB is unauthenticated.** Any local process can receive the stream through usbmuxd; the
   behaviour with other iPhone apps and with untrusted computers is unknown. Possible fix: carry
   the Wi-Fi token over USB too (pair once, then the phone checks the receiver's proof before
   streaming), or run the same pinned TLS over the usbmuxd connection.
2. **Scanning a pairing code replaces the pairing without confirmation.** Show the computer's
   name and fingerprint prefix and ask before saving.
3. **Loopback web page without a key.** Every local user and process can watch and control the
   receiver and read the Wi-Fi token. Possible fix: require the key on loopback too (the cookie
   flow already exists).
4. **Plain HTTP beyond loopback.** Use an SSH tunnel; TLS for the page is not implemented.
5. **No connection limit on the Wi-Fi listener** (threads per connection, no rate limit).
6. **Downloads without integrity pins** (MediaPipe `latest`, Hugging Face models without a
   revision, SAM 2 and the Menagerie from their default branches).
7. **Files at rest.** Recordings and `derived/` are unencrypted; the video, CSVs and `derived/`
   follow the umask; the state directory is 0700 only after `--wifi`. Phone recordings are in backups and in Files/Finder.
8. **Credentials in output.** `--pair`, `--print-pairing` and `WEB_URL` print secrets to the
   terminal.
9. **Certificate lifetime.** The receiver certificate is valid for 10 years and is only replaced
   by `--reset-pairing`.
10. **Not verified on hardware yet.** USB trust behaviour (other iPhone apps, untrusted
    computers), Bonjour discovery, the link on a real network, Files-picker folders in iCloud Drive
    and on USB drives. On the Mac, `Receiver/tools/swift_sender` (the app's own `ComputerLink` and
    `LinkTLS` compiled for macOS) streamed to the receiver over TCP and pinned TLS on 2026-11-20,
    and connections with a wrong pinned fingerprint (refused by the app's TLS check) and with a
    wrong token (refused by the receiver) did not get through.
