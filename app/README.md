# EgoUploader

Expo React Native app for egocentric video capture. Mount your phone on your head, record ultrawide 1080p30 video clips using hand gestures, and upload them to the annotation server for SLAM, hand pose, and speech transcription processing into LeRobot v3.0 datasets.

## Features

- **Ultrawide 1080p30 recording** (non-HDR) via `react-native-vision-camera`
- **Hand gesture control** — show an open hand (all 5 fingers extended) to start/stop recording
- **Voice announcements** — "Recording" / "Stopped" via text-to-speech (useful when phone is head-mounted)
- **Auto-upload** to the annotation server (`visualizer.py` in upload mode)
- **Native hand detection** using Apple Vision framework (`VNDetectHumanHandPoseRequest`) running as a VisionCamera frame processor plugin at 3fps

## Prerequisites

- macOS with Xcode 26+ installed
- Node.js 18+
- An iOS device (hand detection + ultrawide camera require a physical device)
- CocoaPods (`gem install cocoapods` or comes with Xcode)
- `libimobiledevice` for USB port forwarding (`brew install libimobiledevice`)

## Dev Environment Setup

### 1. Install dependencies

```bash
cd app
npm install
```

### 2. Prebuild native projects

This generates the `ios/` and `android/` directories with native code, including the hand detection frame processor plugin:

```bash
npx expo prebuild --clean
```

### 3. Code signing (first time only)

1. Open `ios/EgoUploader.xcworkspace` in Xcode
2. Select the **EgoUploader** target > **Signing & Capabilities**
3. Check "Automatically manage signing"
4. Select your Team (free Apple ID works)
5. On your iPhone: **Settings > General > VPN & Device Management** — trust the developer certificate

### 4. Enable Developer Mode on iPhone (first time only)

**Settings > Privacy & Security > Developer Mode** — toggle on, restart when prompted.

### 5. Build and install

Connect your iPhone via USB, then:

```bash
# Find your device ID
xcrun devicectl list devices

# Build and install
npx expo run:ios --device <DEVICE_ID>
```

### 6. Connect to Metro dev server over USB

Corporate/public Wi-Fi networks often block device-to-device traffic. Use USB port forwarding instead:

**Terminal 1** — start Metro:
```bash
npx expo start --dev-client --host lan
```

**Terminal 2** — forward port over USB:
```bash
brew install libimobiledevice  # first time only
iproxy 8081 8081 -u <DEVICE_ID>
```

If `iproxy` shows "Address already in use" that's fine — it still creates the tunnel.

On your Mac, find the USB network interface IP:
```bash
ifconfig | grep -A2 "en10\|en[2-9]" | grep "inet "
```

Look for a `169.254.x.x` address — that's the USB interface. In the Expo dev client on your phone, enter `http://169.254.x.x:8081` as the bundler URL.

Alternatively, use your iPhone's Personal Hotspot and connect your Mac to it, then `--host lan` works directly.

## Usage

1. Open the app — Camera tab shows the ultrawide camera preview
2. Configure the server URL in **Settings** (gear icon or Clips tab > settings)
3. Mount the phone on your head
4. **Start recording**: hold up an open hand (all 5 fingers spread) for ~0.75 seconds
5. Voice announces "Recording"
6. **Stop recording**: same gesture again (3-second cooldown between toggles)
7. Voice announces "Stopped"
8. Clip auto-uploads to the annotation server
9. Fallback: tap the record button on screen

## Architecture

```
app/
├── app/                    # Expo Router screens
│   ├── _layout.tsx         # Root stack (tabs + settings modal)
│   ├── settings.tsx        # Server URL configuration
│   └── (tabs)/
│       ├── _layout.tsx     # Camera + Clips tabs
│       ├── index.tsx       # Camera screen with gesture detection
│       └── clips.tsx       # Recorded clips list with upload
├── lib/
│   ├── handGesture.ts      # VisionCamera frame processor plugin wrapper
│   ├── gesture.ts          # Open-hand gesture recognition from landmarks
│   ├── upload.ts           # HTTP upload to annotation server
│   └── storage.ts          # Clip metadata + settings persistence
├── plugins/
│   └── withHandDetection.js  # Expo config plugin — injects native iOS code
├── app.json                # Expo config with camera permissions + plugins
└── package.json
```

The native hand detection plugin (`plugins/withHandDetection.js`) injects Swift + ObjC files into the iOS build during `expo prebuild`. The Swift code uses Apple's Vision framework to detect 21 hand landmarks per hand at 3fps, which are then analyzed in JS for the open-hand gesture.
