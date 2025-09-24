# Mood-Bubble-Demo
A tiny, testable plugin that turns text into a **tone** with a **color/animation bubble**.
It also lets you insert a tagged plain-text version (e.g., `[HAPPY] your text`) into the active input on any page.

## Features
- Popup with message box + relationship selector
- Heuristic tone engine (happy/angry/uncertain/calm)
- Animated preview bubble (pulse/flash/ripple/gradient)
- Insert tagged text into focused input on a page

> Privacy: runs locally in the popup — no network calls.

## Install (Chrome, Edge)
1. Download and unzip this folder.
2. Open `chrome://extensions` → enable **Developer mode** (top right).
3. Click **Load unpacked** → select the folder.
4. Pin the extension and open it.

## Quick Test
Try messages like:
- `Let's go!! this is awesome` → happy
- `WHERE WERE YOU???` → uncertain/angry (caps + questions)
- `Not sure about this... maybe later` → uncertain
- `On my way.` → calm

Switch **Relationship** to see the seriousness bias.

## Insert into input
- Click **Insert into page`, then paste is attempted into the currently-focused input or contentEditable.
  (This is a minimal demo; many sites work, some may block or need extra integration.)

## Files
- `manifest.json` — MV3 manifest
- `popup.html` / `popup.js` / `styles.css` — UI and tiny rules engine
- `content.js` — tries to insert text into the focused field
- `sw.js` — placeholder service worker
