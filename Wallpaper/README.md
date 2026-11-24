# Pixel Battle: live wallpaper

The pixel battle from Site 2 as a live wallpaper for Windows, standalone. Run **PixelBattle.exe** and the battle plays on the desktop behind your icons: the endless eight-player fight through random countries, full screen. Nothing else needs installing.

The defaults:

- **2× speed**
- **zoomed out to 75%** (a third more of the battlefield on the desktop)
- **30 frames a second**
- a lighter enemy and particle budget than the site

## Using it

The finished wallpaper is `release/PixelBattle/` (also zipped as `release/PixelBattle.zip`). Copy that folder anywhere, keep its files together, and run `PixelBattle.exe`.

- **The tray icon** (the ^ by the clock) has its menu:
  - Pause
  - Speed (1×, **2×**, 4×)
  - Zoom (100%, **75%**, 50%)
  - Frame rate (20, **30**, 60)
  - Screens (**main**, or every screen)
  - Clicks
  - Sleep on battery
  - Start with Windows
  - New battle
  - **Exit**, which puts your own wallpaper back
- **Click the desktop** to call in a blast there. **Hold** the button for a barrage.
- **From a script or shortcut:**
  - `PixelBattle.exe --exit` closes the running one.
  - `PixelBattle.exe --selftest` runs it for a few seconds and writes `selftest.log`, showing where it attached and what the page reported.
- **Settings** are kept in `%LOCALAPPDATA%\PixelBattle\settings.txt`.

It runs only on what comes with Windows 10 and 11:

- **the .NET Framework 4.8**, which the app runs on;
- **WebView2**, the Edge engine, which shows the battle's page (`battle/index.html`).

The Microsoft WebView2 DLLs that talk to it are in the folder.

It handles both desktop layouts: the classic one (up to Windows 11 23H2) and the "raised" one from 24H2 on. It also comes back by itself if Explorer restarts or the screens change.

## Going easy on the machine

Measured on this PC (2880×1800 screen, 16-thread CPU):

| State | CPU | Memory |
|---|---|---|
| **Asleep** | about 0.2% of one core | about 330 MB |
| **On show** | about 38% of one core (about 2.4% of the whole CPU), at 30 frames a second | about 540 MB |

It's asleep whenever a maximised or full-screen window covers its screen, while the PC is locked, and on battery (if ticked). While asleep its page is suspended and it draws nothing.

Memory is across the WebView2 engine's processes: the browser, the GPU process and the page.

What keeps it light:

- **Frames only when due:** it asks for a frame only when one's due, not for every one a 120 Hz display could show.
- **Software drawing:** it draws in software, so the GPU gets one finished picture a frame, not thousands of little pieces.
- **Lighter budget:** fewer enemies and particles than the site.
- **Lower frame rates:** 20 fps in the tray menu lightens it further.

## Rebuilding

The battle's code lives in `../Site 2/src`. The wallpaper is built straight from it, so a change to the battle in Site 2 comes through on the next build.

```sh
npm install
npm run build     # the page (dist/index.html), then release/PixelBattle/ and its zip
npm run measure   # how hard the page works in Chrome (npm run measure -- 2560 1440 30 for another screen)
npm run dev       # the page in a browser tab, live-reloading
```

What the build does:

- **The page:** `vite build` builds it, and `scripts/pack.mjs` inlines it into one file and checks it runs.
- **The app:** `scripts/build-exe.mjs` compiles `host/PixelBattle.cs` with the C# compiler that comes with Windows (.NET Framework 4). So building needs nothing installed beyond Node either.
- **WebView2 SDK:** Microsoft's SDK is downloaded from nuget.org on the first build and kept in `host/.webview2`.

The page's defaults are at the top of `src/main.tsx` and in `src/shims/device.ts`. The app's are in the `Settings` class in `host/PixelBattle.cs`.
