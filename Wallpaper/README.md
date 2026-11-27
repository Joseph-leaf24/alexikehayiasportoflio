# Pixel Battle: live wallpaper

The pixel battle from Site 2 as your Windows desktop's live wallpaper. It's the endless eight-player fight through random countries, behind your icons.

## Run it

**Double-click `PixelBattle.exe`.** That's all: it's one file, with nothing to install and no other app needed.

- The battle replaces your wallpaper straight away, behind the desktop icons, and a notification says so.
- **If your windows cover the desktop**, the battle sleeps behind them. That means one maximised window, or several between them (two snapped side by side, say). Press **Win+D**, or minimise them, to see it.
- **Its icon by the clock** (under the **^** if it's hidden there) says what it's doing when you point at it ("playing", "asleep (windows cover it)" and so on). **Click it**, with either button, for the menu:

  | Option | Choices |
  |---|---|
  | Pause | your own wallpaper shows until you resume |
  | Speed | 1×, **2×**, 4× |
  | Zoom | 100%, **75%**, 50% |
  | Frame rate | 20, **30**, 60 |
  | Screens | **main screen**, or every screen |
  | Clicking the desktop calls in a blast | on or off |
  | Sleep on battery | on or **off**: tick it to save a laptop's battery |
  | **Start with Windows** | tick it to keep the battle as your wallpaper after a restart |
  | New battle | |
  | **Exit** | puts your own wallpaper back |

- **Click the desktop** to call in a blast there. **Hold** the button for a barrage.
- **Running it again** while it's already your wallpaper just reminds you that it is.

It runs on what Windows 10 and 11 come with:

- **the .NET Framework 4.8**, which the app runs on;
- **WebView2** (the Edge engine), which draws the battle.

Everything else it needs is inside the exe: the battle itself, and Microsoft's WebView2 DLLs. It unpacks these into `%LOCALAPPDATA%\PixelBattle` the first time it runs, and keeps its settings there too.

It handles both desktop layouts: the classic one, up to Windows 11 23H2, and the "raised" one from 24H2 on. It also comes back by itself if Explorer restarts or the screens change.

**To remove it:**

1. Untick *Start with Windows*.
2. Click *Exit*.
3. Delete `PixelBattle.exe` and the `%LOCALAPPDATA%\PixelBattle` folder.

**From a shortcut or script:**

- `PixelBattle.exe --exit` closes the running one.
- `PixelBattle.exe --selftest` runs it for a few seconds, sleeps and wakes it (changing the settings while it sleeps, without saving them), and writes what happened to `%LOCALAPPDATA%\PixelBattle\selftest.log`.
- `PixelBattle.exe --log` runs it as usual, and writes when it slept and woke to `log.txt` there when it closes.

## Going easy on the machine

Measured on this PC (2880×1800 screen, 16-thread CPU):

| State | CPU | Memory |
|---|---|---|
| **Asleep** (desktop covered) | about 0.3% of one core | about 70 MB if it started covered, about 330 MB once the battle has run |
| **On show**, 30 frames a second | about 30–55% of one core (2–3% of the whole CPU); the busier the battle, the higher | about 600 MB |

It sleeps whenever its screen is covered by windows, while the PC is locked or it's paused, and on battery (if ticked). While it sleeps, the battle's page is suspended and draws nothing, and your own wallpaper shows. Settings changed in the menu meanwhile take effect when it wakes. The once-a-second check costs next to nothing. Memory is across the WebView2 engine's processes: the browser, the GPU process and the page.

What keeps it light while it's on show:

- **Frames only when due:** it draws only the frames the rate calls for, not every one a 120 Hz display could show.
- **Software drawing:** the GPU gets one finished picture each frame, not thousands of little pieces.
- **A lighter budget:** fewer enemies and particles than the site.
- **20 fps in the menu** lightens it further.

## Rebuilding

The exe is built from `source/`. The battle's own code lives in `../Site 2/src`, so a change to the battle in Site 2 comes through on the next build.

```sh
cd source
npm install
npm run build     # builds the page, then ../PixelBattle.exe (a running one is asked to close first)
npm run measure   # how hard the page works in Chrome (npm run measure -- 2560 1440 30 for another screen)
npm run dev       # the page in a browser tab, live-reloading
```

What the build does:

- **The page:** `vite build` builds it, and `scripts/pack.mjs` inlines it into one file and checks it runs in Chrome.
- **The app:** `scripts/build-exe.mjs` compiles `host/PixelBattle.cs` with the C# compiler that comes with Windows (.NET Framework 4), and packs the page and the WebView2 DLLs inside it, gzipped. Building needs nothing installed beyond Node.
- **WebView2 SDK:** Microsoft's SDK comes from nuget.org on the first build, and is kept in `source/.build`.

Where the defaults live:

- **The page's:** at the top of `source/src/main.tsx` and in `source/src/shims/device.ts`.
- **The app's:** in the `Settings` class in `source/host/PixelBattle.cs`.
