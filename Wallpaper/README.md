# Pixel Battle: live wallpaper

The pixel battle from Site 2 as a live wallpaper for Windows. It's the same endless eight-player fight through random countries (cities with traffic, day and night and weather, metros, rail yards, mountains, caverns, gunships, nukes), full screen on its own.

It runs at **2× speed**, **zoomed out to 75%** (a third more of the battlefield on the desktop), at **30 frames a second**, and with a lighter enemy and particle budget than the site.

## Install it

The finished wallpaper is in `dist/`: one self-contained `index.html`, plus the files the wallpaper apps read. `PixelBattle-Lively.zip` is the same thing zipped.

### Lively Wallpaper (free)

1. Install **Lively Wallpaper** from the Microsoft Store, or from [rocksdanister.github.io/lively](https://rocksdanister.github.io/lively/).
2. Drag `PixelBattle-Lively.zip` onto Lively's window.
   - Or click **+** (Add wallpaper), **Browse**, and pick `dist/index.html`.
3. Click **Pixel Battle** in the library to set it.

To change its settings, open the wallpaper's **⋯** menu and choose **Customise**. You can change:

- the speed (1×, **2×**, 4×);
- the zoom (50–100%, **75%**);
- the frame rate (**30**, or 60);
- whether clicking the desktop calls in a blast.

### Wallpaper Engine (Steam)

1. Click **Open Wallpaper**, then **Open from File**.
2. Pick `dist/index.html`.

The same settings are in its properties panel. It also respects Wallpaper Engine's own frame-rate limit.

## Playing with it

If the wallpaper app passes mouse input to the desktop, you can interact with the battle:

- **Click** anywhere to call in a blast there.
- **Hold** the button for an artillery barrage.

In Lively, this depends on its **Wallpaper → Input** setting being on (mouse).

## Going easy on the machine

On this machine the wallpaper measured:

| Screen | CPU (one core) | Main-thread time per frame |
|---|---|---|
| 1920×1080 | about 10% | about 3.3 ms |
| 2560×1440 | about 11% | about 3.7 ms |

It's lighter than that whenever it can't be seen, because it stops drawing when the wallpaper app hides it. The scene is drawn small and the graphics card just scales it up, so it costs the GPU very little.

To make it stop entirely when you're not looking at the desktop, set these in **Lively → Settings → Performance**:

- **Another app is fullscreen** → *Pause*
- **Another app is focused** → *Pause*, if you'd rather it only runs on an empty desktop
- **On battery** → *Pause*

Wallpaper Engine has the same options under **Settings → Performance**.

## Rebuilding

The battle's code lives in `../Site 2/src`. The wallpaper is built straight from it, so a change to the battle in Site 2 comes through on the next build.

```sh
npm install
npm run build     # builds dist/ and PixelBattle-Lively.zip; checks the page runs off the disk without errors and takes its preview
npm run measure   # how hard it works: frames a second and CPU (npm run measure -- 2560 1440 30 for another screen size)
npm run dev       # the wallpaper in a browser tab, live-reloading
```

The defaults (speed, zoom, frame rate, budget) are at the top of `src/main.tsx` and in `src/shims/device.ts`.
