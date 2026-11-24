/**
 * Builds the standalone wallpaper: release/PixelBattle/ (and release/PixelBattle.zip).
 *
 *   PixelBattle.exe              the app: puts the battle on the desktop, behind the icons
 *   Microsoft.Web.WebView2.*.dll Microsoft's WebView2 SDK: how the app shows the page
 *   WebView2Loader.dll           (in WebView2, the Edge engine Windows comes with)
 *   battle/index.html            the battle itself (dist/index.html, from `vite build` + pack)
 *   README.txt
 *
 * Compiled with the C# compiler that comes with Windows (.NET Framework 4), so there's nothing
 * to install to build it either. The WebView2 SDK comes from nuget.org the first time (it's
 * kept in host/.webview2 after that).
 */
import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { deflateSync } from "node:zlib";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const host = join(root, "host");
const build = join(host, ".build");
const out = join(root, "release", "PixelBattle");
const ps = (cmd) => execFileSync("powershell", ["-NoProfile", "-NonInteractive", "-Command", cmd], { stdio: "inherit" });

if (!existsSync(join(root, "dist", "index.html"))) throw new Error("build-exe: no dist/index.html (run `vite build` and scripts/pack.mjs first)");
mkdirSync(build, { recursive: true });

/* ---- Microsoft's WebView2 SDK, from nuget.org (once) ---- */
const sdk = join(host, ".webview2");
if (!existsSync(join(sdk, "lib"))) {
  const index = await (await fetch("https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/index.json")).json();
  const version = index.versions.filter((v) => !v.includes("-")).pop();
  console.log(`build-exe: WebView2 SDK ${version} from nuget.org`);
  const res = await fetch(`https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/${version}/microsoft.web.webview2.${version}.nupkg`);
  if (!res.ok) throw new Error(`build-exe: couldn't download the WebView2 SDK (${res.status})`);
  const zip = join(build, "webview2.zip");
  writeFileSync(zip, Buffer.from(await res.arrayBuffer()));
  rmSync(sdk, { recursive: true, force: true });
  ps(`Expand-Archive -LiteralPath '${zip}' -DestinationPath '${sdk}' -Force`);
}
/** The .NET Framework build of a DLL in the SDK (lib/net462, or lib/net45 in older ones). */
const lib = (name) => {
  for (const d of ["net462", "net45"]) if (existsSync(join(sdk, "lib", d, name))) return join(sdk, "lib", d, name);
  throw new Error(`build-exe: ${name} not in the WebView2 SDK`);
};
const core = lib("Microsoft.Web.WebView2.Core.dll");
const winforms = lib("Microsoft.Web.WebView2.WinForms.dll");
const loader = join(sdk, "runtimes", "win-x64", "native", "WebView2Loader.dll");

/* ---- Its icon: a little pixel-art mushroom cloud over the battlefield, drawn here ---- */
const ART = [
  "................",
  ".....oyyyo......",
  "...ooyywwyoo....",
  "..oyywwwwwyyo...",
  "..oywwwwwwwyo...",
  "..ooyywwwyyoo...",
  "....ooyyyoo.....",
  "......oyo.......",
  "......oyo.......",
  "......ryr.......",
  ".....rryrr......",
  "...rrooyoorr....",
  "gggggggggggggggg",
  "dddmddddddmddddd",
  "dmddddddmdddddmd",
  "dddddddddddddddd",
];
const PAL = { ".": [20, 24, 38, 255], o: [255, 138, 26, 255], y: [255, 210, 63, 255], w: [255, 246, 200, 255], r: [200, 52, 26, 255], g: [91, 191, 58, 255], d: [138, 90, 43, 255], m: [110, 69, 32, 255] };
const crcTable = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
const crc = (buf) => {
  let c = 0xffffffff;
  for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};
const chunk = (type, data) => {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const c = Buffer.alloc(4);
  c.writeUInt32BE(crc(td));
  return Buffer.concat([len, td, c]);
};
/** The art at `size` px (each of its pixels a square of them), as a PNG. */
const png = (size) => {
  const k = size / 16;
  const raw = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) {
    raw[y * (size * 4 + 1)] = 0;
    for (let x = 0; x < size; x++) {
      const [r, g, b, a] = PAL[ART[Math.floor(y / k)][Math.floor(x / k)]];
      raw.set([r, g, b, a], y * (size * 4 + 1) + 1 + x * 4);
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr.set([8, 6, 0, 0, 0], 8);
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]);
};
const sizes = [16, 32, 48, 256];
const images = sizes.map(png);
const head = Buffer.alloc(6 + 16 * sizes.length);
head.writeUInt16LE(0, 0);
head.writeUInt16LE(1, 2);
head.writeUInt16LE(sizes.length, 4);
let offset = head.length;
sizes.forEach((s, i) => {
  const e = 6 + i * 16;
  head[e] = s >= 256 ? 0 : s;
  head[e + 1] = s >= 256 ? 0 : s;
  head.writeUInt16LE(1, e + 4);
  head.writeUInt16LE(32, e + 6);
  head.writeUInt32LE(images[i].length, e + 8);
  head.writeUInt32LE(offset, e + 12);
  offset += images[i].length;
});
const ico = join(build, "PixelBattle.ico");
writeFileSync(ico, Buffer.concat([head, ...images]));

/* ---- Compile it, with the C# compiler Windows comes with ---- */
const windir = process.env.WINDIR ?? "C:\\Windows";
const csc = join(windir, "Microsoft.NET", "Framework64", "v4.0.30319", "csc.exe");
if (!existsSync(csc)) throw new Error(`build-exe: no C# compiler at ${csc} (it comes with Windows' .NET Framework 4)`);
rmSync(out, { recursive: true, force: true });
mkdirSync(join(out, "battle"), { recursive: true });
const facades = join(windir, "Microsoft.NET", "Framework64", "v4.0.30319");
const refs = ["System.dll", "System.Core.dll", "System.Drawing.dll", "System.Windows.Forms.dll", core, winforms];
if (existsSync(join(facades, "netstandard.dll"))) refs.push(join(facades, "netstandard.dll"));
execFileSync(
  csc,
  [
    "/nologo",
    "/target:winexe",
    "/platform:x64",
    "/optimize+",
    `/out:${join(out, "PixelBattle.exe")}`,
    `/win32icon:${ico}`,
    `/win32manifest:${join(host, "app.manifest")}`,
    ...refs.map((r) => `/reference:${r}`),
    join(host, "PixelBattle.cs"),
  ],
  { stdio: "inherit" }
);

/* ---- Everything it needs next to it ---- */
for (const f of [core, winforms, loader]) copyFileSync(f, join(out, f.split(/[\\/]/).pop()));
copyFileSync(join(root, "dist", "index.html"), join(out, "battle", "index.html"));
writeFileSync(
  join(out, "README.txt"),
  [
    "PIXEL BATTLE - a live wallpaper",
    "",
    "Run PixelBattle.exe: the battle plays on your desktop, behind the icons.",
    "Its icon in the taskbar's tray (the ^ by the clock) has the menu: pause, speed, zoom,",
    "frame rate, which screens, clicks, start with Windows, and Exit (which puts your own",
    "wallpaper back).",
    "",
    "It sleeps (drawing nothing) whenever a maximised or full-screen window covers the screen,",
    "while the PC is locked, and on battery. Click the desktop to call in a blast; hold the",
    "button for a barrage.",
    "",
    "Needs nothing installed: it runs on the .NET Framework and the WebView2 (Edge) engine that",
    "come with Windows 10 and 11. Keep the files together; the folder can go anywhere.",
    "Its settings are kept in %LOCALAPPDATA%\\PixelBattle.",
    "",
  ].join("\r\n")
);
const zip = join(root, "release", "PixelBattle.zip");
rmSync(zip, { force: true });
ps(`Compress-Archive -Path '${out}\\*' -DestinationPath '${zip}' -Force`);
const size = (p) => (statSync(p).size / 1024).toFixed(0) + " KB";
console.log(`build-exe: ${out}`);
for (const f of readdirSync(out)) if (f !== "battle") console.log(`  ${f} (${size(join(out, f))})`);
console.log(`  battle/index.html (${size(join(out, "battle", "index.html"))})`);
console.log(`build-exe: ${zip} (${size(zip)})`);
