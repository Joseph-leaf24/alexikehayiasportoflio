/**
 * Builds the wallpaper: ../PixelBattle.exe (Wallpaper/PixelBattle.exe), one file with everything
 * it needs inside it.
 *
 *   the app          host/PixelBattle.cs: puts the battle on the desktop, behind the icons
 *   payload/battle/index.html                the battle itself (dist/index.html, from vite + pack)
 *   payload/Microsoft.Web.WebView2.*.dll     Microsoft's WebView2 SDK: how the app shows the page,
 *   payload/WebView2Loader.dll               in WebView2 (the Edge engine Windows comes with)
 *
 * The payload's files go in gzipped, as resources; the exe unpacks them into
 * %LOCALAPPDATA%\PixelBattle\app\<stamp> when it runs (the stamp: a hash of them, so a new build
 * unpacks afresh). Compiled with the C# compiler that comes with Windows (.NET Framework 4), so
 * there's nothing to install to build it either. The WebView2 SDK comes from nuget.org the first
 * time (it's kept in .build/webview2 after that).
 */
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { deflateSync, gzipSync } from "node:zlib";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const host = join(root, "host");
const build = join(root, ".build");
const exe = join(dirname(root), "PixelBattle.exe");
const ps = (cmd) => execFileSync("powershell", ["-NoProfile", "-NonInteractive", "-Command", cmd], { stdio: "inherit" });
const kb = (n) => `${Math.round(n / 1024)} KB`;

const page = join(root, "dist", "index.html");
if (!existsSync(page)) throw new Error("build-exe: no dist/index.html (run `vite build` and scripts/pack.mjs first, or just `npm run build`)");
mkdirSync(build, { recursive: true });

/* ---- Microsoft's WebView2 SDK, from nuget.org (once) ---- */
const sdk = join(build, "webview2");
if (!existsSync(join(sdk, "lib"))) {
  const index = await (await fetch("https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/index.json")).json();
  const version = index.versions.filter((v) => !v.includes("-")).pop();
  console.log(`build-exe: WebView2 SDK ${version}, from nuget.org`);
  const res = await fetch(`https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/${version}/microsoft.web.webview2.${version}.nupkg`);
  if (!res.ok) throw new Error(`build-exe: couldn't download the WebView2 SDK (${res.status})`);
  const zip = join(build, "webview2.zip");
  writeFileSync(zip, Buffer.from(await res.arrayBuffer()));
  rmSync(sdk, { recursive: true, force: true });
  ps(`Expand-Archive -LiteralPath '${zip}' -DestinationPath '${sdk}' -Force`);
  rmSync(zip, { force: true });
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
/** The art's colour at (x, y) of a `size` px square (each of its pixels a block of them). */
const at = (size, x, y) => PAL[ART[Math.floor((y * 16) / size)][Math.floor((x * 16) / size)]];
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
/** The big one, as a PNG. */
const png = (size) => {
  const raw = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) raw.set(at(size, x, y), y * (size * 4 + 1) + 1 + x * 4);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr.set([8, 6, 0, 0, 0], 8);
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]);
};
/** The small ones as plain bitmaps (what the tray and every icon reader take): BGRA, bottom row first, then an all-opaque mask. */
const dib = (size) => {
  const mask = Math.ceil(size / 32) * 4 * size;
  const out = Buffer.alloc(40 + size * size * 4 + mask);
  out.writeUInt32LE(40, 0);
  out.writeInt32LE(size, 4);
  out.writeInt32LE(size * 2, 8);
  out.writeUInt16LE(1, 12);
  out.writeUInt16LE(32, 14);
  out.writeUInt32LE(size * size * 4 + mask, 20);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const [r, g, b, a] = at(size, x, size - 1 - y);
      out.set([b, g, r, a], 40 + (y * size + x) * 4);
    }
  return out;
};
const sizes = [16, 20, 24, 32, 40, 48, 64, 256];
const images = sizes.map((s) => (s >= 256 ? png(s) : dib(s)));
const head = Buffer.alloc(6 + 16 * sizes.length);
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

/* ---- The payload: gzipped, and stamped with a hash of what's in it ---- */
const payload = [
  ["battle/index.html", page],
  ["Microsoft.Web.WebView2.Core.dll", core],
  ["Microsoft.Web.WebView2.WinForms.dll", winforms],
  ["WebView2Loader.dll", loader],
];
const packed = join(build, "payload");
rmSync(packed, { recursive: true, force: true });
const hash = createHash("sha256");
const resources = payload.map(([name, file]) => {
  const data = readFileSync(file);
  hash.update(name).update(data);
  const gz = join(packed, name.replace(/[\\/]/g, "_") + ".gz");
  mkdirSync(dirname(gz), { recursive: true });
  writeFileSync(gz, gzipSync(data, { level: 9 }));
  return `/resource:${gz},payload/${name}.gz`;
});
const stamp = hash.digest("hex").slice(0, 12);
const stampFile = join(build, "Build.g.cs");
writeFileSync(stampFile, `// Written by scripts/build-exe.mjs: which build of the payload this exe carries.\nnamespace PixelBattle\n{\n    static class Build\n    {\n        public const string Stamp = "${stamp}";\n    }\n}\n`);

/* ---- Compile it, with the C# compiler Windows comes with ---- */
const windir = process.env.WINDIR ?? "C:\\Windows";
const fw = join(windir, "Microsoft.NET", "Framework64", "v4.0.30319");
const csc = join(fw, "csc.exe");
if (!existsSync(csc)) throw new Error(`build-exe: no C# compiler at ${csc} (it comes with Windows' .NET Framework 4)`);
const refs = ["System.dll", "System.Core.dll", "System.Drawing.dll", "System.Windows.Forms.dll", core, winforms];
if (existsSync(join(fw, "netstandard.dll"))) refs.push(join(fw, "netstandard.dll"));
const made = join(build, "PixelBattle.exe");
rmSync(made, { force: true });
execFileSync(
  csc,
  [
    "/nologo",
    "/target:winexe",
    "/platform:x64",
    "/optimize+",
    `/out:${made}`,
    `/win32icon:${ico}`,
    `/win32manifest:${join(host, "app.manifest")}`,
    `/resource:${ico},PixelBattle.ico`,
    ...resources,
    ...refs.map((r) => `/reference:${r}`),
    join(host, "PixelBattle.cs"),
    stampFile,
  ],
  { stdio: "inherit" }
);

/* ---- Into place: Wallpaper/PixelBattle.exe (a running one is asked to close first) ---- */
const place = () => {
  try {
    copyFileSync(made, exe);
    return true;
  } catch (e) {
    if (e.code !== "EBUSY" && e.code !== "EPERM") throw e;
    return false;
  }
};
if (!place()) {
  console.log("build-exe: PixelBattle.exe is running; asking it to close");
  spawnSync(exe, ["--exit"]);
  let placed = false;
  for (let i = 0; i < 20 && !placed; i++) {
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 500);
    placed = place();
  }
  if (!placed) throw new Error(`build-exe: couldn't replace ${exe} (is it still running?)`);
}
console.log(`build-exe: ${exe} (${kb(statSync(exe).size)}, payload ${stamp})`);
for (const [name, file] of payload) console.log(`  inside: ${name} (${kb(statSync(file).size)})`);
