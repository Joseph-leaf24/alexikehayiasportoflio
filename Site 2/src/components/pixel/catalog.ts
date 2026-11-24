/**
 * The big catalogue of things in the world, each painted pixel by pixel: furniture and kit for
 * every kind of building (desks and computers, ovens and tables, shelves of goods, hospital
 * beds, gym gear, arcade cabinets...), street furniture and parked vehicles for the roads,
 * docks and port gear for the harbours, farm things, and what lies underground (mine carts,
 * pit props, treasure, fossils, subway cars). All original pixel art, in the style of the scene.
 *
 * Each entry says how big it is, what it breaks into, what lights up, burns, smokes or blows,
 * and how it's mounted (standing on the floor, on the wall, or hanging from the ceiling), with
 * a painter. Painters draw with the tiny helpers below into the context `paint` hands them.
 */
import type { AssetDef } from "./assets";

type Ctx = CanvasRenderingContext2D;
let c: Ctx;
const R = (x: number, y: number, w: number, h: number, col: string) => {
  c.fillStyle = col;
  c.fillRect(x, y, w, h);
};
const P = (x: number, y: number, col: string) => R(x, y, 1, 1, col);
const Ln = (x0: number, y0: number, x1: number, y1: number, col: string) => {
  const n = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
  for (let k = 0; k <= n; k++) P(Math.round(x0 + ((x1 - x0) * k) / n), Math.round(y0 + ((y1 - y0) * k) / n), col);
};
const O = (cx: number, cy: number, r: number, col: string) => {
  c.fillStyle = col;
  for (let y = -Math.floor(r); y <= Math.floor(r); y++) {
    const h = Math.floor(Math.sqrt(Math.max(0, r * r - y * y)) + 0.35);
    c.fillRect(Math.round(cx - h), Math.round(cy + y), h * 2 + 1, 1);
  }
};
/** A box lit on its top and left, shadowed on its right and bottom. */
const B = (x: number, y: number, w: number, h: number, base: string, light?: string, dark?: string) => {
  R(x, y, w, h, base);
  if (light) {
    R(x, y, w, 1, light);
    R(x, y, 1, h, light);
  }
  if (dark) {
    R(x + w - 1, y, 1, h, dark);
    R(x, y + h - 1, w, 1, dark);
  }
};

/* Colours */
const WD = "#9c6834", WL = "#c08a4a", WK = "#5a3816";
const MT = "#5d6578", ML = "#9aa1b0", MK = "#3a3f4f";
const WH = "#e8e4dc", WG = "#c8c4bc", BK = "#1e1e24", GY = "#8a8a92";
const RD = "#c8302a", RL = "#e8584a", RK = "#7a1a16";
const BL = "#3a6ab0", BLL = "#7ab0e8", BLK = "#2a4a80";
const GN = "#4a8a3a", GNL = "#6aaa4a", GNK = "#2c5a24";
const YL = "#ffcc33", YLL = "#ffe27a", OR = "#ff8a1a", PK = "#ff6fae", PU = "#8a4ac8";
const SCR = "#7ab0e8", GLS = "#8fd3ff", SK = "#e0a878", CR = "#d8c8a0";

/* Little pieces used again and again */
const chair = (x: number, y: number, col: string) => {
  R(x, y, 1, 7, col);
  R(x, y + 3, 4, 1, col);
  R(x + 3, y + 3, 1, 4, col);
};
const legs = (x: number, y: number, w: number, h: number, col: string) => {
  R(x, y, 1, h, col);
  R(x + w - 1, y, 1, h, col);
};
const monitor = (x: number, y: number, w = 6, h = 4) => {
  R(x, y, w, h, MK);
  R(x + 1, y + 1, w - 2, h - 2, SCR);
  P(x + 1, y + 1, "#c8e8ff");
  R(x + (w >> 1), y + h, 1, 1, MK);
};
const goods = (x: number, y: number, w: number, cols: string[], tall = 2) => {
  for (let k = 0; k < w; k += 2) R(x + k, y - tall, 1 + ((k >> 1) % 2), tall, cols[(k >> 1) % cols.length]);
};
const shelf = (w: number, h: number, frame: string, rows: string[][]) => {
  R(0, 0, w, h, "#3a2a20");
  legs(0, 0, w, h, frame);
  R(0, 0, w, 1, frame);
  const step = Math.floor((h - 1) / rows.length);
  rows.forEach((cols, i) => {
    const y = 1 + step * (i + 1) - 1;
    R(0, y, w, 1, frame);
    goods(1, y, w - 2, cols, Math.min(3, step - 1));
  });
};
const wheel = (x: number, y: number, r = 2) => {
  O(x, y, r, BK);
  P(x, y, GY);
};
/** A car side-on: body, cabin, windows and wheels. */
const car = (w: number, h: number, body: string, light: string, o: { cab?: [number, number]; win?: string; low?: boolean } = {}) => {
  const [cx, cw] = o.cab ?? [Math.round(w * 0.25), Math.round(w * 0.5)];
  const ch = Math.round(h * 0.42);
  R(cx, 0, cw, ch + 1, body);
  R(cx + 1, 1, cw - 2, ch - 1, o.win ?? GLS);
  R(cx + (cw >> 1), 1, 1, ch - 1, body);
  R(0, ch, w, h - ch - 2, body);
  R(0, ch, w, 1, light);
  R(w - 2, ch + 1, 2, 1, YLL);
  P(0, ch + 1, RD);
  wheel(4, h - 2);
  wheel(w - 5, h - 2);
};

type Entry = AssetDef & { paint: () => void };
const E: Record<string, Entry> = {};
const add = (id: string, w: number, h: number, debris: string, paint: () => void, o: Partial<AssetDef> = {}) => {
  E[id] = { w, h, debris, paint, ...o };
};

/* ================================================================== */
/* Offices                                                             */
/* ================================================================== */
add("desk", 14, 10, WD, () => {
  monitor(5, 0);
  R(0, 5, 14, 2, WD);
  R(0, 5, 14, 1, WL);
  legs(0, 7, 14, 3, WK);
  R(10, 7, 4, 3, WK);
  R(1, 4, 3, 1, BK);
});
add("officechair", 6, 9, BK, () => {
  R(0, 0, 2, 5, "#2a2e3a");
  R(0, 4, 5, 2, "#2a2e3a");
  R(2, 6, 1, 2, MT);
  R(0, 8, 6, 1, MK);
});
add("filingcabinet", 6, 12, MT, () => {
  B(0, 0, 6, 12, "#7a8294", "#9aa1b0", "#4a5264");
  for (const y of [3, 6, 9]) R(0, y, 6, 1, "#4a5264");
  for (const y of [1, 4, 7, 10]) R(2, y, 2, 1, "#c8ced9");
});
add("watercooler", 5, 13, BLL, () => {
  R(1, 0, 3, 5, "#a8d8f0");
  P(1, 1, "#e8f8ff");
  B(0, 5, 5, 8, WH, "#ffffff", WG);
  P(1, 7, BL);
  P(3, 7, RD);
});
add("printer", 10, 7, WG, () => {
  B(0, 1, 10, 6, WH, "#ffffff", WG);
  R(2, 0, 6, 1, "#f8f8f8");
  R(1, 3, 8, 1, BK);
  P(8, 5, "#5bbf3a");
}, { lights: [[8, 5, "#8aff6a", 1.2]] });
add("photocopier", 12, 12, WG, () => {
  B(0, 2, 12, 10, WH, "#ffffff", WG);
  R(0, 0, 12, 2, "#5d6578");
  R(2, 5, 8, 2, MK);
  R(9, 3, 2, 1, "#5bbf3a");
  R(1, 9, 10, 1, WG);
});
add("pottedplant", 7, 12, GN, () => {
  O(3, 4, 3, GN);
  P(2, 2, GNL);
  P(5, 4, GNL);
  R(3, 6, 1, 2, GNK);
  B(1, 8, 5, 4, "#b8543a", "#d8744a", "#8a3e2c");
});
add("whiteboard", 16, 10, WH, () => {
  R(0, 0, 16, 9, MT);
  R(1, 1, 14, 7, "#f8f8f4");
  Ln(2, 6, 6, 3, BL);
  Ln(6, 3, 9, 5, BL);
  Ln(9, 5, 13, 2, RD);
  R(2, 2, 4, 1, BK);
  R(5, 9, 6, 1, MK);
}, { wall: true });
add("wallclock", 7, 7, WH, () => {
  O(3, 3, 3, BK);
  O(3, 3, 2, "#f8f8f4");
  R(3, 1, 1, 3, BK);
  R(3, 3, 2, 1, RD);
}, { wall: true });
add("serverrack", 8, 16, MK, () => {
  B(0, 0, 8, 16, "#2a2e3a", "#3a3f4f", "#14161e");
  for (let y = 2; y < 15; y += 2) {
    R(1, y, 6, 1, "#4a5264");
    P(6, y, y % 4 ? "#8aff6a" : "#5ab0ff");
  }
}, { lights: [[6, 2, "#8aff6a", 3], [6, 6, "#5ab0ff", 2], [6, 10, "#8aff6a", 4], [6, 14, "#ff3b3b", 1.5]] });
add("conferencetable", 22, 9, WD, () => {
  chair(0, 2, MK);
  R(4, 3, 14, 2, WK);
  R(4, 3, 14, 1, WD);
  legs(5, 5, 12, 4, WK);
  monitor(9, 0, 4, 3);
  R(21, 2, 1, 7, MK);
  R(18, 5, 4, 1, MK);
  R(18, 5, 1, 4, MK);
});
add("reception", 18, 10, WD, () => {
  B(0, 3, 18, 7, WD, WL, WK);
  R(0, 3, 18, 1, "#e8e0d0");
  monitor(11, 0, 5, 3);
  R(2, 5, 6, 2, YL);
  P(3, 2, "#ffd479");
  R(3, 1, 1, 2, MK);
});
add("coffeemachine", 6, 9, MK, () => {
  B(0, 0, 6, 9, "#3a3f4f", "#5d6578", "#1e1e24");
  R(1, 2, 4, 1, RD);
  R(2, 5, 2, 2, WH);
  P(1, 1, "#ff3b3b");
}, { lights: [[1, 1, "#ff3b3b", 0.7]] });
add("vendingsoda", 9, 16, RD, () => {
  B(0, 0, 9, 16, RD, RL, RK);
  R(1, 1, 5, 11, "#a8d8f0");
  for (let y = 2; y < 11; y += 3) for (let x = 1; x < 6; x += 2) R(x, y, 1, 2, [RD, BL, "#5bbf3a", YL][(x + y) % 4]);
  R(6, 2, 2, 3, BK);
  R(1, 13, 5, 2, BK);
}, { lights: [[3, 2, "#c8f0ff", 0]] });
add("vendingsnack", 9, 16, BL, () => {
  B(0, 0, 9, 16, BLK, BL, "#1a2a50");
  R(1, 1, 5, 11, "#c8e0f0");
  for (let y = 3; y < 12; y += 3) {
    R(1, y, 5, 1, MT);
    goods(1, y, 5, [OR, YL, RD, "#8a5a2b"], 2);
  }
  R(6, 2, 2, 4, MK);
  R(1, 13, 5, 2, BK);
});
add("bookshelf", 12, 16, WD, () => shelf(12, 16, WD, [[RD, BL, GN, YL, "#8a5a2b"], [PU, RD, "#3a6a5a", WH], [BL, OR, GN, RD], [YL, "#8a5a2b", BL, WH]]));
add("lamp", 5, 14, YLL, () => {
  R(0, 0, 5, 4, "#f0e0a0");
  P(1, 1, "#ffffff");
  R(2, 4, 1, 9, MK);
  R(0, 13, 5, 1, MK);
}, { lights: [[2, 3, "#ffe8a0", 0]] });

/* ================================================================== */
/* Food and drink                                                      */
/* ================================================================== */
add("diningtable", 16, 8, WD, () => {
  chair(0, 1, WK);
  R(4, 2, 8, 1, "#f4f0e8");
  R(4, 3, 8, 1, WD);
  R(7, 4, 2, 4, WK);
  R(5, 1, 2, 1, WH);
  R(9, 0, 1, 2, RD);
  R(15, 1, 1, 7, WK);
  R(12, 4, 4, 1, WK);
  R(12, 4, 1, 4, WK);
});
add("booth", 16, 9, RD, () => {
  R(0, 0, 3, 9, RD);
  R(0, 0, 1, 9, RL);
  R(13, 0, 3, 9, RD);
  R(15, 0, 1, 9, RK);
  R(3, 5, 10, 1, RK);
  R(3, 3, 10, 1, "#e8e0d0");
  R(7, 4, 2, 5, MT);
  R(5, 2, 2, 1, YL);
});
add("barcounter", 22, 10, WD, () => {
  R(0, 2, 22, 2, "#6a3a1e");
  R(0, 2, 22, 1, WL);
  B(1, 4, 20, 6, WK, WD, "#3a2410");
  for (let x = 3; x < 20; x += 5) R(x, 6, 3, 3, "#6e4520");
  for (let x = 2; x < 20; x += 4) R(x, 0, 1, 2, [GN, "#8a3a2a", YL, BLL][x % 4]);
});
add("barstool", 5, 9, RD, () => {
  R(0, 0, 5, 2, RD);
  R(0, 0, 5, 1, RL);
  R(2, 2, 1, 6, MT);
  R(1, 5, 3, 1, MT);
  R(1, 8, 3, 1, MK);
});
add("beertaps", 10, 10, "#c8a040", () => {
  R(0, 2, 10, 2, "#c8a040");
  for (const x of [1, 4, 7]) {
    R(x, 0, 2, 2, [RD, BK, GN][x % 3]);
    R(x + 1, 4, 1, 2, MT);
  }
  B(0, 6, 10, 4, MK, MT);
});
add("pizzaoven", 16, 15, "#a24c36", () => {
  for (let y = 0; y < 15; y++) {
    const half = y < 6 ? Math.round(Math.sqrt(36 - (6 - y) ** 2) * 1.2) : 8;
    R(8 - half, y, half * 2, 1, y % 3 === 0 ? "#8a3e2c" : "#a24c36");
  }
  O(8, 9, 3.5, BK);
  R(5, 9, 7, 4, BK);
  R(6, 11, 5, 2, OR);
  P(7, 11, YLL);
  R(6, 0, 3, 1, MK);
}, { flames: [[8, 11]], lights: [[8, 11, "#ff8a1a", 1.5]] });
add("stove", 14, 12, ML, () => {
  B(0, 4, 14, 8, "#c8ced9", "#e8eef6", "#8a90a0");
  R(2, 6, 10, 4, MK);
  R(3, 7, 8, 2, "#2a2e3a");
  R(0, 3, 14, 1, MK);
  B(1, 0, 5, 3, MT, ML);
  B(8, 1, 5, 2, "#b0b0b8", WH);
}, { smoke: [[4, 0]] });
add("fridge", 8, 17, WH, () => {
  B(0, 0, 8, 17, WH, "#ffffff", WG);
  R(0, 6, 8, 1, WG);
  R(6, 2, 1, 3, MT);
  R(6, 8, 1, 5, MT);
  R(2, 2, 2, 2, [YL, PK, BLL][1]);
});
add("chestfreezer", 14, 8, WH, () => {
  B(0, 1, 14, 7, WH, "#ffffff", WG);
  R(0, 0, 14, 2, "#a8d8f0");
  R(5, 3, 4, 1, MT);
  R(1, 5, 12, 1, BLL);
});
add("sinkcounter", 14, 10, ML, () => {
  R(0, 3, 14, 2, "#c8ced9");
  R(3, 3, 6, 1, MK);
  R(5, 0, 1, 3, MT);
  R(5, 0, 3, 1, MT);
  B(0, 5, 14, 5, WD, WL, WK);
  R(7, 6, 1, 3, WK);
});
add("deepfryer", 9, 11, ML, () => {
  B(0, 3, 9, 8, "#b0b6c4", "#d0d6e4", "#6a7080");
  R(1, 2, 7, 2, "#c8a040");
  R(1, 0, 3, 2, MK);
  R(5, 0, 3, 2, MK);
  R(3, 7, 3, 1, RD);
}, { smoke: [[4, 2]] });
add("grill", 14, 10, MK, () => {
  R(0, 2, 14, 2, BK);
  for (let x = 1; x < 13; x += 2) P(x, 2, OR);
  R(2, 1, 4, 1, "#8a3a2a");
  R(8, 1, 4, 1, "#a05030");
  B(0, 4, 14, 6, "#3a3f4f", "#5d6578", "#1e1e24");
}, { flames: [[5, 2], [10, 2]], smoke: [[7, 0]] });
add("sushicounter", 22, 9, "#e8e0d0", () => {
  R(0, 3, 22, 2, MT);
  for (let x = 1; x < 21; x += 4) {
    O(x + 1, 2, 1.2, WH);
    P(x + 1, 1, x % 8 === 1 ? OR : "#c8302a");
  }
  B(0, 5, 22, 4, "#d8c8a0", "#e8e0c0", "#a89868");
});
add("menuboard", 14, 9, BK, () => {
  R(0, 0, 14, 9, WK);
  R(1, 1, 12, 7, "#1e2a22");
  for (let y = 2; y < 7; y += 2) {
    R(2, y, 6, 1, "#e8e4dc");
    R(10, y, 2, 1, YL);
  }
}, { wall: true });
add("register", 10, 10, WD, () => {
  B(2, 0, 6, 4, MK, MT);
  R(3, 1, 4, 1, "#8aff6a");
  B(0, 4, 10, 6, WD, WL, WK);
  R(1, 3, 8, 1, GY);
}, { lights: [[4, 1, "#8aff6a", 0]] });
add("cakecase", 14, 12, GLS, () => {
  R(0, 0, 14, 7, "#c8e8f8");
  R(0, 0, 14, 1, "#ffffff");
  for (const [x, col] of [[1, PK], [5, "#8a5a2b"], [9, YLL]] as const) {
    R(x, 3, 4, 3, col);
    R(x, 3, 4, 1, WH);
    P(x + 2, 2, RD);
  }
  B(0, 7, 14, 5, WD, WL, WK);
});
add("breadrack", 10, 15, WD, () => shelf(10, 15, WD, [["#c88a40", "#e0a850"], ["#a86a30", "#c88a40", "#e0b060"], ["#d89a50", "#b07a40"]]));
add("espresso", 8, 9, MT, () => {
  B(0, 1, 8, 8, "#c8ced9", "#e8eef6", "#8a90a0");
  R(0, 0, 8, 1, MK);
  R(2, 3, 4, 2, MK);
  R(3, 6, 2, 2, WH);
  P(6, 2, RD);
}, { smoke: [[3, 0]] });
add("icecreamtub", 14, 10, BLL, () => {
  R(0, 0, 14, 4, "#c8e8f8");
  for (let x = 1; x < 13; x += 3) R(x, 2, 2, 2, [PK, "#8a5a2b", YLL, "#a8f0a0"][x % 4]);
  B(0, 4, 14, 6, WH, "#ffffff", WG);
  R(1, 6, 12, 1, PK);
});
add("winerack", 10, 14, WD, () => {
  R(0, 0, 10, 14, WK);
  for (let y = 1; y < 14; y += 3) for (let x = 1; x < 10; x += 3) {
    O(x + 0.5, y + 0.5, 1, "#3a1424");
    P(x, y, "#8a2a3a");
  }
});
add("jukebox", 9, 14, OR, () => {
  for (let y = 0; y < 14; y++) {
    const half = y < 4 ? Math.round(Math.sqrt(16 - (4 - y) ** 2)) + 1 : 4;
    R(4 - half + 1, y, half * 2 - 1, 1, y < 2 ? OR : RD);
  }
  R(2, 3, 5, 4, YLL);
  R(3, 8, 3, 1, BK);
  R(2, 10, 5, 3, "#3a1414");
  P(1, 5, PK);
  P(7, 5, PK);
}, { lights: [[4, 4, "#ffd23f", 2], [1, 5, "#ff6fae", 3], [7, 5, "#5ab0ff", 2.5]] });
add("pooltable", 22, 9, GN, () => {
  R(0, 0, 22, 2, WK);
  R(1, 0, 20, 1, "#2a7a3a");
  P(5, 0, WH);
  P(9, 0, RD);
  P(11, 0, YL);
  B(1, 2, 20, 4, WD, WL, WK);
  R(2, 6, 2, 3, WK);
  R(18, 6, 2, 3, WK);
});
add("dartboard", 7, 7, RD, () => {
  O(3, 3, 3, BK);
  O(3, 3, 2.2, RD);
  O(3, 3, 1.3, "#e8e0c0");
  P(3, 3, GN);
}, { wall: true });
add("djbooth", 16, 11, MK, () => {
  B(0, 4, 16, 7, "#2a2e3a", "#3a3f4f", "#14161e");
  O(4, 3, 2, BK);
  O(11, 3, 2, BK);
  P(4, 3, GY);
  P(11, 3, GY);
  R(7, 2, 2, 2, MT);
  for (let x = 1; x < 15; x += 2) P(x, 6, [PK, "#5ab0ff", "#8aff6a"][x % 3]);
}, { lights: [[1, 6, "#ff6fae", 4], [7, 6, "#5ab0ff", 3], [13, 6, "#8aff6a", 5]] });
add("discoball", 6, 10, WH, () => {
  R(3, 0, 1, 4, MK);
  O(3, 6, 2.6, "#c8ced9");
  P(2, 5, "#ffffff");
  P(4, 7, "#ffffff");
  P(3, 6, BLL);
}, { hang: true, lights: [[2, 5, "#ffffff", 6], [4, 7, "#ff6fae", 5]] });
add("speakerstack", 8, 16, BK, () => {
  for (const y of [0, 8]) {
    B(0, y, 8, 8, "#2a2a30", "#3a3a44", "#14141a");
    O(4, y + 4, 2.2, "#14141a");
    P(4, y + 4, GY);
  }
});
add("micstand", 5, 13, MT, () => {
  R(1, 0, 2, 2, GY);
  R(2, 2, 1, 10, MK);
  R(0, 12, 5, 1, MK);
});
add("karaoke", 16, 13, MK, () => {
  R(0, 0, 16, 10, MK);
  R(1, 1, 14, 8, "#3a2a6a");
  R(2, 6, 10, 1, YL);
  R(2, 4, 7, 1, WH);
  O(12, 3, 1.5, PK);
  R(6, 10, 4, 3, MK);
}, { lights: [[8, 5, "#c8a0ff", 0]] });

/* ================================================================== */
/* Shops                                                               */
/* ================================================================== */
add("clothesrack", 14, 13, PU, () => {
  R(0, 0, 14, 1, MT);
  legs(0, 0, 14, 13, MT);
  for (let x = 2; x < 13; x += 2) {
    R(x, 1, 2, 7, [RD, BL, PK, YL, GN, PU][x % 6]);
    P(x, 1, MK);
  }
  R(0, 12, 14, 1, MK);
});
add("mannequin", 6, 18, WG, () => {
  O(3, 2, 1.8, "#e8d8c8");
  R(1, 4, 4, 6, [RD, BL, PK][0]);
  R(2, 10, 2, 4, BK);
  R(2, 14, 1, 3, MT);
  R(0, 17, 6, 1, MK);
});
add("shoeshelf", 12, 12, WD, () => shelf(12, 12, WL, [[RD, BK, WH, BL], [BK, "#8a5a2b", PK], [WH, RD, GN, BK]]));
add("cans", 14, 14, RD, () => shelf(14, 14, MT, [[RD, RL, RD, YL], [GN, GNL, WH, GN], [BL, BLL, RD, YL]]));
add("cereal", 14, 14, YL, () => shelf(14, 14, MT, [[YL, OR, RD, BL], [OR, YL, GN, PK], [RD, YL, BL, OR]]));
add("bottles", 14, 14, GN, () => shelf(14, 14, MT, [[GN, "#a8d8f0", RD, GN], [BLL, OR, GN, "#a8d8f0"], [YL, GN, RD, BLL]]));
add("producecrates", 16, 9, GN, () => {
  for (const [x, col, hi] of [[0, RD, RL], [5, YL, YLL], [10, GN, GNL]] as const) {
    B(x, 4, 6, 5, WD, WL, WK);
    for (let k = 0; k < 5; k += 2) {
      O(x + 1 + k, 3, 1.2, col);
      P(x + 1 + k, 2, hi);
    }
  }
});
add("checkout", 18, 10, MT, () => {
  B(0, 3, 18, 7, "#5d6578", "#7a8294", "#3a3f4f");
  R(0, 3, 12, 1, BK);
  B(12, 0, 5, 3, MK, MT);
  R(13, 1, 3, 1, "#8aff6a");
  R(2, 1, 3, 2, OR);
  R(6, 1, 2, 2, BL);
});
add("shoppingcart", 10, 8, ML, () => {
  for (let y = 0; y < 5; y++) R(y > 0 ? 1 : 0, y, 8 - (y > 2 ? 1 : 0), 1, y % 2 ? MT : ML);
  R(0, 0, 1, 1, RD);
  R(8, 0, 2, 1, MT);
  R(1, 5, 7, 1, MT);
  wheel(2, 7, 1);
  wheel(7, 7, 1);
});
add("toyshelf", 14, 14, YL, () => shelf(14, 14, RD, [[YL, BL, GN, PK], [RD, OR, PU, YL], [BLL, PK, GN, RD]]));
add("recordbins", 16, 9, WD, () => {
  B(0, 4, 16, 5, WD, WL, WK);
  for (let x = 1; x < 15; x += 2) R(x, 0, 2, 5, [BK, RD, BL, YL, PU, GN, WH][x % 7]);
});
add("tvwall", 18, 12, MK, () => {
  R(0, 0, 18, 12, MK);
  for (const [x, y, col] of [[1, 1, "#3a6ab0"], [10, 1, "#5a3a8a"], [1, 7, "#3a8a5a"], [10, 7, "#8a3a3a"]] as const) {
    R(x, y, 7, 4, col);
    P(x + 1, y + 1, "#c8e8ff");
  }
}, { wall: true, lights: [[4, 3, "#7ab0e8", 3], [13, 3, "#c8a0ff", 2], [4, 9, "#8aff9a", 2.5], [13, 9, "#ff9a8a", 3.5]] });
add("guitarstand", 6, 16, OR, () => {
  O(3, 12, 2.6, OR);
  O(3, 9, 1.8, OR);
  P(3, 11, BK);
  R(3, 1, 1, 8, WK);
  R(2, 0, 3, 2, BK);
  R(0, 15, 6, 1, MK);
});
add("jewelcase", 14, 11, GLS, () => {
  R(0, 0, 14, 5, "#c8e8f8");
  for (let x = 2; x < 13; x += 3) {
    O(x, 3, 1, YL);
    P(x, 2, x % 2 ? "#5ff0e0" : RD);
  }
  B(0, 5, 14, 6, "#3a2a4a", "#5a4a6a", "#1a1a2a");
}, { lights: [[2, 2, "#ffffff", 4], [8, 2, "#5ff0e0", 3]] });
add("flowerbuckets", 14, 9, PK, () => {
  for (const [x, col] of [[0, PK], [5, YL], [10, RD]] as const) {
    B(x, 5, 4, 4, MT, ML, MK);
    for (let k = 0; k < 4; k++) {
      R(x + k, 2, 1, 3, GN);
      P(x + k, 1 + (k % 2), col);
    }
  }
});
add("aquarium", 14, 12, BLL, () => {
  R(0, 0, 14, 8, MK);
  R(1, 1, 12, 6, "#3a8ac8");
  R(1, 6, 12, 1, "#c8a060");
  R(3, 3, 2, 1, OR);
  P(5, 3, OR);
  R(9, 4, 2, 1, YL);
  for (const y of [2, 4]) P(7, y, "#c8f0ff");
  B(1, 8, 12, 4, WD, WL, WK);
}, { lights: [[7, 2, "#7ad0ff", 0.8]] });
add("petcage", 10, 10, ML, () => {
  R(0, 0, 10, 8, "#3a3020");
  for (let x = 0; x < 10; x += 2) R(x, 0, 1, 8, ML);
  R(0, 0, 10, 1, ML);
  O(5, 5, 1.6, "#e8c8a0");
  P(4, 4, BK);
  R(0, 8, 10, 2, RD);
});
add("barberchair", 10, 14, RD, () => {
  R(1, 0, 3, 8, RD);
  R(1, 0, 1, 8, RL);
  R(1, 6, 8, 3, RD);
  R(8, 5, 1, 4, MT);
  R(4, 9, 2, 4, ML);
  R(1, 13, 8, 1, MK);
});
add("hairdryer", 8, 16, PK, () => {
  O(4, 3, 3.2, PK);
  R(1, 3, 7, 2, PK);
  R(1, 5, 6, 1, "#c8508a");
  R(4, 6, 1, 9, ML);
  R(1, 15, 7, 1, MK);
});
add("mirror", 8, 12, GLS, () => {
  R(0, 0, 8, 12, YL);
  R(1, 1, 6, 10, "#c8e8f8");
  Ln(2, 8, 5, 2, "#ffffff");
}, { wall: true });
add("washer", 9, 11, WH, () => {
  B(0, 0, 9, 11, WH, "#ffffff", WG);
  R(0, 2, 9, 1, WG);
  O(4, 6, 2.8, MK);
  O(4, 6, 1.8, "#5a8ad0");
  P(7, 1, "#5bbf3a");
}, { lights: [[7, 1, "#8aff6a", 0]] });
add("laundrybasket", 8, 6, "#c8a060", () => {
  B(0, 2, 8, 4, "#c8a060", "#e0c080", "#8a6a40");
  for (let x = 1; x < 7; x += 2) R(x, 0, 2, 2, [BL, PK, WH][x % 3]);
});
add("tattoochair", 14, 10, BK, () => {
  R(0, 3, 12, 3, "#2a2a30");
  R(0, 3, 12, 1, "#4a4a54");
  R(11, 0, 3, 5, "#2a2a30");
  R(5, 6, 2, 3, MT);
  R(2, 9, 8, 1, MK);
});
add("atm", 8, 15, MT, () => {
  B(0, 0, 8, 15, "#5d6578", "#7a8294", "#3a3f4f");
  R(1, 2, 6, 4, "#3a8ac8");
  R(1, 7, 6, 3, MK);
  R(2, 11, 4, 1, BK);
  R(0, 0, 8, 1, BL);
}, { lights: [[3, 3, "#7ad0ff", 0]] });
add("safe", 10, 10, MK, () => {
  B(0, 0, 10, 10, "#4a5264", "#5d6578", "#2a2e3a");
  O(5, 5, 2, MT);
  P(5, 5, YL);
  R(8, 3, 1, 4, ML);
});

/* ================================================================== */
/* Health, civic, schools, culture                                     */
/* ================================================================== */
add("hospitalbed", 18, 10, WH, () => {
  R(0, 0, 1, 8, ML);
  R(17, 2, 1, 6, ML);
  R(1, 3, 16, 2, "#a8d8f0");
  R(1, 2, 4, 2, WH);
  R(1, 5, 16, 1, ML);
  wheel(3, 8, 1);
  wheel(15, 8, 1);
});
add("ivstand", 5, 16, BLL, () => {
  R(0, 0, 5, 1, ML);
  R(0, 1, 2, 4, "#c8e8f8");
  R(2, 0, 1, 15, ML);
  R(0, 15, 5, 1, MK);
});
add("heartmonitor", 8, 14, MK, () => {
  B(0, 0, 8, 6, MK, MT);
  R(1, 1, 6, 4, "#0a1a10");
  Ln(1, 3, 3, 3, "#8aff6a");
  Ln(3, 1, 4, 4, "#8aff6a");
  P(6, 3, "#8aff6a");
  R(3, 6, 1, 7, ML);
  R(1, 13, 6, 1, MK);
}, { lights: [[4, 2, "#8aff6a", 2]] });
add("xraybox", 14, 9, WH, () => {
  R(0, 0, 14, 9, MT);
  R(1, 1, 12, 7, "#c8e8f8");
  R(6, 2, 2, 5, WH);
  for (let y = 2; y < 7; y += 2) {
    R(3, y, 3, 1, WH);
    R(8, y, 3, 1, WH);
  }
}, { wall: true, lights: [[7, 4, "#e8f8ff", 0]] });
add("wheelchair", 10, 10, MT, () => {
  R(0, 0, 2, 6, MT);
  R(0, 5, 7, 1, MT);
  O(4, 7, 2.8, MK);
  O(4, 7, 1.6, "#2a2e3a");
  wheel(8, 9, 1);
});
add("dentistchair", 18, 12, BLL, () => {
  R(12, 0, 4, 6, "#3a8ac8");
  R(2, 5, 12, 3, "#3a8ac8");
  R(2, 5, 12, 1, BLL);
  R(0, 7, 3, 2, "#3a8ac8");
  R(8, 8, 2, 3, ML);
  R(4, 11, 10, 1, MK);
  R(16, 0, 2, 1, MK);
  R(17, 0, 1, 4, ML);
  O(17, 5, 1.2, YLL);
}, { lights: [[17, 5, "#fff6c8", 0]] });
add("medicineshelf", 12, 14, WH, () => shelf(12, 14, WH, [[WH, RD, BLL, WH], [OR, WH, GN, BLL], [WH, PK, WH, YL]]));
add("policedesk", 16, 10, BL, () => {
  monitor(9, 0, 5, 3);
  B(0, 4, 16, 6, BLK, BL, "#1a2a50");
  R(2, 2, 4, 2, WH);
  R(2, 6, 12, 1, YL);
});
add("jailbars", 12, 16, MT, () => {
  R(0, 0, 12, 1, MK);
  R(0, 15, 12, 1, MK);
  for (let x = 0; x < 12; x += 2) R(x, 1, 1, 14, ML);
  R(0, 7, 12, 1, MT);
});
add("lockers", 12, 16, BL, () => {
  for (const x of [0, 4, 8]) {
    B(x, 0, 4, 16, BL, BLL, BLK);
    for (let y = 2; y < 6; y++) R(x + 1, y, 2, 1, y % 2 ? BLK : BL);
    P(x + 2, 9, ML);
  }
});
add("firehose", 9, 11, RD, () => {
  R(0, 0, 9, 11, RD);
  O(4, 5, 3.4, RK);
  O(4, 5, 2.2, RD);
  P(4, 5, YL);
  R(0, 0, 9, 1, RL);
}, { wall: true });
add("firepole", 3, 24, YL, () => {
  R(1, 0, 1, 24, "#c8b850");
  P(1, 2, YLL);
  R(0, 23, 3, 1, MK);
});
add("mailsort", 14, 14, WD, () => {
  R(0, 0, 14, 14, WK);
  for (let y = 1; y < 13; y += 3) for (let x = 1; x < 13; x += 3) {
    R(x, y, 2, 2, "#3a2410");
    if ((x + y) % 2) P(x, y + 1, WH);
  }
});
add("mailbags", 12, 8, CR, () => {
  O(3, 5, 3, "#a89868");
  O(8, 5, 3, CR);
  R(2, 1, 2, 2, "#8a7a50");
  R(7, 1, 2, 2, "#a89868");
  R(6, 4, 4, 1, BL);
});
add("judgebench", 20, 14, WD, () => {
  B(0, 4, 20, 10, WK, WD, "#3a2410");
  R(0, 4, 20, 1, WL);
  for (let x = 3; x < 18; x += 5) R(x, 7, 3, 5, "#6e4520");
  R(9, 0, 2, 4, WK);
  R(8, 0, 4, 1, YL);
});
add("globe", 8, 12, BL, () => {
  O(4, 4, 3.4, BL);
  R(2, 2, 3, 2, GN);
  R(4, 5, 3, 1, GN);
  Ln(1, 1, 7, 7, YL);
  R(3, 8, 2, 3, WK);
  R(1, 11, 6, 1, WK);
});
add("dinofossil", 30, 18, "#e8e0c8", () => {
  const bone = "#e8e0c8";
  Ln(2, 6, 10, 4, bone);
  Ln(10, 4, 20, 5, bone);
  Ln(20, 5, 28, 10, bone);
  for (let x = 8; x < 20; x += 2) Ln(x, 5, x - 1, 10, bone);
  R(0, 3, 4, 4, bone);
  P(1, 4, BK);
  Ln(9, 6, 8, 15, bone);
  Ln(17, 7, 18, 15, bone);
  R(6, 15, 15, 1, MT);
  R(4, 16, 19, 2, GY);
});
add("bust", 6, 14, WH, () => {
  O(3, 3, 2.5, "#e8e4dc");
  R(1, 5, 4, 3, "#e8e4dc");
  B(0, 8, 6, 6, GY, WG, MT);
});
add("easel", 10, 16, WD, () => {
  Ln(1, 15, 5, 0, WK);
  Ln(9, 15, 5, 0, WK);
  R(1, 3, 8, 7, WH);
  O(4, 6, 1.6, BL);
  R(5, 7, 3, 2, YL);
  P(3, 8, RD);
  R(0, 10, 10, 1, WK);
});
add("painting", 12, 9, YL, () => {
  R(0, 0, 12, 9, "#c8a040");
  R(1, 1, 10, 7, "#5a8ad0");
  R(1, 5, 10, 3, GN);
  O(8, 3, 1.4, YLL);
  Ln(2, 5, 5, 3, "#3a5a40");
}, { wall: true });
add("pedestalvase", 6, 16, "#c8a040", () => {
  O(3, 3, 2.4, "#3a6ab0");
  R(2, 0, 2, 2, BLK);
  R(1, 3, 4, 1, YL);
  B(1, 6, 4, 10, WH, "#ffffff", WG);
}, { lights: [[3, 3, "#ffffff", 0.5]] });
add("schooldesk", 10, 9, WD, () => {
  R(0, 3, 7, 1, WL);
  R(0, 4, 7, 1, WD);
  legs(1, 5, 5, 4, MT);
  chair(7, 2, MT);
  R(1, 2, 3, 1, WH);
});
add("chalkboard", 18, 10, GNK, () => {
  R(0, 0, 18, 10, WD);
  R(1, 1, 16, 7, "#2a4a3a");
  R(2, 2, 7, 1, WH);
  R(2, 4, 4, 1, WH);
  R(8, 4, 6, 1, WH);
  Ln(10, 6, 14, 2, WH);
  R(2, 8, 14, 1, WK);
}, { wall: true });
add("microscope", 7, 10, MK, () => {
  R(3, 0, 2, 5, MK);
  P(3, 0, BK);
  R(2, 5, 4, 1, MT);
  R(5, 4, 1, 5, MK);
  R(0, 9, 7, 1, MK);
  R(1, 6, 3, 1, GLS);
});
add("labbench", 20, 12, WH, () => {
  for (const [x, col] of [[2, "#8aff6a"], [5, "#5ab0ff"], [8, PK], [14, YLL]] as const) {
    R(x, 0, 1, 4, "#c8e8f8");
    R(x, 2, 1, 2, col);
  }
  R(11, 1, 2, 3, "#c8e8f8");
  R(11, 2, 2, 2, "#8aff6a");
  R(0, 4, 20, 2, BK);
  B(0, 6, 20, 6, WH, "#ffffff", WG);
  for (const x of [5, 14]) R(x, 8, 1, 3, WG);
}, { lights: [[2, 2, "#8aff6a", 1.5], [11, 2, "#8aff6a", 0.9], [8, 2, "#ff6fae", 2]] });
add("telescope", 14, 16, WH, () => {
  Ln(2, 11, 12, 2, "#c8ced9");
  Ln(2, 12, 12, 3, "#c8ced9");
  Ln(3, 12, 13, 3, ML);
  R(12, 1, 2, 3, MK);
  Ln(7, 8, 3, 15, MK);
  Ln(7, 8, 11, 15, MK);
  R(7, 8, 1, 7, MK);
});

/* ================================================================== */
/* Sport, fun, stage                                                   */
/* ================================================================== */
add("treadmill", 16, 11, MK, () => {
  R(12, 0, 2, 8, MK);
  R(10, 0, 5, 2, MT);
  P(12, 0, "#8aff6a");
  R(0, 8, 15, 2, BK);
  R(0, 8, 15, 1, MT);
  R(1, 10, 2, 1, MK);
});
add("weightbench", 16, 10, BK, () => {
  R(0, 2, 16, 1, ML);
  R(0, 0, 2, 5, MK);
  R(14, 0, 2, 5, MK);
  R(4, 2, 1, 8, MT);
  R(11, 2, 1, 8, MT);
  R(3, 6, 10, 2, RD);
  R(3, 6, 10, 1, RL);
});
add("dumbbells", 14, 10, MK, () => {
  R(0, 0, 1, 10, MT);
  R(13, 0, 1, 10, MT);
  for (const y of [1, 5]) {
    R(0, y + 2, 14, 1, MT);
    for (let x = 2; x < 13; x += 4) {
      R(x, y, 1, 3, BK);
      R(x + 2, y, 1, 3, BK);
      P(x + 1, y + 1, ML);
    }
  }
});
add("punchbag", 6, 18, RD, () => {
  R(3, 0, 1, 4, MK);
  R(1, 4, 5, 13, RD);
  R(1, 4, 1, 13, RL);
  R(1, 4, 5, 1, BK);
  R(1, 16, 5, 1, BK);
}, { hang: true });
add("boxingring", 30, 12, RD, () => {
  R(0, 9, 30, 3, BL);
  R(0, 9, 30, 1, BLL);
  for (const x of [0, 29]) R(x, 0, 1, 9, WH);
  for (const y of [2, 5]) R(1, y, 28, 1, RD);
});
add("cinemaseats", 20, 9, RD, () => {
  for (let x = 0; x < 20; x += 5) {
    R(x, 0, 4, 6, RD);
    R(x, 0, 4, 1, RL);
    R(x, 5, 4, 2, RK);
    R(x + 4, 3, 1, 6, MK);
  }
  R(0, 8, 20, 1, MK);
});
add("projector", 12, 12, MK, () => {
  O(3, 3, 2.6, MK);
  O(8, 3, 2.6, MK);
  B(1, 5, 10, 4, "#3a3f4f", "#5d6578", "#1e1e24");
  R(11, 6, 1, 2, YLL);
  R(4, 9, 4, 3, MK);
}, { lights: [[11, 6, "#fff6c8", 0]] });
add("curtains", 16, 18, RD, () => {
  R(0, 0, 16, 2, YL);
  for (let x = 0; x < 16; x++) R(x, 2, 1, 16 - Math.round(Math.abs(Math.sin(x * 0.6)) * 2), x % 2 ? RD : RK);
}, { hang: true });
add("piano", 18, 14, BK, () => {
  B(0, 0, 14, 10, "#1e1e24", "#3a3a44", "#0d0b14");
  R(0, 8, 18, 2, "#2a2a30");
  for (let x = 1; x < 17; x += 2) P(x, 8, WH);
  R(1, 10, 1, 4, BK);
  R(16, 10, 1, 4, BK);
  R(12, 1, 1, 6, "#c8a040");
});
add("drumkit", 18, 12, RD, () => {
  O(9, 8, 3.5, WH);
  O(9, 8, 2.5, RD);
  R(3, 6, 4, 3, RD);
  R(3, 6, 4, 1, WH);
  R(12, 5, 4, 3, RD);
  R(12, 5, 4, 1, WH);
  R(1, 1, 5, 1, YL);
  R(3, 2, 1, 4, ML);
  R(13, 0, 5, 1, YL);
  R(15, 1, 1, 4, ML);
});
add("arcade", 9, 18, PU, () => {
  B(0, 0, 9, 18, "#3a1a5a", PU, "#1a0a2a");
  R(1, 1, 7, 2, YL);
  R(1, 4, 7, 6, "#1a1a3a");
  R(2, 5, 2, 2, "#8aff6a");
  R(5, 7, 2, 2, PK);
  R(0, 11, 9, 2, MK);
  P(2, 11, RD);
  P(5, 11, YL);
}, { lights: [[3, 6, "#8aff6a", 3], [4, 2, "#ffd23f", 0]] });
add("arcade2", 9, 18, BL, () => {
  B(0, 0, 9, 18, BLK, BL, "#1a2a50");
  R(1, 1, 7, 2, PK);
  R(1, 4, 7, 6, "#1a1a3a");
  R(3, 5, 3, 1, YL);
  R(2, 8, 5, 1, "#5ab0ff");
  R(0, 11, 9, 2, MK);
  P(2, 11, "#5bbf3a");
}, { lights: [[4, 6, "#5ab0ff", 2.5], [4, 2, "#ff6fae", 0]] });
add("clawmachine", 10, 18, PK, () => {
  R(0, 0, 10, 3, PK);
  R(0, 3, 10, 9, "#c8e8f8");
  R(0, 3, 1, 9, PK);
  R(9, 3, 1, 9, PK);
  R(5, 3, 1, 3, MK);
  R(4, 6, 3, 1, ML);
  for (let x = 1; x < 9; x += 2) O(x + 0.5, 10, 1, [YL, BL, GN, RD][x % 4]);
  B(0, 12, 10, 6, PK, "#ff9ac8", "#c8508a");
}, { lights: [[5, 1, "#ff9ac8", 2]] });
add("bowlingpins", 12, 8, WH, () => {
  for (let x = 1; x < 12; x += 3) {
    R(x, 2, 2, 6, WH);
    R(x, 0, 2, 2, WH);
    R(x, 2, 2, 1, RD);
  }
});
add("ballrack", 14, 10, MK, () => {
  R(0, 4, 14, 1, MK);
  R(0, 9, 14, 1, MK);
  legs(0, 0, 14, 10, MK);
  for (let x = 2; x < 13; x += 4) {
    O(x, 2, 1.6, [BL, RD, PU][x % 3]);
    O(x, 7, 1.6, [GN, YL, BK][x % 3]);
  }
});
add("slotmachine", 8, 16, YL, () => {
  B(0, 2, 8, 14, RD, RL, RK);
  R(0, 0, 8, 2, YL);
  R(1, 4, 6, 4, WH);
  P(2, 6, RD);
  P(4, 6, YL);
  P(6, 6, "#5bbf3a");
  R(7, 4, 1, 4, ML);
  R(1, 10, 6, 2, MK);
}, { lights: [[4, 1, "#ffd23f", 3], [4, 6, "#ffffff", 4]] });
add("roulette", 18, 10, GN, () => {
  R(0, 3, 18, 2, "#2a7a3a");
  O(5, 2, 2.4, WK);
  O(5, 2, 1.4, RD);
  for (let x = 9; x < 17; x += 2) R(x, 3, 1, 2, x % 4 ? RD : BK);
  B(1, 5, 16, 5, WD, WL, WK);
});
add("cardtable", 16, 9, GN, () => {
  chair(0, 1, WK);
  R(4, 2, 8, 1, "#2a7a3a");
  R(5, 1, 2, 1, WH);
  R(8, 1, 1, 1, RD);
  R(10, 1, 1, 1, YL);
  R(7, 3, 2, 6, WK);
  R(15, 1, 1, 7, WK);
  R(12, 4, 4, 1, WK);
});

/* ================================================================== */
/* Homes and hotels                                                    */
/* ================================================================== */
add("bed", 18, 9, WH, () => {
  R(0, 0, 2, 9, WK);
  R(16, 3, 2, 6, WK);
  R(2, 3, 14, 3, "#c8d8f0");
  R(2, 2, 4, 2, WH);
  R(6, 3, 10, 3, BL);
  R(6, 3, 10, 1, BLL);
  R(2, 6, 14, 1, WK);
});
add("bunkbed", 16, 16, MT, () => {
  legs(0, 0, 16, 16, MT);
  for (const y of [3, 11]) {
    R(1, y, 14, 2, "#a89868");
    R(1, y, 3, 1, WH);
    R(1, y + 2, 14, 1, MT);
  }
  R(14, 4, 1, 7, MT);
});
add("sofa", 18, 8, BL, () => {
  R(0, 2, 18, 6, BL);
  R(0, 0, 3, 8, BLK);
  R(15, 0, 3, 8, BLK);
  R(3, 2, 12, 1, BLL);
  R(3, 4, 12, 1, BLK);
  P(5, 3, YL);
});
add("armchair", 9, 8, RD, () => {
  R(0, 0, 9, 8, RK);
  R(2, 2, 5, 3, RD);
  R(0, 0, 9, 1, RL);
  R(1, 7, 1, 1, WK);
  R(7, 7, 1, 1, WK);
});
add("tv", 12, 11, MK, () => {
  R(0, 0, 12, 7, MK);
  R(1, 1, 10, 5, "#3a6ab0");
  P(2, 2, "#c8e8ff");
  B(1, 7, 10, 4, WD, WL, WK);
}, { lights: [[6, 3, "#7ab0e8", 3]] });
add("wardrobe", 12, 18, WD, () => {
  B(0, 0, 12, 18, WD, WL, WK);
  R(6, 1, 1, 16, WK);
  P(4, 9, YL);
  P(8, 9, YL);
  R(0, 0, 12, 1, WK);
});
add("toilet", 7, 10, WH, () => {
  B(4, 0, 3, 6, WH, "#ffffff", WG);
  R(0, 5, 7, 2, WH);
  R(1, 7, 4, 3, WG);
  R(0, 5, 5, 1, "#ffffff");
});
add("bathtub", 18, 8, WH, () => {
  R(0, 2, 18, 4, WH);
  R(1, 2, 16, 1, "#a8d8f0");
  R(0, 6, 2, 2, YL);
  R(16, 6, 2, 2, YL);
  R(15, 0, 1, 2, ML);
  R(14, 0, 2, 1, ML);
});
add("shower", 8, 18, GLS, () => {
  R(0, 0, 8, 18, "#c8e8f866");
  R(0, 0, 1, 18, ML);
  R(7, 0, 1, 18, ML);
  R(2, 1, 3, 1, ML);
  for (let y = 3; y < 16; y += 3) P(3 + (y % 2), y, BLL);
  R(0, 17, 8, 1, WG);
});
add("kitchencounter", 18, 10, WD, () => {
  R(0, 3, 18, 2, WG);
  R(2, 0, 3, 3, RD);
  R(6, 1, 2, 2, WH);
  R(12, 0, 1, 3, ML);
  R(11, 0, 3, 1, ML);
  B(0, 5, 18, 5, WD, WL, WK);
  for (const x of [6, 12]) R(x, 6, 1, 3, WK);
});
add("luggagecart", 12, 16, YL, () => {
  R(0, 0, 1, 16, "#c8a040");
  R(11, 0, 1, 16, "#c8a040");
  R(0, 0, 12, 1, "#c8a040");
  R(1, 7, 4, 7, RD);
  R(5, 9, 6, 5, BL);
  R(2, 3, 6, 4, "#8a5a2b");
  R(0, 14, 12, 1, MK);
  wheel(2, 15, 1);
  wheel(9, 15, 1);
});
add("deskbell", 14, 10, WD, () => {
  B(0, 3, 14, 7, WD, WL, WK);
  O(7, 2, 1.6, YL);
  P(7, 0, YL);
  R(2, 1, 3, 2, WH);
});
add("rug", 18, 2, RD, () => {
  R(0, 0, 18, 2, RD);
  for (let x = 1; x < 17; x += 3) P(x, 0, YL);
  R(0, 1, 18, 1, RK);
});
add("crib", 12, 10, PK, () => {
  R(0, 0, 12, 1, "#f0c8d8");
  for (let x = 0; x < 12; x += 2) R(x, 1, 1, 6, "#f0c8d8");
  R(1, 5, 10, 2, WH);
  R(0, 7, 12, 1, "#f0c8d8");
  R(0, 8, 1, 2, "#f0c8d8");
  R(11, 8, 1, 2, "#f0c8d8");
});

/* ================================================================== */
/* Industry and services                                               */
/* ================================================================== */
add("conveyor", 24, 8, MK, () => {
  R(0, 2, 24, 2, BK);
  for (let x = 1; x < 24; x += 3) P(x, 2, MT);
  for (const x of [3, 11, 19]) B(x, 0, 4, 2, WD, WL, WK);
  for (let x = 1; x < 24; x += 5) R(x, 4, 1, 4, MT);
});
add("press", 14, 18, YL, () => {
  B(0, 0, 14, 4, "#c8a020", YL, "#8a7010");
  R(2, 4, 2, 9, MT);
  R(10, 4, 2, 9, MT);
  R(4, 6, 6, 3, MK);
  B(0, 13, 14, 5, "#c8a020", YL, "#8a7010");
  for (let x = 0; x < 14; x += 4) R(x, 17, 2, 1, BK);
});
add("lathe", 18, 10, GN, () => {
  B(0, 1, 6, 6, "#3a6a4a", "#5a8a6a", "#2a4a3a");
  R(6, 3, 10, 1, ML);
  R(15, 2, 3, 3, "#3a6a4a");
  B(0, 7, 18, 3, MK, MT);
});
add("boiler", 14, 18, RD, () => {
  R(1, 0, 3, 4, MK);
  B(0, 4, 14, 14, "#8a3a2a", "#a85a3a", "#5a2014");
  for (const y of [7, 13]) R(0, y, 14, 1, "#5a2014");
  O(7, 10, 2, BK);
  P(7, 10, OR);
  R(11, 5, 2, 2, ML);
}, { smoke: [[2, 0]], lights: [[7, 10, "#ff8a1a", 1]] });
add("controlpanel", 16, 12, MK, () => {
  R(0, 0, 16, 5, "#2a2e3a");
  R(1, 1, 8, 3, "#0a2a1a");
  Ln(1, 3, 8, 2, "#8aff6a");
  for (let x = 11; x < 15; x += 2) P(x, 2, x % 4 ? RD : YL);
  B(0, 5, 16, 7, "#4a5264", "#5d6578", "#2a2e3a");
  for (let x = 2; x < 14; x += 3) R(x, 7, 2, 1, [RD, YL, "#5bbf3a", BL][x % 4]);
}, { lights: [[11, 2, "#ff3b3b", 2], [13, 2, "#ffd23f", 3], [4, 2, "#8aff6a", 0]] });
add("transformer", 14, 14, MT, () => {
  B(1, 3, 12, 11, "#5d6a5a", "#7a8a78", "#3a4a3a");
  for (let x = 2; x < 13; x += 2) R(x, 4, 1, 9, "#4a5a4a");
  for (const x of [3, 7, 11]) {
    R(x, 0, 1, 3, "#c8a040");
    P(x, 0, WH);
  }
  R(4, 8, 6, 3, YL);
  P(7, 9, BK);
}, { lights: [[7, 0, "#c8f0ff", 5]] });
add("turbine", 20, 16, ML, () => {
  O(8, 8, 7, "#7a8294");
  O(8, 8, 5, MT);
  O(8, 8, 2, ML);
  R(14, 6, 6, 5, "#7a8294");
  R(2, 14, 14, 2, MK);
});
add("brewvat", 14, 18, "#c8a040", () => {
  for (let y = 0; y < 3; y++) R(3 - y, y, 8 + y * 2, 1, "#c8a040");
  B(0, 3, 14, 12, "#c8a040", "#f0d070", "#8a6a20");
  R(0, 8, 14, 1, "#8a6a20");
  R(6, 0, 2, 1, MK);
  legs(1, 15, 12, 3, MK);
});
add("woodbarrels", 14, 12, WD, () => {
  for (const [x, y] of [[0, 6], [7, 6], [3, 0]]) {
    R(x, y, 7, 6, WD);
    R(x, y, 7, 1, WL);
    R(x, y + 2, 7, 1, MK);
    R(x, y + 4, 7, 1, MK);
    R(x + 6, y, 1, 6, WK);
  }
});
add("sawblade", 14, 12, ML, () => {
  O(7, 5, 5, ML);
  for (let a = 0; a < 12; a++) P(Math.round(7 + Math.cos(a * 0.52) * 5.5), Math.round(5 + Math.sin(a * 0.52) * 5.5), "#c8ced9");
  O(7, 5, 1.5, MK);
  B(0, 8, 14, 4, WD, WL, WK);
});
add("printpress", 20, 14, MK, () => {
  B(0, 4, 20, 10, "#3a3f4f", "#5d6578", "#1e1e24");
  O(5, 4, 3.5, MT);
  O(14, 4, 3.5, MT);
  R(2, 6, 16, 1, WH);
  R(0, 12, 20, 2, BK);
});
add("paperrolls", 16, 10, WH, () => {
  for (const [x, y] of [[2, 5], [10, 5], [6, 0]] as const) {
    O(x + 2.5, y + 2.5, 2.6, WH);
    O(x + 2.5, y + 2.5, 0.8, WG);
  }
});
add("bales", 16, 10, "#8a7a5a", () => {
  for (const [x, y, col] of [[0, 5, "#8a7a5a"], [8, 5, "#6a7a8a"], [4, 0, "#7a6a4a"]] as const) {
    R(x, y, 8, 5, col);
    R(x, y + 1, 8, 1, MK);
    R(x, y + 3, 8, 1, MK);
    P(x + 2, y + 2, RD);
    P(x + 5, y + 2, BL);
  }
});
add("carlift", 26, 16, YL, () => {
  R(1, 0, 2, 16, "#c8a020");
  R(23, 0, 2, 16, "#c8a020");
  car(22, 9, BL, BLL);
  c.translate(0, 0);
  R(0, 9, 26, 1, MK);
  R(0, 15, 26, 1, MK);
});
add("toolchest", 10, 12, RD, () => {
  B(0, 0, 10, 12, RD, RL, RK);
  for (const y of [2, 4, 6, 8]) R(1, y, 8, 1, RK);
  for (const y of [1, 3, 5, 7, 9]) R(4, y, 2, 1, ML);
  wheel(2, 11, 1);
  wheel(7, 11, 1);
});
add("workbench", 18, 11, WD, () => {
  R(0, 3, 18, 2, WD);
  R(0, 3, 18, 1, WL);
  legs(1, 5, 16, 6, WK);
  R(1, 8, 16, 1, WK);
  R(2, 1, 4, 2, MT);
  R(10, 0, 1, 3, RD);
  R(13, 2, 3, 1, ML);
});
add("engine", 14, 10, MK, () => {
  B(1, 2, 12, 7, "#4a5264", "#5d6578", "#2a2e3a");
  for (let x = 2; x < 12; x += 3) R(x, 0, 2, 2, MT);
  O(3, 6, 2, BK);
  R(9, 4, 3, 3, RD);
  R(0, 9, 14, 1, MK);
});
add("tirerack", 14, 14, BK, () => {
  legs(0, 0, 14, 14, MT);
  for (const y of [1, 8]) for (const x of [1, 7]) {
    R(x, y, 6, 5, BK);
    R(x + 2, y + 1, 2, 3, MT);
  }
});
add("carwash", 16, 18, BL, () => {
  R(0, 0, 16, 2, BL);
  legs(0, 0, 16, 18, MT);
  for (const x of [3, 11]) for (let y = 3; y < 17; y++) R(x - 1 + (y % 2), y, 3, 1, y % 2 ? "#5ab0ff" : RD);
});
add("ticketmachine", 7, 15, OR, () => {
  B(0, 0, 7, 15, OR, YL, "#a85a10");
  R(1, 2, 5, 3, "#1a2a3a");
  R(1, 6, 5, 2, MK);
  R(2, 10, 3, 1, BK);
}, { lights: [[3, 3, "#7ad0ff", 0]] });
add("departureboard", 22, 10, BK, () => {
  R(0, 0, 22, 9, "#14161e");
  for (let y = 1; y < 8; y += 2) {
    R(1, y, 9, 1, YL);
    R(12, y, 4, 1, YL);
    R(18, y, 3, 1, y === 3 ? RD : "#8aff6a");
  }
  R(10, 9, 2, 1, MK);
}, { hang: true, lights: [[5, 1, "#ffd23f", 0], [19, 3, "#ff3b3b", 1.5]] });
add("radarconsole", 16, 12, GN, () => {
  R(0, 0, 16, 7, "#2a2e3a");
  O(5, 3, 2.8, "#0a2a1a");
  Ln(5, 3, 7, 1, "#8aff6a");
  P(3, 4, "#8aff6a");
  R(10, 1, 5, 4, "#0a1a2a");
  B(0, 7, 16, 5, "#4a5a3a", "#6a7a4a", "#2a3a20");
}, { lights: [[5, 3, "#8aff6a", 1]] });
add("weaponrack", 14, 14, WK, () => {
  R(0, 0, 14, 14, WK);
  R(0, 3, 14, 1, MK);
  for (let x = 1; x < 14; x += 3) {
    R(x, 2, 1, 10, BK);
    R(x, 9, 2, 3, "#5a3816");
  }
});
add("maptable", 18, 10, WD, () => {
  R(0, 2, 18, 2, "#c8b888");
  R(3, 2, 4, 1, GN);
  R(9, 2, 3, 1, BL);
  P(13, 2, RD);
  P(6, 2, RD);
  legs(1, 4, 16, 6, WK);
  R(1, 0, 3, 2, YL);
});

/* ================================================================== */
/* Streets                                                             */
/* ================================================================== */
add("trafficlight", 5, 26, MK, () => {
  R(2, 9, 1, 17, MK);
  B(0, 0, 5, 10, "#2a2e3a", "#3a3f4f");
  P(2, 2, RD);
  P(2, 5, YL);
  P(2, 8, "#5bbf3a");
  R(0, 25, 5, 1, MK);
}, { lights: [[2, 2, "#ff3b3b", 0.25], [2, 5, "#ffd23f", 0.33], [2, 8, "#5bff6a", 0.2]] });
add("stopsign", 7, 18, RD, () => {
  R(3, 6, 1, 12, ML);
  O(3, 3, 3.2, WH);
  O(3, 3, 2.5, RD);
  R(1, 3, 5, 1, WH);
});
add("busstop", 22, 18, GLS, () => {
  R(0, 0, 22, 2, MT);
  R(0, 2, 1, 16, MT);
  R(21, 2, 1, 16, MT);
  R(1, 2, 20, 12, "#c8e8f855");
  R(14, 3, 6, 8, YL);
  R(15, 4, 4, 1, BK);
  R(3, 12, 10, 1, WD);
  legs(3, 13, 10, 5, MK);
});
add("bench", 14, 7, WD, () => {
  R(0, 0, 14, 1, WD);
  R(0, 2, 14, 1, WD);
  R(0, 4, 14, 1, WL);
  for (const x of [1, 12]) R(x, 0, 1, 7, MK);
});
add("trashbin", 6, 9, GN, () => {
  R(0, 0, 6, 2, GNK);
  B(0, 2, 6, 7, GN, GNL, GNK);
  for (let x = 1; x < 6; x += 2) R(x, 3, 1, 5, GNK);
});
add("dumpster", 20, 12, GN, () => {
  R(0, 0, 20, 2, "#2a5a3a");
  B(0, 2, 20, 9, "#3a7a4a", "#5a9a6a", "#2a5a3a");
  R(2, 5, 6, 2, WH);
  R(0, 11, 2, 1, BK);
  R(18, 11, 2, 1, BK);
});
add("mailbox", 7, 14, BL, () => {
  R(0, 0, 7, 3, BL);
  R(1, -1, 5, 1, BL);
  B(0, 2, 7, 8, BL, BLL, BLK);
  R(2, 4, 3, 1, BK);
  R(1, 10, 1, 4, MK);
  R(5, 10, 1, 4, MK);
});
add("newsbox", 6, 11, YL, () => {
  B(0, 0, 6, 9, YL, YLL, "#a88a10");
  R(1, 2, 4, 3, WH);
  R(1, 3, 4, 1, BK);
  R(1, 9, 1, 2, MK);
  R(4, 9, 1, 2, MK);
});
add("parkingmeter", 4, 14, ML, () => {
  O(2, 2, 2, ML);
  R(1, 1, 2, 2, "#3a3f4f");
  R(1, 4, 2, 10, MK);
});
add("powerpole", 14, 46, WK, () => {
  R(6, 0, 2, 46, WK);
  R(0, 3, 14, 1, WK);
  R(1, 8, 12, 1, WK);
  for (const x of [0, 13, 3, 10]) P(x, 2, WH);
  B(8, 12, 5, 7, MT, ML, MK);
  Ln(13, 3, 13, 12, BK);
});
add("utilitybox", 9, 10, "#6a7a6a", () => {
  B(0, 0, 9, 10, "#6a7a6a", "#8a9a8a", "#4a5a4a");
  R(4, 1, 1, 8, "#4a5a4a");
  R(1, 6, 2, 2, YL);
});
add("bikerack", 16, 8, ML, () => {
  for (let x = 1; x < 16; x += 5) {
    R(x, 1, 1, 7, ML);
    R(x, 1, 3, 1, ML);
    R(x + 2, 1, 1, 7, ML);
  }
  R(0, 7, 16, 1, MK);
});
add("bicycle", 14, 9, RD, () => {
  O(3, 6, 2.5, BK);
  O(3, 6, 1.5, "#00000000");
  O(10, 6, 2.5, BK);
  P(3, 6, GY);
  P(10, 6, GY);
  Ln(3, 6, 6, 2, RD);
  Ln(6, 2, 10, 6, RD);
  Ln(6, 6, 9, 2, RD);
  R(5, 1, 3, 1, BK);
  R(9, 1, 2, 1, MK);
});
add("treeplanter", 12, 22, GN, () => {
  O(6, 6, 5.5, GN);
  O(4, 4, 2.5, GNL);
  R(5, 11, 2, 7, "#7a5230");
  B(1, 17, 10, 5, GY, WG, MT);
});
add("hedge", 16, 8, GN, () => {
  for (let x = 2; x < 16; x += 4) O(x, 3, 3, GN);
  R(0, 3, 16, 5, GN);
  for (const [x, y] of [[2, 2], [7, 1], [12, 3], [5, 5]]) P(x, y, GNL);
});
add("fountain", 26, 16, BLL, () => {
  R(12, 2, 2, 8, WG);
  O(13, 2, 2, "#c8e8f8");
  for (const x of [8, 18]) Ln(13, 1, x, 8, "#7ad0ff");
  R(5, 6, 16, 2, WG);
  R(0, 11, 26, 5, GY);
  R(1, 11, 24, 2, "#5ab0e0");
  R(0, 10, 26, 1, WG);
}, { lights: [[13, 1, "#c8f0ff", 2]] });
add("statue", 12, 26, "#6a9a8a", () => {
  const S = "#6a9a8a";
  O(6, 2, 1.8, S);
  R(4, 4, 4, 7, S);
  R(8, 0, 1, 6, S);
  R(2, 5, 2, 4, S);
  R(4, 11, 1, 5, S);
  R(7, 11, 1, 5, S);
  B(1, 16, 10, 10, WG, "#e8e4dc", GY);
  R(3, 20, 6, 1, GY);
});
add("flagpole", 14, 30, ML, () => {
  R(0, 0, 1, 30, ML);
  for (let x = 1; x < 13; x++) {
    const o = Math.round(Math.sin(x * 0.6));
    R(x, 1 + o, 1, 3, BL);
    R(x, 4 + o, 1, 3, WH);
    R(x, 7 + o, 1, 2, "#5bbf3a");
  }
  R(0, 29, 3, 1, MK);
});
add("streetclock", 8, 26, BK, () => {
  O(4, 4, 3.6, BK);
  O(4, 4, 2.8, WH);
  R(4, 2, 1, 3, BK);
  R(4, 4, 2, 1, BK);
  R(3, 8, 2, 18, BK);
  R(1, 25, 6, 1, BK);
});
add("infoboard", 14, 18, BL, () => {
  R(1, 10, 1, 8, MK);
  R(12, 10, 1, 8, MK);
  R(0, 0, 14, 11, BL);
  R(1, 1, 12, 9, WH);
  R(2, 2, 6, 4, GN);
  R(3, 3, 2, 2, BL);
  P(6, 4, RD);
  R(9, 2, 3, 1, BK);
  R(9, 4, 3, 1, BK);
});
add("hotdogcart", 20, 16, RD, () => {
  R(2, 0, 16, 2, YL);
  for (let x = 2; x < 18; x += 4) R(x, 0, 2, 2, RD);
  R(3, 2, 1, 5, ML);
  R(16, 2, 1, 5, ML);
  B(1, 7, 18, 6, RD, RL, RK);
  R(4, 8, 8, 2, "#c88a40");
  wheel(5, 14);
  wheel(15, 14);
});
add("marketstall", 22, 16, "#c8a040", () => {
  R(0, 0, 22, 3, GN);
  for (let x = 0; x < 22; x += 4) R(x, 0, 2, 3, WH);
  R(1, 3, 1, 13, WK);
  R(20, 3, 1, 13, WK);
  R(1, 9, 20, 2, WD);
  for (let x = 3; x < 20; x += 3) O(x, 8, 1.3, [RD, OR, YL, GN][x % 4]);
  R(2, 11, 18, 5, "#6e4520");
});
add("umbrellatable", 14, 16, RD, () => {
  for (let y = 0; y < 4; y++) R(6 - y * 2, y, 2 + y * 4, 1, y % 2 ? WH : RD);
  R(7, 4, 1, 12, ML);
  R(3, 10, 9, 1, WH);
  legs(3, 11, 9, 5, ML);
});
add("picnictable", 18, 8, WD, () => {
  R(2, 2, 14, 1, WL);
  R(0, 5, 18, 1, WD);
  Ln(4, 3, 2, 7, WK);
  Ln(14, 3, 16, 7, WK);
});
add("slide", 22, 18, RD, () => {
  R(0, 0, 1, 18, YL);
  R(5, 0, 1, 18, YL);
  for (let y = 2; y < 18; y += 3) R(0, y, 6, 1, YL);
  R(0, 0, 6, 1, RD);
  Ln(6, 1, 21, 16, RD);
  Ln(6, 2, 21, 17, RD);
});
add("swings", 22, 18, BL, () => {
  Ln(0, 17, 3, 0, BL);
  Ln(21, 17, 18, 0, BL);
  R(3, 0, 16, 1, BL);
  for (const x of [7, 14]) {
    R(x, 1, 1, 10, GY);
    R(x + 2, 1, 1, 10, GY);
    R(x, 11, 3, 1, RD);
  }
});
add("hoop", 10, 26, OR, () => {
  R(2, 4, 2, 22, MT);
  R(3, 0, 7, 6, WH);
  R(5, 2, 3, 2, RD);
  R(6, 6, 4, 1, OR);
  for (let k = 0; k < 3; k++) P(6 + k, 7 + k, WH);
});
add("roadworks", 10, 12, OR, () => {
  for (let y = 0; y < 6; y++) R(5 - Math.floor(y / 1.5), y, Math.floor(y / 0.75) + 1, 1, y % 3 ? OR : WH);
  R(4, 2, 1, 3, BK);
  R(4, 6, 2, 6, MK);
  R(1, 11, 8, 1, MK);
});
add("trafficbarrels", 14, 9, OR, () => {
  for (const x of [0, 7]) {
    R(x, 0, 7, 9, OR);
    R(x, 2, 7, 2, WH);
    R(x, 6, 7, 1, WH);
    R(x, 8, 7, 1, BK);
  }
});
add("jerseybarrier", 16, 7, WG, () => {
  for (let y = 0; y < 7; y++) R(y < 3 ? 3 : 3 - (y - 2), y, y < 3 ? 10 : 10 + (y - 2) * 2, 1, y % 3 ? WG : GY);
  R(4, 4, 8, 1, RD);
});
add("bollards", 14, 7, YL, () => {
  for (const x of [0, 5, 10]) {
    R(x, 0, 3, 7, MK);
    R(x, 1, 3, 2, YL);
  }
});
add("scaffold", 24, 30, MT, () => {
  for (const x of [0, 11, 22]) R(x, 0, 2, 30, ML);
  for (let y = 0; y < 30; y += 10) {
    R(0, y, 24, 2, WD);
    Ln(1, y + 2, 11, y + 9, MT);
    Ln(12, y + 9, 22, y + 2, MT);
  }
});
add("cementmixer", 26, 16, OR, () => {
  car(10, 12, OR, YL, { cab: [0, 8] });
  c.save();
  c.translate(9, 0);
  for (let y = 1; y < 10; y++) R(Math.abs(5 - y), y, 16 - Math.abs(5 - y) * 2, 1, y % 3 ? "#d8d4cc" : OR);
  R(0, 10, 17, 2, MK);
  c.restore();
  wheel(13, 14);
  wheel(21, 14);
});
add("crane", 40, 60, YL, () => {
  for (let y = 4; y < 60; y += 4) {
    R(18, y, 4, 1, "#c8a020");
    Ln(18, y, 21, y + 4, "#c8a020");
  }
  R(18, 4, 1, 56, YL);
  R(21, 4, 1, 56, YL);
  R(0, 2, 40, 2, YL);
  for (let x = 0; x < 40; x += 4) Ln(x, 2, x + 4, 0, "#c8a020");
  R(19, 0, 2, 2, YL);
  R(2, 4, 5, 5, MK);
  R(34, 4, 1, 16, BK);
  R(32, 20, 5, 3, MK);
  R(14, 56, 12, 4, MK);
}, { lights: [[20, 0, "#ff3b3b", 1]] });

/* ================================================================== */
/* Vehicles                                                            */
/* ================================================================== */
add("sedan", 26, 11, RD, () => car(26, 11, RD, RL));
add("sedanblue", 26, 11, BL, () => car(26, 11, BL, BLL));
add("hatchback", 22, 11, "#5bbf3a", () => car(22, 11, "#4a9a3a", "#6aba5a", { cab: [5, 14] }));
add("taxi", 26, 13, YL, () => {
  c.save();
  c.translate(0, 2);
  car(26, 11, YL, YLL);
  R(0, 6, 26, 1, BK);
  for (let x = 1; x < 26; x += 4) R(x, 6, 2, 1, WH);
  c.restore();
  R(11, 0, 4, 2, WH);
  R(12, 1, 2, 1, RD);
});
add("policecar", 26, 13, WH, () => {
  c.save();
  c.translate(0, 2);
  car(26, 11, WH, "#ffffff");
  R(0, 6, 26, 2, BK);
  R(10, 6, 6, 1, YL);
  c.restore();
  R(10, 0, 3, 2, RD);
  R(13, 0, 3, 2, BL);
}, { lights: [[11, 0, "#ff3b3b", 4], [14, 0, "#5ab0ff", 4]] });
add("ambulance", 30, 16, WH, () => {
  R(0, 2, 22, 11, WH);
  R(0, 2, 22, 1, "#ffffff");
  R(22, 6, 8, 7, WH);
  R(23, 7, 5, 3, GLS);
  R(0, 8, 30, 2, RD);
  R(8, 4, 5, 1, RD);
  R(10, 3, 1, 3, RD);
  R(2, 0, 4, 2, RD);
  R(16, 0, 4, 2, BL);
  wheel(5, 14);
  wheel(24, 14);
}, { lights: [[3, 0, "#ff3b3b", 4], [17, 0, "#5ab0ff", 4]] });
add("van", 28, 15, WH, () => {
  R(0, 0, 22, 12, WH);
  R(0, 0, 22, 1, "#ffffff");
  R(22, 3, 6, 9, WH);
  R(23, 4, 4, 3, GLS);
  R(3, 4, 12, 3, BL);
  R(0, 11, 28, 2, MK);
  wheel(5, 13);
  wheel(23, 13);
});
add("deliverytruck", 36, 20, "#8a5a2b", () => {
  R(0, 0, 26, 15, "#c8a060");
  R(0, 0, 26, 1, "#e0c080");
  R(4, 5, 14, 4, RD);
  R(26, 6, 10, 9, BL);
  R(27, 7, 6, 3, GLS);
  R(0, 15, 36, 2, MK);
  wheel(5, 18);
  wheel(30, 18);
  wheel(10, 18);
});
add("pickup", 28, 12, "#8a4a2a", () => {
  R(0, 4, 14, 4, "#8a4a2a");
  R(0, 4, 14, 1, "#a86a40");
  R(14, 0, 8, 8, "#8a4a2a");
  R(15, 1, 6, 3, GLS);
  R(22, 4, 6, 4, "#8a4a2a");
  R(1, 2, 3, 2, "#c8a060");
  R(0, 8, 28, 2, MK);
  wheel(5, 10);
  wheel(22, 10);
});
add("bus", 50, 18, OR, () => {
  R(0, 0, 50, 14, OR);
  R(0, 0, 50, 1, YL);
  for (let x = 2; x < 46; x += 6) R(x, 2, 5, 5, GLS);
  R(46, 2, 3, 9, GLS);
  R(0, 9, 50, 1, WH);
  R(0, 14, 50, 1, MK);
  wheel(8, 16);
  wheel(41, 16);
});
add("icecreamvan", 30, 17, PK, () => {
  R(0, 2, 22, 11, "#f8f0f4");
  R(0, 2, 22, 2, PK);
  R(3, 5, 10, 5, "#3a2a3a");
  R(4, 6, 3, 3, YLL);
  R(8, 6, 3, 3, PK);
  R(22, 6, 8, 7, "#f8f0f4");
  R(23, 7, 5, 3, GLS);
  R(14, 0, 3, 2, PK);
  O(15, 0, 1.2, "#8a5a2b");
  R(0, 13, 30, 1, MK);
  wheel(5, 15);
  wheel(24, 15);
});
add("firetruck", 44, 20, RD, () => {
  R(0, 6, 34, 10, RD);
  R(0, 6, 34, 1, RL);
  R(34, 4, 10, 12, RD);
  R(35, 5, 6, 4, GLS);
  for (let x = 1; x < 32; x += 3) Ln(x, 3, x + 2, 0, ML);
  R(0, 3, 32, 1, ML);
  R(0, 0, 32, 1, ML);
  R(2, 10, 30, 1, WH);
  R(36, 2, 4, 2, RD);
  wheel(6, 18);
  wheel(38, 18);
  wheel(12, 18);
}, { lights: [[37, 2, "#ff3b3b", 4]] });
add("motorbike", 16, 10, BK, () => {
  O(3, 7, 2.5, BK);
  O(12, 7, 2.5, BK);
  P(3, 7, GY);
  P(12, 7, GY);
  R(4, 3, 8, 3, RD);
  R(5, 2, 4, 1, BK);
  Ln(11, 3, 13, 0, ML);
  R(4, 5, 7, 1, MK);
});
add("scooter", 12, 10, "#5ab0b0", () => {
  O(2, 8, 1.6, BK);
  O(9, 8, 1.6, BK);
  R(0, 4, 6, 4, "#5ab0b0");
  R(1, 3, 4, 1, BK);
  R(8, 0, 1, 7, "#5ab0b0");
  R(7, 0, 3, 1, MK);
  R(5, 6, 4, 1, "#5ab0b0");
});
add("tractor", 26, 18, RD, () => {
  R(4, 4, 12, 8, RD);
  R(14, 0, 1, 6, MK);
  R(4, 0, 8, 1, RD);
  R(4, 1, 1, 4, MK);
  R(11, 1, 1, 4, MK);
  R(16, 7, 8, 5, RD);
  R(16, 7, 8, 1, RL);
  O(7, 13, 5, BK);
  O(7, 13, 2, YL);
  O(21, 15, 3, BK);
  P(21, 15, YL);
}, { smoke: [[14, 0]] });
add("forklift", 20, 16, YL, () => {
  R(2, 4, 10, 8, YL);
  R(2, 4, 10, 1, YLL);
  R(3, 0, 1, 5, MK);
  R(10, 0, 1, 5, MK);
  R(3, 0, 8, 1, MK);
  R(13, 0, 2, 13, MK);
  R(15, 11, 5, 1, MK);
  R(15, 9, 4, 2, WD);
  wheel(5, 13);
  wheel(11, 13);
});
add("schoolbus", 46, 18, YL, () => {
  R(0, 0, 46, 14, YL);
  R(0, 0, 46, 1, YLL);
  for (let x = 2; x < 40; x += 5) R(x, 2, 4, 4, GLS);
  R(40, 4, 6, 10, YL);
  R(41, 5, 4, 3, GLS);
  R(0, 8, 46, 1, BK);
  R(0, 11, 46, 1, BK);
  R(0, 14, 46, 1, MK);
  wheel(7, 16);
  wheel(37, 16);
});
add("limo", 40, 11, BK, () => car(40, 11, "#1e1e24", "#3a3a44", { cab: [8, 26], win: "#3a4a5a" }));

/* ================================================================== */
/* Docks, ports and the coast                                          */
/* ================================================================== */
const container = (col: string, hi: string, lo: string) => () => {
  B(0, 0, 32, 14, col, hi, lo);
  for (let x = 2; x < 31; x += 2) R(x, 1, 1, 12, lo);
  R(28, 2, 1, 10, MK);
};
add("containerred", 32, 14, RD, container(RD, RL, RK));
add("containerblue", 32, 14, BL, container(BL, BLL, BLK));
add("containergreen", 32, 14, GN, container("#3a7a4a", "#5a9a6a", "#2a5a3a"));
add("containeryellow", 32, 14, YL, container("#d8a020", YL, "#8a6a10"));
add("containerstack", 32, 28, BL, () => {
  container(BL, BLL, BLK)();
  c.save();
  c.translate(0, 14);
  container(RD, RL, RK)();
  c.restore();
});
add("dockcrane", 44, 64, RD, () => {
  for (const x of [4, 20]) {
    R(x, 18, 2, 46, RD);
    for (let y = 22; y < 62; y += 8) Ln(x, y, x === 4 ? 20 : 4, y + 8, RK);
  }
  R(0, 14, 44, 4, RD);
  R(0, 14, 44, 1, RL);
  for (let x = 0; x < 44; x += 4) Ln(x, 14, x + 2, 8, RK);
  R(0, 8, 44, 1, RK);
  B(6, 18, 10, 6, WH, "#ffffff", WG);
  R(7, 19, 8, 2, GLS);
  R(36, 18, 1, 14, BK);
  R(33, 32, 7, 3, MK);
}, { lights: [[22, 8, "#ff3b3b", 1]] });
add("capstan", 8, 7, MK, () => {
  R(1, 0, 6, 2, MK);
  R(2, 2, 4, 3, MT);
  R(0, 5, 8, 2, MK);
});
add("lifering", 8, 8, RD, () => {
  O(4, 4, 3.6, RD);
  O(4, 4, 1.8, "#00000000");
  c.clearRect(3, 3, 3, 3);
  P(0, 4, WH);
  P(7, 4, WH);
  P(4, 0, WH);
  P(4, 7, WH);
}, { wall: true });
add("buoy", 8, 14, RD, () => {
  R(3, 0, 2, 3, MK);
  P(3, 0, YL);
  for (let y = 3; y < 11; y++) R(4 - Math.floor((y - 2) / 2), y, Math.floor((y - 2) / 2) * 2 + 1, 1, y % 4 < 2 ? RD : WH);
  R(0, 11, 8, 3, RD);
}, { lights: [[3, 0, "#ffd23f", 1]] });
add("anchor", 12, 14, MK, () => {
  O(6, 1.5, 1.5, MK);
  R(5, 3, 2, 9, MK);
  R(2, 4, 8, 1, MK);
  Ln(0, 9, 5, 13, MK);
  Ln(12, 9, 7, 13, MK);
  R(0, 8, 2, 2, MK);
  R(10, 8, 2, 2, MK);
});
add("fishingnets", 16, 10, "#8a9a7a", () => {
  for (let y = 0; y < 10; y += 2) for (let x = (y >> 1) % 2; x < 16 - y / 2; x += 2) P(x + y / 4, y, "#8a9a7a");
  R(0, 8, 16, 2, "#6a7a5a");
  for (const x of [3, 9, 13]) O(x, 6, 1, OR);
});
add("pallets", 16, 9, WD, () => {
  for (let y = 0; y < 9; y += 3) {
    R(0, y, 16, 1, WL);
    for (const x of [0, 7, 14]) R(x, y + 1, 2, 2, WK);
  }
});
add("lpgtank", 34, 18, WH, () => {
  O(6, 7, 6, WH);
  O(27, 7, 6, WH);
  R(6, 1, 21, 13, WH);
  R(6, 1, 21, 1, "#ffffff");
  R(6, 12, 21, 2, WG);
  R(10, 6, 14, 2, RD);
  R(15, 0, 4, 1, MT);
  for (const x of [6, 26]) R(x, 14, 3, 4, MT);
}, { boom: 46 });
add("gascylinders", 14, 16, GN, () => {
  for (let x = 0; x < 14; x += 3) {
    R(x, 2, 3, 14, ["#3a7a4a", WH, "#3a6ab0", RD, YL][(x / 3) % 5]);
    R(x, 2, 1, 14, "#ffffff55");
    R(x + 1, 0, 1, 2, MK);
  }
  R(0, 8, 14, 1, MK);
}, { boom: 26 });
add("propanebottles", 12, 9, WH, () => {
  for (const x of [0, 4, 8]) {
    O(x + 2, 5, 2, ["#d8dce4", "#c8302a", "#3a6ab0"][x / 4]);
    R(x, 5, 4, 4, ["#d8dce4", "#c8302a", "#3a6ab0"][x / 4]);
    R(x + 1, 1, 2, 2, MK);
  }
}, { boom: 20 });
add("oiltank", 30, 26, WG, () => {
  for (let y = 0; y < 4; y++) R(4 - y, y, 22 + y * 2, 1, "#a8a8b0");
  B(0, 4, 30, 22, "#c8c8cc", "#e8e8ec", "#8a8a92");
  for (let y = 8; y < 26; y += 6) R(0, y, 30, 1, "#a8a8b0");
  Ln(28, 4, 22, 26, MK);
  R(6, 12, 12, 4, BL);
}, { boom: 40 });
add("lighthousebig", 18, 60, RD, () => {
  for (let y = 12; y < 60; y++) {
    const half = Math.round(4 + (y - 12) * 0.08);
    R(9 - half, y, half * 2, 1, Math.floor(y / 8) % 2 ? RD : WH);
  }
  R(4, 4, 10, 8, MK);
  R(5, 5, 8, 6, YLL);
  R(3, 2, 12, 2, RD);
  R(7, 0, 4, 2, RD);
  R(2, 12, 14, 1, MK);
}, { lights: [[9, 7, "#fff6c8", 0]] });
add("pierlamp", 5, 18, MK, () => {
  R(2, 4, 1, 14, MK);
  R(0, 0, 5, 4, MK);
  R(1, 1, 3, 2, YLL);
}, { lights: [[2, 2, "#ffd479", 0]] });

/* ================================================================== */
/* Farms                                                               */
/* ================================================================== */
add("haybales", 18, 10, YL, () => {
  for (const [x, y] of [[0, 5], [9, 5], [4, 0]] as const) {
    R(x, y, 9, 5, "#d8b850");
    for (let k = 1; k < 9; k += 2) R(x + k, y, 1, 5, "#c8a040");
    R(x, y, 9, 1, "#e8d070");
  }
});
add("scarecrow", 14, 22, CR, () => {
  R(4, 0, 6, 2, WK);
  R(3, 2, 8, 1, WK);
  O(7, 5, 2, CR);
  P(6, 5, BK);
  P(8, 5, BK);
  R(0, 8, 14, 2, RD);
  R(4, 8, 6, 6, "#4a6ab0");
  R(6, 14, 2, 8, WK);
});
add("chickencoop", 18, 14, WD, () => {
  for (let y = 0; y < 4; y++) R(4 - y * 2 + 4, y, 2 + y * 4, 1, RD);
  B(1, 4, 16, 8, WD, WL, WK);
  R(5, 7, 4, 5, "#3a2410");
  O(13, 9, 1.4, WH);
  P(14, 8, RD);
  legs(1, 12, 16, 2, WK);
});
add("trough", 14, 6, WD, () => {
  R(0, 0, 14, 4, WD);
  R(1, 0, 12, 1, "#5ab0e0");
  R(1, 4, 2, 2, WK);
  R(11, 4, 2, 2, WK);
});
add("milkcans", 12, 10, ML, () => {
  for (const x of [0, 6]) {
    R(x + 1, 0, 4, 2, ML);
    B(x, 2, 6, 8, "#c8ced9", "#e8eef6", "#8a90a0");
  }
});
add("veggies", 20, 6, GN, () => {
  R(0, 4, 20, 2, "#5a3a20");
  for (let x = 1; x < 20; x += 3) {
    R(x, 1, 1, 3, GN);
    P(x - 1, 1, GNL);
    P(x + 1, 2, GNL);
    if (x % 2) P(x, 3, OR);
  }
});
add("farmsilo", 18, 50, ML, () => {
  O(9, 6, 8, "#8a90a0");
  R(1, 6, 16, 44, "#c8ced9");
  for (let y = 10; y < 50; y += 6) R(1, y, 16, 1, "#a8aeb8");
  R(1, 6, 2, 44, "#e8eef6");
  R(13, 12, 2, 38, "#8a90a0");
});
add("windmill", 30, 48, WH, () => {
  for (let y = 16; y < 48; y++) {
    const half = Math.round(3 + (y - 16) * 0.12);
    R(15 - half, y, half * 2, 1, WH);
  }
  R(13, 38, 4, 10, WD);
  O(15, 14, 2, MK);
  Ln(15, 14, 2, 1, WD);
  Ln(15, 14, 28, 27, WD);
  Ln(15, 14, 2, 27, WD);
  Ln(15, 14, 28, 1, WD);
  for (const [x, y] of [[5, 4], [25, 24], [5, 24], [25, 4]]) R(x - 1, y - 1, 3, 3, WG);
});
add("greenhouseplants", 20, 10, GN, () => {
  R(0, 6, 20, 4, WD);
  for (let x = 1; x < 20; x += 3) {
    O(x + 1, 3, 2, x % 2 ? GN : GNL);
    if (x % 2) P(x + 1, 2, RD);
  }
});

/* ================================================================== */
/* Rooftops                                                            */
/* ================================================================== */
add("acunit", 12, 9, WG, () => {
  B(0, 0, 12, 9, WG, WH, GY);
  O(5, 4, 3, MT);
  for (let a = 0; a < 4; a++) Ln(5, 4, Math.round(5 + Math.cos(a * 1.57) * 3), Math.round(4 + Math.sin(a * 1.57) * 3), MK);
  R(10, 2, 1, 5, GY);
});
add("roofwatertank", 16, 22, WD, () => {
  for (let y = 0; y < 3; y++) R(4 - y, y, 8 + y * 2, 1, WK);
  B(1, 3, 14, 11, WD, WL, WK);
  R(1, 6, 14, 1, MK);
  R(1, 11, 14, 1, MK);
  for (const x of [2, 7, 13]) R(x, 14, 1, 8, MK);
  Ln(2, 15, 13, 21, MK);
});
add("helipad", 30, 3, YL, () => {
  R(0, 1, 30, 2, MK);
  R(10, 0, 1, 1, YL);
  R(19, 0, 1, 1, YL);
  R(11, 0, 8, 1, WH);
});
add("chimney", 8, 14, "#a24c36", () => {
  B(0, 2, 8, 12, "#a24c36", "#b95f45", "#6a2e20");
  for (let y = 4; y < 14; y += 3) R(0, y, 8, 1, "#6a2e20");
  R(0, 0, 8, 2, "#4e2a20");
}, { smoke: [[4, 0]] });
add("vent", 6, 8, ML, () => {
  R(0, 0, 6, 2, MT);
  R(1, 2, 4, 6, ML);
  R(1, 4, 4, 1, MT);
}, { smoke: [[3, 0]] });
add("solarpanels", 20, 8, BL, () => {
  for (const x of [0, 10]) {
    Ln(x, 6, x + 9, 1, MT);
    for (let k = 0; k < 6; k++) R(x + k + 1, 5 - Math.floor(k / 2) * 2 + 0, 2, 2, BLK);
    R(x + 8, 2, 1, 6, MK);
  }
});
add("neonsign", 22, 10, PK, () => {
  R(0, 0, 22, 10, "#1a1424");
  R(2, 2, 18, 1, PK);
  R(2, 7, 18, 1, PK);
  R(2, 2, 1, 6, PK);
  R(19, 2, 1, 6, PK);
  R(5, 4, 12, 2, "#5ab0ff");
}, { lights: [[6, 2, "#ff6fae", 0.5], [11, 5, "#5ab0ff", 0.7], [16, 7, "#ff6fae", 0.5]] });
add("gargoyle", 8, 8, GY, () => {
  R(1, 4, 6, 4, GY);
  O(4, 3, 2, GY);
  P(3, 2, RD);
  P(5, 2, RD);
  Ln(0, 2, 2, 4, GY);
  Ln(8, 2, 6, 4, GY);
});
add("roofgarden", 20, 10, GN, () => {
  B(0, 6, 20, 4, "#8a5a2b", "#a87040", "#5a3816");
  for (let x = 1; x < 20; x += 4) O(x + 1, 3, 2.6, x % 8 === 1 ? GN : GNL);
  P(6, 2, PK);
  P(14, 1, YL);
});
add("billboardroof", 34, 26, ML, () => {
  for (const x of [6, 26]) R(x, 16, 2, 10, MK);
  Ln(6, 25, 26, 16, MT);
  R(0, 0, 34, 16, MK);
  R(1, 1, 32, 14, "#ffcc33");
  R(3, 3, 12, 10, RD);
  R(17, 4, 14, 3, BK);
  R(17, 9, 10, 2, BK);
}, { lights: [[8, 0, "#ffd479", 0], [25, 0, "#ffd479", 0]] });
add("bellframe", 14, 14, "#c8a040", () => {
  R(0, 0, 14, 2, WK);
  legs(0, 0, 14, 14, WK);
  for (let y = 3; y < 10; y++) R(7 - Math.floor((y - 2) / 2) - 1, y, Math.floor((y - 2) / 2) * 2 + 2, 1, "#c8a040");
  P(6, 10, "#8a6a20");
});

/* ================================================================== */
/* Underground                                                         */
/* ================================================================== */
add("minecart", 14, 10, MK, () => {
  for (let y = 0; y < 4; y++) O(3 + y * 2 + 1, 2, 1.6, ["#3a3a44", "#c8a040", "#5a5a64", "#3a3a44"][y]);
  B(0, 2, 14, 6, "#6a5a4a", "#8a7a6a", "#3a2a20");
  R(0, 4, 14, 1, MK);
  wheel(3, 8, 1.6);
  wheel(10, 8, 1.6);
});
add("pitprops", 16, 24, WK, () => {
  R(0, 0, 16, 3, WD);
  R(0, 0, 16, 1, WL);
  R(1, 3, 2, 21, WD);
  R(13, 3, 2, 21, WD);
  Ln(3, 3, 6, 6, WK);
  Ln(13, 3, 10, 6, WK);
});
add("minelantern", 5, 8, YL, () => {
  R(2, 0, 1, 2, MK);
  R(0, 2, 5, 1, MK);
  R(1, 3, 3, 4, YLL);
  R(0, 7, 5, 1, MK);
}, { hang: true, lights: [[2, 5, "#ffd479", 0.4]] });
add("orepile", 14, 7, "#c8a040", () => {
  O(4, 5, 3.5, "#4a4a54");
  O(10, 5, 3, "#5a5a64");
  for (const [x, y, col] of [[3, 3, YL], [6, 5, "#5ff0e0"], [10, 4, YL], [12, 5, "#c8a040"]] as const) P(x, y, col);
}, { lights: [[3, 3, "#ffe27a", 2]] });
add("toolsrack", 12, 14, WK, () => {
  R(0, 0, 12, 2, WK);
  Ln(2, 2, 2, 13, WK);
  R(0, 2, 5, 2, MT);
  Ln(6, 2, 6, 13, WK);
  R(5, 12, 3, 2, MT);
  Ln(10, 2, 10, 13, WK);
  R(9, 2, 3, 1, ML);
});
add("treasurechest", 12, 9, YL, () => {
  R(0, 0, 12, 4, WD);
  R(0, 0, 12, 1, WL);
  for (let x = 2; x < 11; x += 2) P(x, 3, YL);
  B(0, 4, 12, 5, WD, WL, WK);
  R(0, 4, 12, 1, "#c8a040");
  R(5, 4, 2, 3, "#c8a040");
  P(5, 1, YLL);
}, { lights: [[6, 3, "#ffe27a", 1.5]] });
add("skeleton", 14, 6, "#e8e0c8", () => {
  const bone = "#e8e0c8";
  O(2, 3, 2, bone);
  P(1, 2, BK);
  Ln(4, 3, 10, 3, bone);
  for (let x = 5; x < 10; x += 2) R(x, 2, 1, 3, bone);
  Ln(10, 3, 13, 1, bone);
  Ln(10, 3, 13, 5, bone);
});
add("fossilfish", 16, 8, "#c8c0a8", () => {
  R(0, 0, 16, 8, "#8a8070");
  const bone = "#d8d0b8";
  O(4, 4, 2.2, bone);
  Ln(6, 4, 13, 4, bone);
  for (let x = 7; x < 13; x += 2) {
    P(x, 2, bone);
    P(x, 6, bone);
  }
  Ln(13, 4, 15, 1, bone);
  Ln(13, 4, 15, 7, bone);
}, { wall: true });
add("amphora", 7, 12, "#b8543a", () => {
  R(2, 0, 3, 1, "#8a3e2c");
  R(3, 1, 1, 2, "#b8543a");
  O(3, 7, 3.2, "#b8543a");
  R(1, 5, 5, 1, "#3a2010");
  R(2, 11, 3, 1, "#8a3e2c");
  Ln(0, 3, 1, 5, "#8a3e2c");
  Ln(6, 3, 5, 5, "#8a3e2c");
});
add("sarcophagus", 22, 9, YL, () => {
  B(0, 2, 22, 7, "#c8a040", YL, "#8a6a20");
  O(4, 4, 2.6, "#c8a040");
  P(3, 4, "#3a6ab0");
  P(5, 4, "#3a6ab0");
  for (let x = 8; x < 21; x += 3) R(x, 4, 2, 3, "#3a6ab0");
});
add("subwaycar", 64, 22, ML, () => {
  R(0, 1, 64, 17, "#c8ced9");
  R(0, 0, 64, 2, MT);
  for (let x = 4; x < 60; x += 9) R(x, 4, 6, 6, "#1a2a3a");
  R(0, 11, 64, 2, BL);
  R(30, 4, 6, 13, "#5d6578");
  R(0, 18, 64, 2, MK);
  for (const x of [8, 18, 46, 56]) wheel(x, 20, 1.6);
}, { lights: [[7, 6, "#fff6c8", 0], [25, 6, "#fff6c8", 0], [52, 6, "#fff6c8", 0]] });
add("platformbench", 14, 7, MT, () => {
  R(0, 2, 14, 2, "#3a6ab0");
  R(0, 2, 14, 1, BLL);
  for (const x of [1, 12]) R(x, 4, 1, 3, MK);
});
add("tunnellamp", 8, 4, YL, () => {
  R(0, 0, 8, 1, MK);
  R(1, 1, 6, 2, YLL);
  R(1, 3, 6, 1, MK);
}, { hang: true, lights: [[4, 2, "#ffd479", 0]] });
add("cablespool", 12, 12, WD, () => {
  O(6, 6, 5.6, WK);
  O(6, 6, 4.4, "#2a2a30");
  for (let k = 0; k < 4; k++) O(6, 6, 3.5 - k * 0.7, k % 2 ? "#c8302a" : "#1e1e24");
  O(6, 6, 1.2, WD);
});
add("gaspipe", 24, 10, YL, () => {
  R(0, 3, 24, 4, "#c8a020");
  R(0, 3, 24, 1, YL);
  for (const x of [5, 18]) R(x, 2, 2, 6, "#8a6a10");
  R(10, 0, 4, 3, MT);
  R(9, 0, 6, 1, RD);
  R(14, 7, 3, 1, BK);
  P(20, 4, BK);
});
add("sewerpipe", 16, 14, MT, () => {
  O(8, 7, 7, "#4a5264");
  O(8, 7, 5, "#2a2e3a");
  O(8, 9, 3, "#5a8a5a");
  R(3, 12, 10, 2, "#5a8a5a");
});
add("bunkerbed", 18, 9, "#4a5a30", () => {
  legs(0, 2, 18, 7, MT);
  R(1, 3, 16, 3, "#4a5a30");
  R(1, 3, 4, 2, CR);
  R(1, 6, 16, 1, MT);
});
add("roots", 16, 12, "#6a4a2a", () => {
  for (const [x, len] of [[2, 10], [6, 7], [10, 12], [13, 6]] as const) {
    let px = x;
    for (let y = 0; y < len; y++) {
      if (y % 3 === 2) px += (x + y) % 2 ? 1 : -1;
      P(px, y, y < 3 ? "#7a5a3a" : "#6a4a2a");
    }
  }
}, { hang: true });
add("stalagmites", 16, 10, "#6d7184", () => {
  for (const [x, h] of [[1, 6], [5, 10], [9, 7], [13, 4]] as const)
    for (let y = 0; y < h; y++) R(x + Math.floor(y / 3) - Math.floor(h / 3) + 1, 10 - h + y, Math.max(1, Math.floor(y / 3)), 1, y % 2 ? "#6d7184" : "#5a5e70");
});
add("undergroundpool", 24, 4, "#3a8ac8", () => {
  R(0, 1, 24, 3, "#2e7aa0");
  R(0, 0, 24, 1, "#5ab0e0");
  P(6, 1, "#c8f0ff");
  P(17, 2, "#c8f0ff");
}, { lights: [[12, 1, "#5ab0e0", 0.6]] });

/* ================================================================== */
/* Railways, power, the sea floor                                      */
/* ================================================================== */
const railcar = (body: string, light: string, dark: string, fn?: () => void) => () => {
  R(0, 2, 40, 9, body);
  R(0, 2, 40, 1, light);
  R(0, 10, 40, 1, dark);
  if (fn) fn();
  R(2, 11, 36, 1, MK);
  for (const x of [5, 11, 29, 35]) wheel(x, 12, 1.6);
};
add("boxcar", 40, 14, "#8a3a24", railcar("#8a3a24", "#a85a3a", "#5a2414", () => {
  for (let x = 2; x < 38; x += 3) R(x, 3, 1, 7, "#6a2a1a");
  R(16, 3, 8, 7, "#5a2414");
  R(4, 5, 8, 2, WH);
}));
add("tankcar", 40, 14, "#3a3a44", () => {
  O(6, 6, 4.5, "#3a3a44");
  O(34, 6, 4.5, "#3a3a44");
  R(6, 2, 28, 9, "#3a3a44");
  R(6, 2, 28, 1, "#5a5a64");
  R(18, 0, 4, 2, MT);
  R(10, 5, 20, 2, YL);
  R(2, 11, 36, 1, MK);
  for (const x of [5, 11, 29, 35]) wheel(x, 12, 1.6);
});
add("flatcar", 40, 14, WD, railcar(WD, WL, WK, () => {
  R(0, 2, 40, 7, "#00000000");
  c.clearRect(0, 2, 40, 6);
  R(3, 3, 14, 5, "#3a7a4a");
  R(20, 4, 16, 4, "#5a6a3a");
  R(24, 2, 9, 2, "#5a6a3a");
  R(0, 8, 40, 3, WD);
}));
add("railsignal", 5, 22, MK, () => {
  R(2, 6, 1, 16, MK);
  R(0, 0, 5, 7, BK);
  P(2, 2, RD);
  P(2, 4, "#5bbf3a");
  R(0, 21, 5, 1, MK);
}, { lights: [[2, 2, "#ff3b3b", 0.5], [2, 4, "#5bff6a", 0.5]] });
add("coolingtower", 40, 70, WG, () => {
  for (let y = 0; y < 70; y++) {
    const u = y / 69;
    const half = Math.round(13 + Math.pow(Math.abs(u - 0.62) / 0.62, 2) * 7);
    R(20 - half, y, half * 2, 1, y < 3 ? "#a8a8b0" : u > 0.92 ? "#8a8a92" : (y >> 3) % 2 ? "#c8c8d0" : "#d8d8e0");
    R(20 - half, y, 2, 1, "#e8e8f0");
    R(20 + half - 3, y, 3, 1, "#a0a0a8");
  }
  R(9, 66, 22, 4, "#3a3a44");
  for (let x = 11; x < 30; x += 4) R(x, 66, 2, 4, "#1e1e24");
}, { smoke: [[14, 0], [20, 0], [26, 0]] });
add("hazardsign", 9, 13, YL, () => {
  R(4, 8, 1, 5, MT);
  R(0, 0, 9, 8, YL);
  O(4, 4, 2.5, BK);
  O(4, 4, 1, YL);
  P(4, 4, BK);
});
add("seaweed", 8, 14, GN, () => {
  for (const [x, h] of [[1, 12], [3, 14], [5, 10], [7, 8]] as const) {
    let px = x;
    for (let y = 0; y < h; y++) {
      if (y % 3 === 1) px += y % 6 === 1 ? 1 : -1;
      P(Math.max(0, Math.min(7, px)), 14 - y - 1, y % 2 ? "#3a8a4a" : "#4aa05a");
    }
  }
});
add("coral", 12, 8, PK, () => {
  for (const [x, col] of [[1, "#ff6fae"], [5, "#ff8a5a"], [9, "#c77dff"]] as const) {
    R(x, 2, 2, 6, col);
    R(x - 1, 1, 1, 3, col);
    R(x + 2, 0, 1, 4, col);
  }
});

/* ================================================================== */
/* Mines and airfields                                                 */
/* ================================================================== */
add("headframe", 22, 40, MT, () => {
  // A pithead: a lattice frame over the shaft, the winding wheel on top, the engine house beside it
  Ln(2, 39, 10, 4, MT);
  Ln(18, 39, 10, 4, MT);
  for (let y = 10; y < 38; y += 6) R(4 + Math.round((y - 10) * 0.08), y, 13, 1, ML);
  O(10, 4, 3.6, MK);
  O(10, 4, 2.2, "#00000000");
  P(10, 4, ML);
  Ln(10, 4, 20, 22, BK);
  B(14, 28, 8, 12, "#8a3e2c", "#a24c36", "#5a2414");
  R(16, 31, 2, 2, YLL);
}, { lights: [[17, 32, "#ffd479", 0]] });
add("carttipped", 16, 9, MK, () => {
  // A mine cart over on its side, its load spilled out in front of it
  B(4, 0, 7, 8, "#6a5a4a", "#8a7a6a", "#3a2a20");
  R(4, 3, 7, 1, MK);
  wheel(12, 2, 1.6);
  wheel(12, 6, 1.6);
  O(2, 7, 2.2, "#3a3a44");
  for (const [x, y, col] of [[1, 6, "#c8a040"], [3, 7, "#5a5a64"], [0, 8, "#2a2a30"]] as const) P(x, y, col);
});
add("cartcoal", 14, 10, MK, () => {
  // A mine cart heaped with coal
  for (let x = 1; x < 13; x++) R(x, 2 - Math.round(Math.sin((x / 12) * Math.PI) * 2), 1, 3, x % 3 ? "#1e1e24" : "#3a3a44");
  P(5, 0, "#5a5a64");
  B(0, 2, 14, 6, "#5a4a3a", "#7a6a5a", "#2a1e18");
  R(0, 4, 14, 1, MK);
  wheel(3, 8, 1.6);
  wheel(10, 8, 1.6);
});
add("railbuffer", 8, 8, WK, () => {
  // The end of the line: a timber buffer stop with a striped board
  R(0, 2, 8, 3, "#c8302a");
  for (let x = 1; x < 8; x += 3) R(x, 2, 1, 3, "#f0f0e8");
  R(1, 5, 1, 3, WK);
  R(6, 5, 1, 3, WK);
  Ln(1, 5, 6, 7, WD);
  R(0, 1, 8, 1, WK);
});
add("dynamite", 12, 8, RD, () => {
  // A crate of dynamite, the lid off: red sticks and a fuse
  B(0, 3, 12, 5, WD, WL, WK);
  R(3, 5, 6, 1, RD);
  for (let x = 2; x < 10; x += 2) R(x, 1, 2, 2, x % 4 ? "#e0402a" : RD);
  Ln(9, 1, 11, 0, "#d8c8a0");
  P(11, 0, YLL);
}, { boom: 22, lights: [[11, 0, "#ffd479", 6]] });
add("winch", 22, 18, "#8a3e2c", () => {
  // A winding house: a brick shed with the drum the shaft's cable winds onto, a lit window
  B(0, 4, 22, 14, "#8a3e2c", "#a24c36", "#5a2414");
  R(0, 2, 22, 2, MK);
  R(1, 1, 20, 1, ML);
  O(8, 11, 4, MK);
  O(8, 11, 2.4, "#5a5a64");
  P(8, 11, ML);
  Ln(8, 7, 21, 4, BK);
  R(15, 8, 4, 4, "#1e1e24");
  R(16, 9, 2, 2, YLL);
  R(14, 14, 6, 4, WK);
}, { lights: [[17, 10, "#ffd479", 0]], smoke: [[3, 2]] });
add("orebin", 18, 22, WD, () => {
  // An ore bin on timber legs, a chute down out of the bottom of it, ore heaped inside
  for (let x = 1; x < 17; x++) R(x, 1 - Math.round(Math.sin((x / 16) * Math.PI) * 1.5), 1, 2, x % 4 ? "#5a5a64" : "#c8a040");
  B(0, 2, 18, 9, WD, WL, WK);
  for (let x = 4; x < 18; x += 5) R(x, 2, 1, 9, WK);
  R(6, 11, 6, 2, WK);
  Ln(9, 13, 13, 18, MT);
  Ln(10, 13, 14, 18, ML);
  R(1, 11, 2, 11, WK);
  R(15, 11, 2, 11, WK);
  Ln(3, 12, 15, 20, WK);
});
add("minecage", 12, 16, MT, () => {
  // The cage that rides the shaft, hanging on its cable
  R(5, 0, 2, 4, BK);
  R(0, 4, 12, 1, MK);
  R(0, 15, 12, 1, MK);
  for (let x = 0; x < 12; x += 3) R(x, 5, 1, 10, ML);
  R(11, 5, 1, 10, ML);
  for (let y = 7; y < 15; y += 4) R(0, y, 12, 1, MT);
  R(4, 9, 3, 6, "#3a3a44");
  P(5, 9, "#e8c088");
}, { hang: true });
add("drillrig", 14, 14, YL, () => {
  // A rock drill on its tripod, its bit at the face, the air line coiled beside it
  Ln(3, 13, 7, 3, MK);
  Ln(11, 13, 7, 3, MK);
  B(4, 1, 7, 4, YL, YLL, "#a07a10");
  R(0, 2, 4, 2, MT);
  R(0, 3, 1, 1, ML);
  O(11, 11, 2.5, BK);
  O(11, 11, 1.4, "#00000000");
});
add("timbers", 14, 6, WD, () => {
  // A stack of pit timbers waiting to go down
  for (let y = 0; y < 3; y++) for (let x = (y % 2) * 2; x < 14; x += 4) B(x, 4 - y * 2, 4, 2, WD, WL, WK);
});
add("runwaylight", 4, 3, YL, () => {
  R(0, 1, 4, 2, MK);
  R(1, 0, 2, 1, YLL);
}, { lights: [[1, 0, "#ffd479", 1.2]] });
add("parkedjet", 32, 12, "#5d6578", () => {
  R(2, 4, 26, 4, "#6a7280");
  R(2, 4, 26, 1, "#8a92a0");
  R(0, 5, 3, 2, "#4a525e");
  Ln(10, 7, 18, 11, "#4a525e");
  Ln(11, 7, 19, 11, "#4a525e");
  R(24, 0, 3, 5, "#6a7280");
  R(6, 3, 5, 1, GLS);
  R(28, 5, 3, 2, MK);
  wheel(6, 10, 1);
  wheel(20, 10, 1);
});
add("parkedheli", 30, 14, "#4a5a30", () => {
  R(0, 1, 30, 1, MK);
  R(13, 1, 1, 3, MK);
  R(6, 4, 14, 7, "#4a5a30");
  R(6, 4, 14, 1, "#6a7a48");
  R(7, 5, 4, 3, GLS);
  R(20, 6, 9, 2, "#3e4c28");
  R(27, 3, 2, 5, "#3e4c28");
  R(6, 12, 14, 1, MK);
  R(8, 11, 1, 1, MK);
  R(18, 11, 1, 1, MK);
});
add("cargoplane", 70, 22, "#6a7280", () => {
  R(4, 8, 58, 9, "#7a8290");
  R(4, 8, 58, 1, "#9aa2b0");
  O(6, 12, 4.5, "#7a8290");
  R(2, 10, 4, 3, GLS);
  R(56, 0, 6, 9, "#7a8290");
  R(50, 2, 12, 2, "#6a7280");
  R(14, 5, 34, 3, "#5a6270");
  for (const x of [20, 38]) {
    R(x, 4, 6, 3, MK);
    R(x - 1, 5, 1, 1, BK);
  }
  R(10, 18, 4, 3, MK);
  R(44, 18, 4, 3, MK);
  wheel(12, 20, 1.5);
  wheel(46, 20, 1.5);
});
add("windsock", 12, 22, OR, () => {
  R(0, 2, 1, 20, ML);
  for (let k = 0; k < 10; k++) R(1 + k, 2 + Math.floor(k / 4), 1, 4 - Math.floor(k / 4), k % 4 < 2 ? OR : WH);
  R(0, 21, 3, 1, MK);
});

/* ================================================================== */
/* The city's streets and its metro                                    */
/* ================================================================== */
add("newsstand", 14, 14, GN, () => {
  // A green newsstand, its shutters up, papers and magazines along the front
  B(0, 2, 14, 12, GN, GNL, GNK);
  R(0, 0, 14, 2, GNK);
  R(1, 4, 12, 5, "#1e2a1e");
  for (let x = 2; x < 12; x += 2) R(x, 5, 1, 3, [WH, RD, YL, BLL, PK][(x >> 1) % 5]);
  R(1, 10, 12, 2, WD);
  for (let x = 2; x < 12; x += 3) R(x, 9, 2, 1, [WH, YLL, RL][x % 3]);
});
add("foodtruck", 28, 16, YL, () => {
  // A food truck, its serving hatch open under an awning, a menu on the side
  B(0, 3, 22, 10, YL, YLL, "#a07a10");
  B(22, 6, 6, 7, YL, YLL, "#a07a10");
  R(23, 7, 4, 3, GLS);
  R(4, 5, 12, 5, "#3a2a20");
  R(5, 6, 3, 2, "#e8c088");
  R(3, 2, 14, 2, RD);
  for (let x = 3; x < 17; x += 2) P(x, 3, WH);
  R(17, 6, 4, 6, WH);
  R(18, 7, 2, 1, RD);
  R(18, 9, 2, 1, RD);
  wheel(6, 14, 2);
  wheel(22, 14, 2);
}, { lights: [[8, 6, "#ffd479", 0]], smoke: [[10, 2]] });
add("kiosk", 12, 16, BL, () => {
  // A ticket kiosk with a window and a little roof
  R(0, 0, 12, 3, BLK);
  R(1, 3, 10, 13, BL);
  R(2, 5, 8, 5, GLS);
  R(2, 5, 8, 1, "#d8f4ff");
  R(1, 11, 10, 1, BLK);
  R(4, 12, 4, 4, BLK);
}, { lights: [[6, 7, "#ffe27a", 0]] });
add("vending", 8, 16, RD, () => {
  // A drinks machine, lit up, rows of cans behind its glass
  B(0, 0, 8, 16, RD, RL, RK);
  R(1, 2, 4, 9, "#1e1e24");
  for (let y = 3; y < 10; y += 2) for (let x = 1; x < 5; x++) P(x, y, [BLL, YL, GNL, WH][(x + y) % 4]);
  R(6, 3, 1, 4, WH);
  R(1, 12, 5, 2, BK);
}, { lights: [[2, 5, "#e8f4ff", 0]] });
add("metrosign", 8, 22, BL, () => {
  // The way down to the metro: a pole with its roundel on top
  R(3, 8, 2, 14, MK);
  O(4, 4, 3.8, RD);
  O(4, 4, 2.6, WH);
  R(1, 3, 7, 3, BL);
  P(2, 4, WH);
  P(4, 4, WH);
  P(6, 4, WH);
}, { lights: [[4, 4, "#ffffff", 0]] });
add("metromap", 16, 10, WH, () => {
  // The map of the lines, on the wall
  B(0, 0, 16, 10, WH, "#ffffff", WG);
  Ln(1, 7, 14, 2, RD);
  Ln(1, 3, 14, 6, BL);
  Ln(2, 5, 13, 5, GN);
  for (const [x, y] of [[4, 6], [8, 4], [11, 5], [6, 5]] as const) R(x, y, 2, 2, BK);
}, { wall: true });
add("arrivalboard", 20, 8, BK, () => {
  // The departure board over the platform, its amber lines
  R(9, 0, 1, 2, MK);
  B(0, 2, 20, 6, "#141418", MK, MK);
  for (let x = 2; x < 18; x += 2) P(x, 4, YL);
  for (let x = 2; x < 12; x += 2) P(x, 6, YL);
  P(15, 6, "#7aff6a");
}, { hang: true, lights: [[4, 4, "#ffb020", 0], [14, 4, "#ffb020", 0]] });

/* ================================================================== */
/* Railways and works                                                  */
/* ================================================================== */
add("hopper", 40, 14, "#4a4a54", railcar("#4a4a54", "#6a6a74", "#2a2a30", () => {
  // An open hopper wagon heaped with coal, its chutes under it
  for (let x = 2; x < 38; x++) R(x, 1 - Math.round(Math.sin((x / 36) * Math.PI) * 1.5), 1, 2, x % 3 ? "#1e1e24" : "#3a3a44");
  for (const x of [8, 20, 32]) R(x - 2, 6, 4, 4, "#3a3a44");
}));
add("signalgantry", 44, 34, MT, () => {
  // A steel gantry over the tracks, a signal for each line on it
  R(1, 4, 2, 30, MT);
  R(41, 4, 2, 30, MT);
  R(0, 2, 44, 3, ML);
  R(0, 5, 44, 1, MK);
  for (let x = 4; x < 40; x += 4) Ln(x, 5, x + 4, 2, MK);
  for (const x of [10, 22, 34]) {
    B(x, 6, 5, 9, BK, MK, BK);
    O(x + 2, 8, 1.2, "#ff3b3b");
    O(x + 2, 12, 1.2, "#2a5a2a");
  }
}, { lights: [[12, 8, "#ff3b3b", 0.8], [24, 12, "#5aff6a", 0], [36, 8, "#ff3b3b", 0.6]] });
add("railtunnel", 30, 30, "#7a7a84", () => {
  // A tunnel mouth: a stone portal, its arch, the dark of the tunnel in it
  B(0, 0, 30, 30, "#7a7a84", "#9a9aa4", "#4a4a54");
  for (let y = 3; y < 30; y += 4) R(0, y, 30, 1, "#5a5a64");
  O(15, 16, 11, "#0d0b14");
  R(4, 16, 22, 14, "#0d0b14");
  O(15, 16, 9, "#1a1822");
  R(6, 16, 18, 14, "#1a1822");
  R(10, 1, 10, 3, "#5a5a64");
});
add("watercrane", 14, 26, MT, () => {
  // A water crane for the engines: a column, its arm and the leather hose hanging off it
  R(5, 4, 4, 22, MT);
  R(4, 23, 6, 3, MK);
  R(0, 2, 14, 3, ML);
  R(0, 5, 2, 6, "#3a2a20");
  O(7, 3, 2, MK);
});
add("signalbox", 24, 26, "#8a3e2c", () => {
  // A signal box: a brick base, the cabin on it glazed all round, its lever frame inside
  B(2, 14, 20, 12, "#8a3e2c", "#a24c36", "#5a2414");
  B(0, 6, 24, 9, WH, "#ffffff", WG);
  R(2, 8, 20, 4, GLS);
  for (let x = 3; x < 22; x += 3) R(x, 10, 1, 2, RK);
  R(0, 3, 24, 3, "#3a3a44");
  Ln(0, 3, 12, 0, "#3a3a44");
  Ln(23, 3, 12, 0, "#3a3a44");
  R(10, 18, 4, 8, WK);
}, { lights: [[6, 9, "#ffd479", 0], [16, 9, "#ffd479", 0]] });
add("smokestack", 12, 64, "#8a3e2c", () => {
  // A tall works chimney, brick, its iron bands, smoke pouring out of the top
  for (let y = 0; y < 64; y++) {
    const w = 6 + Math.round((y / 63) * 5);
    R(6 - (w >> 1), y, w, 1, y % 4 === 0 ? "#7a3424" : "#8a3e2c");
  }
  for (const y of [6, 22, 40]) R(1, y, 10, 1, MK);
  R(3, 0, 6, 2, MK);
}, { smoke: [[6, 0]] });
add("gasholder", 46, 34, MT, () => {
  // A gas holder: its tank inside a ring of lattice columns tied together at the top
  B(4, 10, 38, 24, "#5a6a5a", "#7a8a7a", "#3a4a3a");
  for (let y = 14; y < 34; y += 5) R(4, y, 38, 1, "#4a5a4a");
  for (const x of [1, 15, 29, 43]) {
    R(x, 2, 2, 32, MT);
    for (let y = 4; y < 32; y += 6) Ln(x, y, x + 2, y + 3, MK);
  }
  R(0, 2, 46, 2, ML);
  R(0, 18, 46, 1, ML);
});
add("factoryshed", 50, 30, "#8a3e2c", () => {
  // A works shed: brick walls, a sawtooth roof of glazed north lights, big doors
  B(0, 10, 50, 20, "#8a3e2c", "#a24c36", "#5a2414");
  for (let x = 0; x < 50; x += 10) {
    Ln(x, 10, x + 10, 2, MK);
    R(x + 9, 2, 1, 8, MK);
    for (let y = 3; y < 10; y++) R(x + 10 - Math.round((y - 2) * 1.25), y, 1, 1, GLS);
  }
  for (let x = 4; x < 46; x += 6) R(x, 14, 3, 4, "#1e1e24");
  R(20, 20, 10, 10, MK);
  R(25, 20, 1, 10, ML);
}, { lights: [[5, 15, "#ffd479", 0], [41, 15, "#ffd479", 0]] });
add("storagetank", 30, 24, WG, () => {
  // A big storage tank: a white cylinder, a ladder spiralling up it, a walkway round the top
  B(0, 4, 30, 20, WG, WH, GY);
  for (let y = 8; y < 24; y += 5) R(0, y, 30, 1, "#b0aca4");
  O(15, 4, 3, WG);
  R(0, 3, 30, 1, MK);
  for (let k = 0; k < 18; k++) P(4 + k, 22 - k, MK);
  R(8, 12, 12, 4, RD);
  R(9, 13, 10, 2, WH);
});
add("pylon", 22, 60, MT, () => {
  // A pylon carrying the power lines across the works
  Ln(2, 59, 9, 2, MT);
  Ln(19, 59, 12, 2, MT);
  for (let y = 8; y < 58; y += 8) {
    const a = 2 + Math.round(((59 - y) / 57) * 7);
    Ln(a, y, 21 - a, y + 8, MK);
    Ln(21 - a, y, a, y + 8, MK);
  }
  R(0, 8, 22, 2, MT);
  R(3, 18, 16, 2, MT);
  for (const x of [0, 21, 3, 18]) R(x, x < 3 || x > 18 ? 10 : 20, 1, 3, "#c8c4bc");
});
add("pipebridge", 48, 24, MT, () => {
  // A pipe bridge: a truss on two towers, three pipes along it
  for (const x of [2, 44]) R(x, 6, 3, 18, MT);
  R(0, 4, 48, 2, ML);
  for (let x = 2; x < 46; x += 6) Ln(x, 4, x + 6, 10, MK);
  R(0, 10, 48, 1, MK);
  R(0, 0, 48, 2, "#c8302a");
  R(0, 2, 48, 1, "#5a7a6a");
  R(0, 11, 48, 2, "#8a8a92");
});
add("coalheap", 20, 8, "#2a2a30", () => {
  // A heap of coal
  for (let x = 0; x < 20; x++) {
    const h = Math.round(Math.sin(((x + 0.5) / 20) * Math.PI) * 7) + 1;
    R(x, 8 - h, 1, h, x % 3 ? "#1e1e24" : "#3a3a44");
  }
  for (const [x, y] of [[6, 3], [11, 2], [14, 4]] as const) P(x, y, "#5a5a64");
});

/* ================================================================== */
/* Parks, plants and the expressway                                    */
/* ================================================================== */
add("gazebo", 24, 20, WH, () => {
  // A bandstand: a domed roof on its slender columns, a rail round its floor
  for (let x = 2; x < 22; x++) R(x, 4 - Math.round(Math.sin(((x - 2) / 19) * Math.PI) * 3), 1, 3, x % 4 ? GN : GNK);
  R(1, 6, 22, 2, WH);
  P(12, 0, YL);
  for (const x of [2, 8, 15, 21]) R(x, 8, 1, 9, WH);
  R(0, 17, 24, 3, WK);
  R(0, 17, 24, 1, WL);
  for (let x = 1; x < 23; x += 2) R(x, 14, 1, 3, WG);
  R(0, 13, 24, 1, WH);
});
add("playground", 28, 16, RD, () => {
  // A playground: a swing on its frame, a slide down off a little tower
  Ln(1, 15, 6, 1, MT);
  Ln(11, 15, 6, 1, MT);
  R(4, 0, 6, 1, MT);
  Ln(5, 1, 5, 9, ML);
  Ln(8, 1, 8, 9, ML);
  R(4, 9, 5, 1, RD);
  R(16, 5, 5, 11, BL);
  R(16, 4, 5, 1, YL);
  for (let y = 7; y < 15; y += 2) R(15, y, 1, 1, MT);
  Ln(21, 6, 27, 14, RD);
  Ln(21, 7, 27, 15, RL);
});
add("flowerbed", 14, 5, PK, () => {
  // A raised bed of flowers
  B(0, 2, 14, 3, "#7a5a3a", "#9a7a5a", "#4a3a24");
  for (let x = 1; x < 13; x++) P(x, 1, x % 2 ? GNK : GN);
  for (let x = 1; x < 13; x += 2) P(x, 0, [PK, YL, RD, WH, PU][(x >> 1) % 5]);
});
add("icecreamcart", 16, 14, PK, () => {
  // An ice-cream cart: its striped canopy, the freezer box, a cone on top
  for (let x = 0; x < 16; x += 2) R(x, 0, 2, 3, (x >> 1) % 2 ? PK : WH);
  R(7, 3, 1, 4, MT);
  B(1, 7, 14, 6, WH, "#ffffff", WG);
  R(3, 9, 10, 2, PK);
  wheel(4, 13, 1.6);
  wheel(12, 13, 1.6);
});
add("picnictable", 14, 7, WD, () => {
  R(0, 2, 14, 2, WD);
  R(0, 2, 14, 1, WL);
  Ln(2, 4, 4, 7, WK);
  Ln(12, 4, 10, 7, WK);
  R(0, 5, 3, 1, WD);
  R(11, 5, 3, 1, WD);
});
add("clarifier", 30, 12, WG, () => {
  // A clarifier: a round concrete tank, its bridge turning slowly over the water
  B(0, 4, 30, 8, WG, WH, GY);
  R(2, 4, 26, 2, "#4a8ab0");
  R(2, 4, 26, 1, "#7ab0d0");
  R(14, 0, 2, 5, MT);
  R(2, 2, 26, 1, MT);
  for (let x = 3; x < 28; x += 3) P(x, 1, ML);
});
add("reactordome", 40, 40, WG, () => {
  // A reactor's containment dome, its cylinder walls and ring beams, a hatch
  O(20, 18, 17, WG);
  R(3, 18, 34, 22, WG);
  for (let y = 18; y < 40; y += 6) R(3, y, 34, 1, GY);
  O(20, 18, 15, WH);
  R(5, 18, 30, 1, WH);
  R(16, 32, 8, 8, MT);
  R(17, 33, 6, 6, MK);
  R(2, 37, 36, 3, GY);
}, { lights: [[20, 3, "#ff3b3b", 1]] });
add("hwysign", 40, 14, GN, () => {
  // An expressway sign over the road: white letters on green, its posts
  R(4, 9, 1, 5, MT);
  R(35, 9, 1, 5, MT);
  B(0, 0, 40, 10, "#1e6a3a", "#2e8a4a", "#124a2a");
  R(1, 1, 38, 1, WH);
  for (let x = 4; x < 36; x += 2) R(x, 4, 1, (x * 7) % 3 ? 3 : 2, WH);
  R(30, 3, 6, 4, WH);
  R(32, 4, 2, 2, "#1e6a3a");
});
add("buswreck", 40, 13, "#6a4a20", () => {
  // A burnt-out bus or lorry, its windows gone, its paint blistered black
  B(0, 1, 40, 9, "#3a3430", "#5a4a40", "#1e1a18");
  for (let x = 3; x < 37; x += 6) R(x, 3, 4, 4, "#0d0b14");
  for (let x = 2; x < 38; x += 5) P(x, 1, "#8a4a2a");
  R(0, 10, 40, 1, MK);
  for (const x of [6, 32]) {
    O(x, 11, 1.8, "#1e1e24");
  }
}, { flames: [[10, 1], [28, 2]], smoke: [[20, 0]] });

export const CATALOG = E;
/** Paint a catalogue entry into a context (art.ts calls this once per asset, lazily). */
export function paintCatalog(id: string, ctx: Ctx) {
  c = ctx;
  E[id].paint();
}
