/**
 * The building types: materials, size, roof, windows, sign and awning, and the furniture on each
 * floor. level.ts builds them cut away, side-on, and maps the material names here to block kinds.
 */
import type { Backdrop } from "./themes";

type Roof =
  | "flat"
  | "parapet"
  | "pitched"
  | "dome"
  | "spire"
  | "sawtooth"
  | "stepped"
  | "garden"
  | "helipad"
  | "tank"
  | "neon"
  | "billboard"
  | "antenna"
  | "chimney"
  | "solar"
  | "bell"
  | "pagoda"
  | "onion"
  | "minaret"
  | "mansard"
  | "stepgable"
  | "columns"
  | "pyramid"
  | "castle"
  | "chalet"
  | "igloo"
  | "stupa"
  | "clock"
  | "artdeco"
  | "brutalist";
type Windows = "grid" | "strip" | "glass" | "arched" | "porthole" | "none" | "shop" | "slits" | "pairs";
export type Material =
  | "brick"
  | "darkbrick"
  | "concrete"
  | "glass"
  | "marble"
  | "plaster"
  | "siding"
  | "stucco"
  | "tiles"
  | "wood"
  | "steel"
  | "tin"
  | "sandstone"
  | "stone"
  | "metal"
  | "parquet"
  | "carpet"
  | "moss"
  | "timber"
  | "adobe"
  | "terracotta"
  | "jade"
  | "slate"
  | "shoji"
  | "azulejo"
  | "limestone"
  | "snow"
  | "gold";

export interface BuildingSpec {
  key: string;
  /** The text of the sign over it (no sign if unset), and its colours */
  sign?: string;
  ink?: string;
  board?: string;
  neon?: boolean;
  wall: Material;
  /** The back wall inside, and the floors between storeys */
  back: Material;
  floor?: Material;
  /** Storeys as [min, max], rows per storey, and width in blocks as [min, max] */
  floors: [number, number];
  fh: number;
  width: [number, number];
  roof: Roof;
  /** The roof's material, for styles that have their own */
  roofMat?: Material;
  windows: Windows;
  /** A striped awning over the door, in this colour */
  awning?: string;
  /** Furniture on the ground floor (every floor if `upper` is unset), then on the walls, upstairs, on the roof and out front */
  items: string[];
  walls?: string[];
  upper?: string[];
  top?: string[];
  outside?: string[];
  where: Backdrop[];
  /** Guards per storey */
  guards?: number;
}

/* The backdrops each group of buildings stands in */
const CITY: Backdrop[] = ["city", "megacity"];
const TOWN: Backdrop[] = ["city", "factory", "fort", "megacity", "farmland", "savanna", "autumn"];
const IND: Backdrop[] = ["factory", "city", "silo", "megacity", "badlands"];
const MIL: Backdrop[] = ["fort", "silo", "airbase", "tundra"];
const RURAL: Backdrop[] = [
  "jungle",
  "monsoon",
  "swamp",
  "snow",
  "desert",
  "canyon",
  "savanna",
  "autumn",
  "farmland",
  "tundra",
  "mangrove",
  "badlands",
  "archipelago",
];
const HOT: Backdrop[] = ["desert", "canyon"];
const WET: Backdrop[] = ["jungle", "deepjungle", "monsoon", "swamp", "temple"];
const COLD: Backdrop[] = ["snow"];
const ALL_OUT: Backdrop[] = [
  "city",
  "factory",
  "fort",
  "silo",
  "jungle",
  "monsoon",
  "desert",
  "canyon",
  "snow",
  "swamp",
];

/* Furniture sets */
const OFFICE = [
  "desk",
  "desk",
  "officechair",
  "filingcabinet",
  "watercooler",
  "printer",
  "pottedplant",
  "photocopier",
  "coffeemachine",
  "bookshelf",
];
const OFFICE_W = ["whiteboard", "wallclock", "painting"];
const DINER = ["diningtable", "diningtable", "booth", "register", "jukebox", "fridge", "stove"];
const BAR = ["barcounter", "barstool", "barstool", "beertaps", "pooltable", "jukebox", "winerack"];
const SHOP = ["checkout", "shoppingcart", "cans", "cereal", "bottles", "producecrates"];
const HOME = [
  "bed",
  "sofa",
  "armchair",
  "tv",
  "wardrobe",
  "lamp",
  "kitchencounter",
  "diningtable",
  "bookshelf",
  "pottedplant",
  "rug",
];
const HOME_W = ["painting", "wallclock", "mirror"];
const BATH = ["toilet", "bathtub", "shower", "sinkcounter"];
const HOSP = ["hospitalbed", "hospitalbed", "ivstand", "heartmonitor", "wheelchair", "medicineshelf"];
const FACT = ["conveyor", "press", "lathe", "boiler", "controlpanel", "workbench", "toolchest", "pallets", "forklift"];
const STORE = ["pallets", "crates", "drums", "forklift", "containerred", "woodbarrels", "bales"];
const LAB = ["labbench", "microscope", "serverrack", "controlpanel", "filingcabinet", "desk"];
const ARMY = ["weaponrack", "ammocrate", "maptable", "radarconsole", "bunkbed", "radio", "lockers", "crates"];
const FUN = ["arcade", "arcade2", "clawmachine", "slotmachine", "vendingsoda"];
const STAGE = ["speakerstack", "micstand", "drumkit", "piano", "guitarstand"];
const ROOFTOP = ["acunit", "acunit", "vent", "dish", "solarpanels"];

type Opts = Partial<BuildingSpec>;
const SPECS: BuildingSpec[] = [];
const b = (
  key: string,
  sign: string | undefined,
  wall: Material,
  back: Material,
  floors: [number, number],
  width: [number, number],
  roof: Roof,
  windows: Windows,
  items: string[],
  o: Opts = {}
) => SPECS.push({ key, sign, wall, back, floors, fh: 5, width, roof, windows, items, where: CITY, top: ROOFTOP, ...o });

/* Offices and business */
b("officetower", "OFFICES", "glass", "plaster", [4, 7], [10, 14], "helipad", "glass", OFFICE, {
  walls: OFFICE_W,
  floor: "carpet",
  board: "#1e2a3a",
  ink: "#9ad8ff",
});
b("skyscraper", "SKY TOWER", "glass", "tiles", [6, 9], [11, 14], "antenna", "glass", OFFICE, {
  walls: OFFICE_W,
  floor: "carpet",
  board: "#14161e",
  ink: "#ffd479",
  neon: true,
});
b(
  "bank",
  "BANK",
  "marble",
  "marble",
  [2, 3],
  [10, 13],
  "dome",
  "arched",
  ["safe", "reception", "atm", "desk", "officechair", "pottedplant"],
  { walls: ["wallclock", "painting"], floor: "marble", board: "#2a3a5a", ink: "#ffd23f", guards: 2 }
);
b("insurance", "INSURANCE", "concrete", "plaster", [3, 5], [9, 12], "parapet", "grid", OFFICE, {
  walls: OFFICE_W,
  floor: "carpet",
  board: "#3a6ab0",
});
b(
  "lawfirm",
  "LAW",
  "darkbrick",
  "parquet",
  [2, 4],
  [8, 11],
  "parapet",
  "pairs",
  ["desk", "bookshelf", "bookshelf", "officechair", "filingcabinet", "globe"],
  { walls: ["painting", "wallclock"], floor: "parquet", board: "#3a2410", ink: "#ffd23f" }
);
b(
  "startup",
  "STARTUP",
  "glass",
  "tiles",
  [2, 4],
  [9, 12],
  "garden",
  "glass",
  ["desk", "desk", "officechair", "pooltable", "coffeemachine", "pottedplant", "arcade", "vendingsnack"],
  { walls: ["whiteboard"], floor: "parquet", board: "#ff6fae", ink: "#ffffff", neon: true }
);
b(
  "callcentre",
  "CALLS",
  "concrete",
  "carpet",
  [3, 4],
  [10, 13],
  "flat",
  "strip",
  ["desk", "desk", "desk", "officechair", "officechair", "watercooler"],
  { walls: ["wallclock", "whiteboard"], floor: "carpet", board: "#5ab0b0" }
);
b(
  "newspaper",
  "THE DAILY",
  "brick",
  "plaster",
  [3, 4],
  [10, 13],
  "billboard",
  "grid",
  ["printpress", "paperrolls", "desk", "officechair", "filingcabinet"],
  { upper: OFFICE, walls: ["wallclock", "painting"], board: "#f4f0e8", ink: "#1e1e24" }
);
b(
  "tvstation",
  "TV 7",
  "concrete",
  "tiles",
  [3, 5],
  [10, 13],
  "antenna",
  "strip",
  ["tvwall", "desk", "controlpanel", "projector", "speakerstack"],
  { upper: OFFICE, walls: ["tvwall"], board: "#c8302a", ink: "#ffffff" }
);
b(
  "radiostation",
  "RADIO",
  "brick",
  "carpet",
  [2, 3],
  [8, 10],
  "antenna",
  "pairs",
  ["controlpanel", "micstand", "speakerstack", "desk", "recordbins"],
  { walls: ["wallclock"], board: "#8a4ac8", ink: "#ffffff", neon: true }
);
b(
  "datacentre",
  "DATA",
  "steel",
  "tiles",
  [2, 3],
  [10, 14],
  "solar",
  "slits",
  ["serverrack", "serverrack", "serverrack", "controlpanel", "cablespool"],
  { board: "#14161e", ink: "#8aff6a", where: IND, guards: 2 }
);
b(
  "stockexchange",
  "EXCHANGE",
  "marble",
  "marble",
  [3, 4],
  [11, 14],
  "dome",
  "arched",
  ["desk", "desk", "tvwall", "officechair", "departureboard"],
  { walls: ["tvwall", "wallclock"], floor: "marble", board: "#1e2a3a", ink: "#8aff6a" }
);

/* Food and drink */
b("diner", "DINER", "tin", "tiles", [1, 1], [9, 12], "neon", "shop", DINER, {
  walls: ["menuboard", "wallclock"],
  board: "#c8302a",
  ink: "#ffffff",
  neon: true,
  awning: "#c8302a",
  where: [...CITY, ...HOT, "snow"],
});
b(
  "pizzeria",
  "PIZZA",
  "brick",
  "tiles",
  [1, 2],
  [8, 11],
  "chimney",
  "shop",
  ["pizzaoven", "diningtable", "diningtable", "register", "fridge"],
  { upper: HOME, walls: ["menuboard", "painting"], board: "#3a8a3a", ink: "#ffffff", awning: "#3a8a3a" }
);
b(
  "burgers",
  "BURGERS",
  "stucco",
  "tiles",
  [1, 1],
  [9, 12],
  "neon",
  "shop",
  ["grill", "deepfryer", "register", "booth", "diningtable", "vendingsoda"],
  { walls: ["menuboard"], board: "#ffcc33", ink: "#c8302a", neon: true, awning: "#ffcc33" }
);
b(
  "sushi",
  "SUSHI",
  "wood",
  "plaster",
  [1, 2],
  [8, 11],
  "pitched",
  "shop",
  ["sushicounter", "barstool", "barstool", "fridge", "sinkcounter"],
  {
    upper: HOME,
    walls: ["painting", "menuboard"],
    board: "#1e1e24",
    ink: "#ff5a3a",
    awning: "#c8302a",
    where: [...CITY, "monsoon"],
  }
);
b(
  "noodles",
  "NOODLES",
  "plaster",
  "tiles",
  [1, 2],
  [8, 10],
  "flat",
  "shop",
  ["stove", "diningtable", "barstool", "barcounter", "register"],
  {
    upper: HOME,
    walls: ["menuboard"],
    board: "#c8302a",
    ink: "#ffd23f",
    neon: true,
    awning: "#ffcc33",
    where: [...CITY, "monsoon", "jungle"],
  }
);
b(
  "tacos",
  "TACOS",
  "stucco",
  "tiles",
  [1, 1],
  [8, 10],
  "flat",
  "shop",
  ["grill", "diningtable", "register", "fridge"],
  {
    walls: ["menuboard"],
    board: "#ff8a1a",
    ink: "#ffffff",
    awning: "#5bbf3a",
    outside: ["umbrellatable"],
    where: [...CITY, ...HOT],
  }
);
b(
  "bakery",
  "BAKERY",
  "brick",
  "plaster",
  [1, 2],
  [8, 10],
  "chimney",
  "shop",
  ["breadrack", "breadrack", "cakecase", "register", "pizzaoven"],
  { upper: HOME, walls: ["menuboard"], board: "#f0d8b0", ink: "#8a5a2b", awning: "#ff6fae" }
);
b(
  "cafe",
  "CAFE",
  "siding",
  "parquet",
  [1, 2],
  [8, 11],
  "pitched",
  "shop",
  ["espresso", "cakecase", "diningtable", "diningtable", "register", "pottedplant"],
  {
    upper: HOME,
    walls: ["menuboard", "painting"],
    board: "#5a3816",
    ink: "#f4f0e8",
    awning: "#5a3816",
    outside: ["umbrellatable"],
  }
);
b(
  "icecream",
  "ICE CREAM",
  "stucco",
  "tiles",
  [1, 1],
  [8, 10],
  "flat",
  "shop",
  ["icecreamtub", "icecreamtub", "register", "diningtable"],
  {
    walls: ["menuboard"],
    board: "#ff9ac8",
    ink: "#ffffff",
    awning: "#ff6fae",
    outside: ["icecreamvan"],
    where: [...CITY, ...HOT],
  }
);
b(
  "steakhouse",
  "STEAKS",
  "darkbrick",
  "parquet",
  [1, 2],
  [10, 12],
  "chimney",
  "pairs",
  ["grill", "diningtable", "diningtable", "winerack", "barcounter"],
  { walls: ["painting"], board: "#3a1414", ink: "#ffcc33", where: [...CITY, ...HOT] }
);
b("pub", "PUB", "darkbrick", "parquet", [2, 2], [9, 12], "chimney", "pairs", BAR, {
  upper: HOME,
  walls: ["dartboard", "painting"],
  board: "#2a3a2a",
  ink: "#ffd23f",
  awning: "#2a5a3a",
});
b(
  "nightclub",
  "CLUB",
  "concrete",
  "carpet",
  [1, 2],
  [10, 13],
  "neon",
  "none",
  ["djbooth", "speakerstack", "speakerstack", "barcounter", "barstool", "discoball"],
  { walls: [], board: "#1a1424", ink: "#ff6fae", neon: true, guards: 2 }
);
b(
  "chicken",
  "CHICKEN",
  "plaster",
  "tiles",
  [1, 1],
  [9, 11],
  "neon",
  "shop",
  ["deepfryer", "deepfryer", "register", "booth", "vendingsoda"],
  { walls: ["menuboard"], board: "#c8302a", ink: "#ffffff", awning: "#ffffff", neon: true }
);
b(
  "fishchips",
  "FISH",
  "siding",
  "tiles",
  [1, 2],
  [8, 10],
  "pitched",
  "shop",
  ["deepfryer", "register", "diningtable", "fridge"],
  { walls: ["menuboard"], board: "#3a6ab0", ink: "#ffffff", awning: "#3a6ab0", where: [...CITY, "monsoon"] }
);
b(
  "foodmarket",
  "MARKET",
  "steel",
  "concrete",
  [1, 1],
  [14, 18],
  "sawtooth",
  "none",
  ["marketstall", "marketstall", "producecrates", "hotdogcart", "flowerbuckets"],
  { board: "#3a8a3a", ink: "#ffffff", where: [...CITY, "monsoon", "jungle"] }
);
b(
  "saloon",
  "SALOON",
  "wood",
  "wood",
  [2, 2],
  [10, 12],
  "parapet",
  "pairs",
  ["barcounter", "barstool", "cardtable", "piano", "woodbarrels"],
  { upper: ["bed", "wardrobe", "lamp"], walls: ["painting"], board: "#5a3816", ink: "#ffcc33", where: HOT, guards: 2 }
);
b(
  "teahouse",
  "TEA",
  "wood",
  "plaster",
  [1, 2],
  [8, 10],
  "pitched",
  "pairs",
  ["diningtable", "kitchencounter", "pottedplant", "rug"],
  { walls: ["painting"], board: "#c8302a", ink: "#ffcc33", where: [...WET, "monsoon"] }
);

/* Shops */
b("supermarket", "GROCERY", "concrete", "tiles", [1, 1], [14, 18], "flat", "shop", SHOP, {
  board: "#5bbf3a",
  ink: "#ffffff",
  outside: ["shoppingcart"],
  where: TOWN,
});
b(
  "pharmacy",
  "PHARMACY",
  "plaster",
  "tiles",
  [1, 2],
  [8, 10],
  "flat",
  "shop",
  ["medicineshelf", "medicineshelf", "checkout", "pottedplant"],
  { upper: HOME, board: "#5bbf3a", ink: "#ffffff", neon: true, awning: "#5bbf3a" }
);
b(
  "hardware",
  "HARDWARE",
  "brick",
  "concrete",
  [1, 1],
  [11, 14],
  "flat",
  "shop",
  ["toolchest", "workbench", "tirerack", "cablespool", "checkout", "pallets"],
  { walls: ["toolsrack"], board: "#ff8a1a", ink: "#1e1e24", awning: "#ff8a1a", where: TOWN }
);
b(
  "bookshop",
  "BOOKS",
  "brick",
  "parquet",
  [1, 2],
  [8, 10],
  "pitched",
  "shop",
  ["bookshelf", "bookshelf", "bookshelf", "armchair", "register"],
  { upper: HOME, board: "#3a2410", ink: "#f4f0e8", awning: "#8a3a4a" }
);
b(
  "recordstore",
  "RECORDS",
  "brick",
  "carpet",
  [1, 2],
  [8, 10],
  "flat",
  "shop",
  ["recordbins", "recordbins", "speakerstack", "register", "guitarstand"],
  { upper: HOME, walls: ["painting"], board: "#1e1e24", ink: "#ff6fae", neon: true, awning: "#8a4ac8" }
);
b(
  "toyshop",
  "TOYS",
  "stucco",
  "carpet",
  [1, 2],
  [8, 11],
  "flat",
  "shop",
  ["toyshelf", "toyshelf", "register", "clawmachine"],
  { upper: HOME, board: "#ffcc33", ink: "#3a6ab0", awning: "#ff6fae" }
);
b(
  "boutique",
  "FASHION",
  "marble",
  "parquet",
  [1, 2],
  [9, 11],
  "flat",
  "shop",
  ["clothesrack", "clothesrack", "mannequin", "mannequin", "mirror", "register"],
  { walls: ["mirror"], board: "#1e1e24", ink: "#ffffff", awning: "#1e1e24" }
);
b(
  "shoeshop",
  "SHOES",
  "plaster",
  "carpet",
  [1, 1],
  [8, 10],
  "flat",
  "shop",
  ["shoeshelf", "shoeshelf", "armchair", "register"],
  { board: "#8a4ac8", ink: "#ffffff", awning: "#8a4ac8" }
);
b(
  "electronics",
  "ELECTRO",
  "concrete",
  "tiles",
  [1, 2],
  [10, 12],
  "billboard",
  "shop",
  ["tvwall", "tv", "tv", "checkout", "speakerstack"],
  { walls: ["tvwall"], board: "#3a6ab0", ink: "#ffffff", neon: true }
);
b(
  "pawnshop",
  "PAWN",
  "darkbrick",
  "parquet",
  [1, 2],
  [8, 10],
  "flat",
  "slits",
  ["jewelcase", "guitarstand", "safe", "tv", "register"],
  { upper: HOME, walls: ["painting"], board: "#ffcc33", ink: "#1e1e24", where: [...CITY, ...HOT] }
);
b(
  "florist",
  "FLOWERS",
  "siding",
  "tiles",
  [1, 1],
  [8, 10],
  "garden",
  "shop",
  ["flowerbuckets", "flowerbuckets", "pottedplant", "pottedplant", "register"],
  { board: "#ff6fae", ink: "#ffffff", awning: "#5bbf3a", outside: ["flowerbuckets"] }
);
b(
  "petshop",
  "PETS",
  "stucco",
  "tiles",
  [1, 1],
  [8, 11],
  "flat",
  "shop",
  ["aquarium", "petcage", "petcage", "register", "aquarium"],
  { board: "#5ab0b0", ink: "#ffffff", awning: "#5ab0b0" }
);
b(
  "barber",
  "BARBER",
  "brick",
  "tiles",
  [1, 2],
  [7, 9],
  "flat",
  "shop",
  ["barberchair", "barberchair", "mirror", "sinkcounter"],
  { upper: HOME, walls: ["mirror"], board: "#c8302a", ink: "#ffffff", awning: "#3a6ab0" }
);
b(
  "salon",
  "SALON",
  "plaster",
  "tiles",
  [1, 2],
  [8, 10],
  "flat",
  "shop",
  ["hairdryer", "hairdryer", "barberchair", "sinkcounter", "mirror"],
  { upper: HOME, walls: ["mirror"], board: "#ff9ac8", ink: "#1e1e24", awning: "#ff6fae" }
);
b(
  "tattoo",
  "TATTOO",
  "darkbrick",
  "carpet",
  [1, 2],
  [7, 9],
  "neon",
  "slits",
  ["tattoochair", "armchair", "speakerstack"],
  { upper: HOME, walls: ["painting", "mirror"], board: "#1e1e24", ink: "#5ff0e0", neon: true }
);
b(
  "laundromat",
  "LAUNDRY",
  "tiles",
  "tiles",
  [1, 2],
  [10, 13],
  "flat",
  "shop",
  ["washer", "washer", "washer", "laundrybasket", "bench"],
  { upper: HOME, top: ["vent", "vent", "acunit"], board: "#5ab0ff", ink: "#ffffff", neon: true }
);
b(
  "convenience",
  "24/7",
  "concrete",
  "tiles",
  [1, 1],
  [9, 12],
  "flat",
  "shop",
  ["cans", "bottles", "vendingsnack", "checkout", "chestfreezer", "atm"],
  { board: "#ff8a1a", ink: "#ffffff", neon: true, where: [...TOWN, ...HOT, "snow"] }
);
b(
  "gasstation",
  "GAS",
  "steel",
  "tiles",
  [1, 1],
  [9, 11],
  "flat",
  "shop",
  ["cans", "bottles", "checkout", "vendingsoda"],
  {
    board: "#c8302a",
    ink: "#ffcc33",
    neon: true,
    outside: ["fuelpump", "fuelpump", "lpgtank"],
    where: [...TOWN, ...HOT, "snow", "jungle"],
  }
);
b(
  "jeweller",
  "JEWELS",
  "marble",
  "carpet",
  [1, 2],
  [8, 10],
  "parapet",
  "arched",
  ["jewelcase", "jewelcase", "safe", "mirror"],
  { board: "#1e1e24", ink: "#ffd23f", guards: 2 }
);
b(
  "gunshop",
  "GUNS",
  "darkbrick",
  "concrete",
  [1, 1],
  [8, 10],
  "flat",
  "slits",
  ["weaponrack", "weaponrack", "ammocrate", "register"],
  { board: "#3a3f4f", ink: "#ff8a1a", where: [...TOWN, ...HOT], guards: 2 }
);
b(
  "musicshop",
  "MUSIC",
  "siding",
  "parquet",
  [1, 2],
  [8, 10],
  "flat",
  "shop",
  ["piano", "guitarstand", "guitarstand", "drumkit", "register"],
  { upper: HOME, board: "#8a4ac8", ink: "#ffd23f", awning: "#8a4ac8" }
);

/* Cars */
b(
  "garage",
  "GARAGE",
  "tin",
  "concrete",
  [1, 1],
  [12, 15],
  "flat",
  "none",
  ["carlift", "toolchest", "workbench", "engine", "tirerack"],
  { walls: ["toolsrack"], board: "#3a3f4f", ink: "#ffcc33", where: [...TOWN, ...HOT, "snow", "jungle"] }
);
b(
  "cardealer",
  "CARS",
  "glass",
  "tiles",
  [1, 1],
  [14, 18],
  "flat",
  "glass",
  ["sedan", "sedanblue", "desk", "officechair"],
  { board: "#3a6ab0", ink: "#ffffff", neon: true, outside: ["flagpole"], where: TOWN }
);
b("carwash", "CAR WASH", "tin", "tiles", [1, 1], [10, 12], "flat", "none", ["carwash", "sedan"], {
  board: "#5ab0ff",
  ink: "#ffffff",
  where: TOWN,
});

/* Health */
b("hospital", "HOSPITAL", "tiles", "tiles", [3, 5], [12, 15], "helipad", "grid", HOSP, {
  walls: ["xraybox", "wallclock"],
  board: "#ffffff",
  ink: "#c8302a",
  outside: ["ambulance"],
  where: TOWN,
});
b(
  "clinic",
  "CLINIC",
  "plaster",
  "tiles",
  [1, 2],
  [9, 11],
  "flat",
  "grid",
  ["hospitalbed", "ivstand", "medicineshelf", "desk", "wheelchair"],
  { walls: ["xraybox"], board: "#5bbf3a", ink: "#ffffff", where: [...TOWN, ...RURAL] }
);
b(
  "dentist",
  "DENTIST",
  "plaster",
  "tiles",
  [1, 2],
  [8, 10],
  "flat",
  "grid",
  ["dentistchair", "medicineshelf", "sinkcounter", "reception"],
  { walls: ["xraybox", "painting"], board: "#5ab0ff", ink: "#ffffff" }
);
b(
  "vet",
  "VET",
  "siding",
  "tiles",
  [1, 1],
  [8, 10],
  "pitched",
  "grid",
  ["petcage", "petcage", "medicineshelf", "reception"],
  { board: "#5bbf3a", ink: "#ffffff", where: [...TOWN, ...RURAL] }
);
b("biolab", "BIOLAB", "steel", "tiles", [2, 3], [10, 13], "solar", "slits", LAB, {
  walls: ["xraybox", "whiteboard"],
  board: "#8aff6a",
  ink: "#1e1e24",
  where: [...IND, "hive", ...MIL],
  guards: 2,
});

/* Civic */
b(
  "police",
  "POLICE",
  "concrete",
  "tiles",
  [2, 3],
  [11, 14],
  "antenna",
  "grid",
  ["policedesk", "policedesk", "jailbars", "lockers", "filingcabinet", "watercooler"],
  {
    walls: ["wallclock", "whiteboard"],
    board: "#3a6ab0",
    ink: "#ffffff",
    outside: ["policecar"],
    where: TOWN,
    guards: 2,
  }
);
b(
  "firestation",
  "FIRE",
  "brick",
  "concrete",
  [2, 2],
  [12, 15],
  "bell",
  "pairs",
  ["firetruck", "firehose", "lockers", "firepole"],
  {
    upper: ["bunkbed", "bunkbed", "tv", "sofa", "kitchencounter"],
    walls: ["firehose", "wallclock"],
    board: "#c8302a",
    ink: "#ffffff",
    where: TOWN,
  }
);
b(
  "postoffice",
  "POST",
  "brick",
  "plaster",
  [1, 2],
  [10, 12],
  "parapet",
  "arched",
  ["mailsort", "mailbags", "reception", "mailbags"],
  { walls: ["wallclock"], board: "#c8302a", ink: "#ffcc33", outside: ["mailbox"], where: [...TOWN, ...RURAL] }
);
b(
  "townhall",
  "CITY HALL",
  "marble",
  "marble",
  [2, 3],
  [13, 16],
  "dome",
  "arched",
  ["reception", "desk", "officechair", "bookshelf", "bust"],
  {
    walls: ["painting", "wallclock"],
    floor: "marble",
    board: "#1e2a3a",
    ink: "#ffd23f",
    outside: ["flagpole", "statue"],
    where: TOWN,
  }
);
b(
  "courthouse",
  "COURT",
  "marble",
  "parquet",
  [2, 3],
  [12, 15],
  "parapet",
  "arched",
  ["judgebench", "bench", "bench", "bust", "bookshelf"],
  { walls: ["wallclock", "painting"], floor: "parquet", board: "#1e1e24", ink: "#ffffff", where: TOWN }
);
b(
  "library",
  "LIBRARY",
  "sandstone",
  "parquet",
  [2, 3],
  [11, 14],
  "dome",
  "arched",
  ["bookshelf", "bookshelf", "bookshelf", "desk", "globe", "armchair"],
  { walls: ["wallclock", "painting"], floor: "parquet", board: "#3a2410", ink: "#ffd23f", where: TOWN }
);
b(
  "museum",
  "MUSEUM",
  "marble",
  "marble",
  [2, 3],
  [14, 18],
  "dome",
  "arched",
  ["dinofossil", "bust", "pedestalvase", "sarcophagus", "amphora"],
  {
    walls: ["painting", "fossilfish"],
    floor: "marble",
    board: "#1e1e24",
    ink: "#ffffff",
    outside: ["statue"],
    where: [...TOWN, "temple"],
  }
);
b(
  "gallery",
  "GALLERY",
  "plaster",
  "plaster",
  [2, 2],
  [10, 13],
  "sawtooth",
  "none",
  ["easel", "bust", "pedestalvase", "bench"],
  { walls: ["painting", "painting"], floor: "parquet", board: "#ffffff", ink: "#1e1e24" }
);
b(
  "school",
  "SCHOOL",
  "brick",
  "plaster",
  [2, 3],
  [13, 16],
  "bell",
  "grid",
  ["schooldesk", "schooldesk", "schooldesk", "globe", "lockers"],
  {
    walls: ["chalkboard", "wallclock"],
    board: "#3a6ab0",
    ink: "#ffcc33",
    outside: ["schoolbus", "hoop"],
    where: [...TOWN, ...RURAL],
  }
);
b("university", "LAB", "sandstone", "tiles", [3, 4], [12, 15], "spire", "arched", LAB, {
  walls: ["chalkboard", "whiteboard"],
  board: "#5a2a4a",
  ink: "#ffd23f",
  where: TOWN,
});
b(
  "observatory",
  "STARS",
  "concrete",
  "plaster",
  [2, 2],
  [9, 11],
  "dome",
  "porthole",
  ["telescope", "desk", "bookshelf", "globe"],
  { walls: ["painting"], board: "#1e1e3a", ink: "#ffd479", where: [...COLD, ...HOT, "fort"] }
);
b(
  "embassy",
  "EMBASSY",
  "marble",
  "parquet",
  [3, 4],
  [12, 14],
  "parapet",
  "arched",
  ["desk", "reception", "bookshelf", "safe", "pottedplant"],
  {
    walls: ["painting", "wallclock"],
    floor: "parquet",
    board: "#1e2a3a",
    ink: "#ffffff",
    outside: ["flagpole"],
    guards: 2,
  }
);
b("chapel", "CHAPEL", "stone", "plaster", [1, 1], [10, 12], "spire", "arched", ["bench", "bench", "piano", "lamp"], {
  board: "#f4f0e8",
  ink: "#3a2410",
  where: [...TOWN, ...RURAL],
});

/* Sport and fun */
b(
  "gym",
  "GYM",
  "concrete",
  "tiles",
  [1, 2],
  [11, 14],
  "flat",
  "strip",
  ["treadmill", "treadmill", "weightbench", "dumbbells", "punchbag", "lockers"],
  { walls: ["mirror", "wallclock"], board: "#c8302a", ink: "#ffffff", neon: true }
);
b(
  "boxing",
  "BOXING",
  "darkbrick",
  "concrete",
  [1, 2],
  [12, 14],
  "flat",
  "slits",
  ["boxingring", "punchbag", "weightbench", "lockers"],
  { walls: ["painting"], board: "#1e1e24", ink: "#c8302a" }
);
b("pool", "POOL", "tiles", "tiles", [1, 1], [14, 18], "sawtooth", "strip", ["bench", "lockers", "lifering", "shower"], {
  walls: ["lifering", "wallclock"],
  board: "#5ab0ff",
  ink: "#ffffff",
});
b(
  "cinema",
  "CINEMA",
  "darkbrick",
  "carpet",
  [2, 2],
  [12, 15],
  "neon",
  "none",
  ["cinemaseats", "cinemaseats", "projector", "curtains", "vendingsnack"],
  { walls: [], board: "#c8302a", ink: "#ffd23f", neon: true }
);
b(
  "theatre",
  "THEATRE",
  "marble",
  "carpet",
  [2, 3],
  [12, 15],
  "dome",
  "arched",
  ["curtains", "cinemaseats", "piano", "speakerstack", "micstand"],
  { walls: ["painting"], board: "#5a1a2a", ink: "#ffd23f", neon: true }
);
b("arcadehall", "ARCADE", "concrete", "carpet", [1, 1], [11, 14], "neon", "none", FUN, {
  walls: [],
  board: "#1e1e3a",
  ink: "#5ff0e0",
  neon: true,
});
b(
  "bowling",
  "BOWLING",
  "tin",
  "parquet",
  [1, 1],
  [14, 18],
  "neon",
  "none",
  ["bowlingpins", "bowlingpins", "ballrack", "barstool", "jukebox"],
  { walls: [], board: "#3a6ab0", ink: "#ffffff", neon: true }
);
b(
  "casino",
  "CASINO",
  "marble",
  "carpet",
  [2, 3],
  [13, 16],
  "neon",
  "arched",
  ["slotmachine", "slotmachine", "roulette", "cardtable", "barcounter"],
  { walls: ["painting"], board: "#1e1e24", ink: "#ffd23f", neon: true, guards: 2 }
);
b(
  "karaokebar",
  "KARAOKE",
  "stucco",
  "carpet",
  [1, 2],
  [8, 11],
  "neon",
  "slits",
  ["karaoke", "micstand", "sofa", "barcounter", "discoball"],
  { walls: [], board: "#8a4ac8", ink: "#ff6fae", neon: true, where: [...CITY, "monsoon"] }
);
b("studio", "STUDIO", "concrete", "carpet", [1, 2], [9, 11], "flat", "slits", STAGE, {
  walls: ["painting"],
  board: "#1e1e24",
  ink: "#ff8a1a",
});
b("icerink", "ICE RINK", "steel", "tiles", [1, 1], [16, 20], "sawtooth", "strip", ["bench", "lockers", "vendingsoda"], {
  board: "#a8d8f0",
  ink: "#1e2a3a",
  where: [...CITY, ...COLD],
});

/* Living */
b(
  "hotel",
  "HOTEL",
  "brick",
  "carpet",
  [4, 7],
  [11, 14],
  "tank",
  "pairs",
  ["bed", "wardrobe", "lamp", "tv", "armchair"],
  {
    items: ["reception", "deskbell", "luggagecart", "sofa", "pottedplant"],
    upper: ["bed", "wardrobe", "lamp", "tv", "armchair", "luggagecart"],
    walls: ["painting", "mirror"],
    floor: "carpet",
    board: "#8a3a4a",
    ink: "#ffd23f",
    neon: true,
    where: [...TOWN, ...HOT],
  }
);
b(
  "motel",
  "MOTEL",
  "stucco",
  "carpet",
  [1, 2],
  [12, 16],
  "neon",
  "pairs",
  ["bed", "bed", "tv", "lamp", "vendingsoda"],
  {
    walls: ["painting"],
    board: "#ff6fae",
    ink: "#ffffff",
    neon: true,
    outside: ["sedan"],
    where: [...HOT, "snow", "city"],
  }
);
b("apartments", "APARTMENTS", "brick", "plaster", [3, 6], [10, 13], "tank", "pairs", HOME, {
  walls: HOME_W,
  upper: [...HOME, ...BATH],
  board: "#3a3f4f",
  ink: "#ffffff",
  where: TOWN,
});
b("townhouse", undefined, "siding", "plaster", [2, 3], [7, 9], "pitched", "pairs", HOME, {
  walls: HOME_W,
  upper: [...HOME, ...BATH, "crib"],
  top: ["chimney"],
  where: [...TOWN, ...RURAL],
});
b("penthouse", "RESIDENCE", "glass", "marble", [5, 8], [10, 12], "garden", "glass", HOME, {
  walls: HOME_W,
  upper: [...HOME, "piano", "aquarium"],
  floor: "marble",
  board: "#1e1e24",
  ink: "#ffd23f",
});
b(
  "hostel",
  "HOSTEL",
  "plaster",
  "plaster",
  [2, 3],
  [9, 11],
  "flat",
  "grid",
  ["bunkbed", "bunkbed", "lockers", "sofa", "kitchencounter"],
  { walls: ["painting"], board: "#5bbf3a", ink: "#ffffff", where: [...CITY, "monsoon", "jungle", "snow"] }
);
b("tenement", undefined, "darkbrick", "plaster", [3, 5], [9, 11], "chimney", "pairs", [...HOME, ...BATH], {
  walls: HOME_W,
  where: [...TOWN, "factory"],
});
b(
  "skilodge",
  "LODGE",
  "wood",
  "wood",
  [2, 3],
  [11, 14],
  "pitched",
  "pairs",
  ["sofa", "armchair", "barcounter", "lamp", "bookshelf"],
  {
    upper: ["bed", "wardrobe", "lamp"],
    walls: ["painting"],
    top: ["chimney"],
    board: "#5a3816",
    ink: "#ffffff",
    where: COLD,
  }
);
b(
  "cabin",
  undefined,
  "wood",
  "wood",
  [1, 1],
  [7, 9],
  "pitched",
  "pairs",
  ["bed", "stove", "lamp", "armchair", "woodbarrels"],
  { top: ["chimney"], where: [...COLD, "swamp", "jungle"] }
);
b(
  "stilthouse",
  undefined,
  "wood",
  "wood",
  [1, 1],
  [7, 9],
  "pitched",
  "pairs",
  ["bed", "kitchencounter", "lamp", "fishingnets"],
  { where: [...WET, "monsoon"] }
);
b(
  "baitshop",
  "BAIT",
  "tin",
  "wood",
  [1, 1],
  [8, 10],
  "flat",
  "shop",
  ["fishingnets", "chestfreezer", "register", "aquarium"],
  { board: "#5a8a5a", ink: "#ffffff", outside: ["fishingnets"], where: ["swamp", "monsoon"] }
);

/* Farms and the country */
b(
  "farmhouse",
  undefined,
  "siding",
  "plaster",
  [2, 2],
  [10, 12],
  "pitched",
  "pairs",
  ["kitchencounter", "diningtable", "stove", "armchair", "milkcans"],
  {
    upper: ["bed", "wardrobe", "crib", "lamp"],
    walls: HOME_W,
    top: ["chimney"],
    outside: ["tractor", "haybales"],
    where: RURAL,
  }
);
b(
  "barn",
  "BARN",
  "wood",
  "wood",
  [1, 2],
  [12, 15],
  "pitched",
  "none",
  ["haybales", "haybales", "tractor", "trough", "milkcans"],
  { upper: ["haybales", "haybales"], board: "#c8302a", ink: "#ffffff", where: RURAL }
);
b(
  "greenhouse",
  undefined,
  "glass",
  "glass",
  [1, 1],
  [12, 16],
  "sawtooth",
  "glass",
  ["greenhouseplants", "greenhouseplants", "pottedplant", "workbench"],
  { where: RURAL }
);
b(
  "tradingpost",
  "TRADING",
  "wood",
  "wood",
  [1, 1],
  [9, 11],
  "pitched",
  "shop",
  ["crates", "woodbarrels", "register", "weaponrack", "producecrates"],
  { board: "#5a3816", ink: "#ffcc33", awning: "#c8a060", where: [...WET, ...HOT], guards: 2 }
);

/* Industry */
b("factory", "FACTORY", "brick", "concrete", [2, 3], [14, 18], "sawtooth", "strip", FACT, {
  top: ["chimney", "chimney", "vent"],
  board: "#3a3f4f",
  ink: "#ffcc33",
  where: IND,
  guards: 2,
});
b("warehouse", "STORAGE", "tin", "concrete", [1, 2], [14, 18], "flat", "slits", STORE, {
  board: "#3a6ab0",
  ink: "#ffffff",
  where: [...IND, "fort", ...HOT],
});
b(
  "brewery",
  "BREWERY",
  "brick",
  "concrete",
  [2, 3],
  [12, 15],
  "chimney",
  "arched",
  ["brewvat", "brewvat", "woodbarrels", "conveyor", "controlpanel"],
  { board: "#c8a040", ink: "#1e1e24", where: IND }
);
b(
  "printworks",
  "PRINT",
  "brick",
  "concrete",
  [2, 2],
  [11, 13],
  "sawtooth",
  "strip",
  ["printpress", "paperrolls", "conveyor", "pallets"],
  { board: "#1e1e24", ink: "#ffffff", where: IND }
);
b(
  "sawmill",
  "SAWMILL",
  "wood",
  "wood",
  [1, 1],
  [12, 15],
  "pitched",
  "none",
  ["sawblade", "logs", "logs", "workbench", "forklift"],
  { board: "#5a3816", ink: "#ffcc33", where: [...COLD, "jungle", "swamp"] }
);
b(
  "powerstation",
  "POWER",
  "concrete",
  "steel",
  [2, 3],
  [13, 16],
  "chimney",
  "slits",
  ["turbine", "transformer", "controlpanel", "boiler"],
  { top: ["chimney", "chimney"], board: "#ffcc33", ink: "#1e1e24", where: [...IND, "volcano"], guards: 2 }
);
b(
  "waterworks",
  "WATER",
  "concrete",
  "tiles",
  [1, 2],
  [12, 14],
  "tank",
  "strip",
  ["gaspipe", "controlpanel", "boiler", "valve"],
  { board: "#5ab0ff", ink: "#ffffff", where: IND }
);
b(
  "recycling",
  "RECYCLE",
  "tin",
  "concrete",
  [1, 1],
  [14, 17],
  "flat",
  "none",
  ["bales", "bales", "conveyor", "forklift", "dumpster"],
  { board: "#5bbf3a", ink: "#ffffff", where: IND }
);
b(
  "refinery",
  "REFINERY",
  "steel",
  "steel",
  [2, 3],
  [12, 15],
  "chimney",
  "slits",
  ["boiler", "gaspipe", "controlpanel", "gascylinders", "drums"],
  { top: ["chimney", "vent", "vent"], board: "#ff8a1a", ink: "#1e1e24", where: [...IND, ...HOT], guards: 2 }
);
b(
  "chemplant",
  "CHEMICALS",
  "steel",
  "tiles",
  [2, 2],
  [12, 14],
  "solar",
  "slits",
  ["toxicdrum", "labbench", "boiler", "gascylinders", "controlpanel"],
  { board: "#8aff6a", ink: "#1e1e24", where: [...IND, "hive"], guards: 2 }
);

/* Transport */
b(
  "trainstation",
  "STATION",
  "brick",
  "tiles",
  [2, 2],
  [16, 20],
  "sawtooth",
  "arched",
  ["platformbench", "ticketmachine", "departureboard", "vendingsnack", "bench"],
  { walls: ["wallclock"], board: "#1e2a3a", ink: "#ffffff", where: TOWN }
);
b("busdepot", "BUSES", "concrete", "concrete", [1, 1], [16, 20], "flat", "none", ["bus", "toolchest", "tirerack"], {
  board: "#ff8a1a",
  ink: "#ffffff",
  where: TOWN,
});
b(
  "parkinggarage",
  "PARKING",
  "concrete",
  "concrete",
  [3, 5],
  [14, 17],
  "flat",
  "strip",
  ["sedan", "sedanblue", "hatchback", "van"],
  { board: "#3a6ab0", ink: "#ffffff", where: TOWN }
);
b(
  "airportterminal",
  "AIRPORT",
  "glass",
  "tiles",
  [2, 2],
  [16, 20],
  "flat",
  "glass",
  ["departureboard", "platformbench", "luggagecart", "ticketmachine", "vendingsoda"],
  { board: "#3a6ab0", ink: "#ffffff", where: [...TOWN, ...HOT] }
);
b(
  "controltower",
  "TOWER",
  "concrete",
  "tiles",
  [4, 6],
  [6, 7],
  "antenna",
  "porthole",
  ["radarconsole", "controlpanel", "desk"],
  { board: "#3a3f4f", ink: "#8aff6a", where: [...TOWN, ...MIL, ...HOT] }
);
b(
  "harbourmaster",
  "HARBOUR",
  "siding",
  "wood",
  [2, 2],
  [9, 11],
  "pitched",
  "porthole",
  ["desk", "maptable", "radio", "lifering"],
  {
    walls: ["lifering", "painting"],
    board: "#3a6ab0",
    ink: "#ffffff",
    outside: ["anchor", "capstan"],
    where: [...TOWN, "monsoon"],
  }
);
b("lighthouse", undefined, "stone", "plaster", [3, 4], [6, 6], "dome", "porthole", ["lamp", "maptable", "radio"], {
  where: ["monsoon", "jungle", "city", "fort"],
});

/* Military */
b("commandhq", "COMMAND", "concrete", "concrete", [3, 4], [13, 16], "antenna", "slits", ARMY, {
  walls: ["whiteboard", "wallclock"],
  board: "#4a5a30",
  ink: "#ffcc33",
  outside: ["flagpole", "hedgehog"],
  where: MIL,
  guards: 3,
});
b(
  "armory",
  "ARMORY",
  "darkbrick",
  "concrete",
  [1, 2],
  [10, 13],
  "flat",
  "slits",
  ["weaponrack", "weaponrack", "ammocrate", "ammocrate", "gascylinders"],
  { board: "#3a3f4f", ink: "#ff3b3b", where: [...MIL, "city"], guards: 3 }
);
b(
  "barracks",
  "BARRACKS",
  "tin",
  "wood",
  [1, 2],
  [12, 15],
  "pitched",
  "pairs",
  ["bunkbed", "bunkbed", "lockers", "weaponrack"],
  { board: "#4a5a30", ink: "#ffffff", where: [...MIL, ...HOT, "jungle", "snow"], guards: 2 }
);
b(
  "radarstation",
  "RADAR",
  "concrete",
  "steel",
  [2, 2],
  [9, 11],
  "antenna",
  "slits",
  ["radarconsole", "radarconsole", "controlpanel", "serverrack"],
  { top: ["dish", "dish"], board: "#3a3f4f", ink: "#8aff6a", where: [...MIL, ...COLD, ...HOT], guards: 2 }
);
b(
  "hangar",
  "HANGAR",
  "tin",
  "concrete",
  [1, 1],
  [18, 22],
  "sawtooth",
  "none",
  ["forklift", "toolchest", "engine", "drums", "missiletrailer"],
  { board: "#4a5a30", ink: "#ffcc33", where: MIL, guards: 2 }
);
b("bunkerhq", undefined, "concrete", "concrete", [1, 1], [12, 14], "parapet", "slits", ARMY, {
  where: [...MIL, "snow", "desert"],
  guards: 3,
});

/* Traditional architecture from around the world */
const ASIA: Backdrop[] = [
  "city",
  "monsoon",
  "jungle",
  "temple",
  "deepjungle",
  "snow",
  "sakura",
  "archipelago",
  "mangrove",
];
const EUROPE: Backdrop[] = ["city", "fort", "snow", "factory", "swamp", "autumn", "farmland", "megacity", "tundra"];
const MIDEAST: Backdrop[] = ["desert", "canyon", "city", "savanna", "badlands"];
const AMERICAS: Backdrop[] = [
  "city",
  "desert",
  "canyon",
  "jungle",
  "swamp",
  "savanna",
  "farmland",
  "archipelago",
  "megacity",
];
b(
  "pagoda",
  undefined,
  "wood",
  "shoji",
  [3, 5],
  [7, 9],
  "pagoda",
  "pairs",
  ["diningtable", "pottedplant", "lamp", "rug"],
  { roofMat: "jade", where: ASIA }
);
b(
  "jpcastle",
  undefined,
  "plaster",
  "wood",
  [3, 4],
  [9, 11],
  "pagoda",
  "slits",
  ["weaponrack", "maptable", "lamp", "bookshelf"],
  { roofMat: "slate", where: ASIA, guards: 2 }
);
b(
  "machiya",
  undefined,
  "wood",
  "shoji",
  [2, 2],
  [6, 8],
  "pitched",
  "pairs",
  ["kitchencounter", "diningtable", "lamp", "pottedplant"],
  { where: ASIA }
);
b(
  "ramen",
  "RAMEN",
  "wood",
  "shoji",
  [1, 2],
  [7, 9],
  "pitched",
  "shop",
  ["barcounter", "barstool", "stove", "register"],
  { walls: ["menuboard"], board: "#c8302a", ink: "#ffffff", awning: "#1e1e24", where: ASIA }
);
b(
  "templehall",
  undefined,
  "brick",
  "plaster",
  [2, 3],
  [10, 13],
  "pagoda",
  "arched",
  ["bust", "pedestalvase", "lamp", "bench"],
  { roofMat: "terracotta", where: ASIA }
);
b(
  "hanok",
  undefined,
  "wood",
  "plaster",
  [1, 1],
  [8, 10],
  "pitched",
  "pairs",
  ["diningtable", "rug", "lamp", "kitchencounter"],
  { roofMat: "slate", where: ASIA }
);
b("wat", undefined, "plaster", "plaster", [1, 2], [8, 11], "stupa", "arched", ["bust", "pedestalvase", "lamp"], {
  where: [...ASIA, "temple"],
});
b("thaitemple", undefined, "stucco", "plaster", [1, 2], [9, 11], "pagoda", "arched", ["bust", "lamp", "rug"], {
  roofMat: "terracotta",
  where: [...ASIA, "temple"],
});
b(
  "palace",
  "PALACE",
  "limestone",
  "marble",
  [2, 3],
  [12, 15],
  "onion",
  "arched",
  ["bust", "pedestalvase", "rug", "armchair", "painting"],
  { roofMat: "gold", walls: ["painting", "mirror"], board: "#3a1a4a", ink: "#ffd23f", where: [...ASIA, ...MIDEAST] }
);
b("domehall", undefined, "azulejo", "tiles", [1, 2], [10, 13], "minaret", "arched", ["rug", "lamp", "bench"], {
  roofMat: "azulejo",
  where: MIDEAST,
});
b(
  "riad",
  undefined,
  "stucco",
  "azulejo",
  [2, 2],
  [9, 11],
  "flat",
  "arched",
  ["rug", "pottedplant", "diningtable", "lamp"],
  { top: ["roofgarden"], where: MIDEAST }
);
b(
  "bazaar",
  "BAZAAR",
  "adobe",
  "terracotta",
  [1, 1],
  [14, 18],
  "minaret",
  "arched",
  ["marketstall", "marketstall", "rug", "jewelcase"],
  { roofMat: "terracotta", board: "#c8302a", ink: "#ffd23f", where: MIDEAST }
);
b(
  "pyramidtemple",
  undefined,
  "sandstone",
  "sandstone",
  [1, 1],
  [12, 16],
  "pyramid",
  "none",
  ["sarcophagus", "amphora", "treasurechest", "bust"],
  { where: ["desert", "canyon", "temple"] }
);
b("mudmosque", undefined, "adobe", "adobe", [2, 2], [12, 15], "castle", "slits", ["rug", "lamp", "bench"], {
  roofMat: "adobe",
  where: ["desert", "canyon"],
});
b(
  "greektemple",
  undefined,
  "limestone",
  "marble",
  [1, 1],
  [12, 15],
  "columns",
  "none",
  ["bust", "pedestalvase", "statue", "amphora"],
  { where: [...EUROPE, "temple", "desert"] }
);
b(
  "romanvilla",
  undefined,
  "stucco",
  "plaster",
  [1, 2],
  [11, 13],
  "pitched",
  "arched",
  ["diningtable", "amphora", "bust", "pottedplant"],
  { roofMat: "terracotta", where: [...EUROPE, "desert"] }
);
b(
  "palazzo",
  undefined,
  "stucco",
  "marble",
  [3, 4],
  [10, 12],
  "parapet",
  "arched",
  ["armchair", "painting", "bookshelf", "piano"],
  { walls: ["painting", "mirror"], where: EUROPE }
);
b(
  "hacienda",
  undefined,
  "adobe",
  "plaster",
  [1, 2],
  [11, 13],
  "pitched",
  "arched",
  ["diningtable", "rug", "woodbarrels", "lamp"],
  { roofMat: "terracotta", where: AMERICAS }
);
b("mission", undefined, "adobe", "plaster", [1, 1], [10, 12], "bell", "arched", ["bench", "bench", "lamp"], {
  where: AMERICAS,
});
b("azulejohouse", undefined, "azulejo", "plaster", [2, 3], [7, 9], "pitched", "pairs", HOME, {
  roofMat: "terracotta",
  walls: HOME_W,
  where: EUROPE,
});
b("haussmann", undefined, "limestone", "parquet", [4, 6], [10, 13], "mansard", "pairs", HOME, {
  walls: HOME_W,
  upper: [...HOME, "piano"],
  floor: "parquet",
  where: EUROPE,
});
b("canalhouse", undefined, "brick", "plaster", [3, 4], [6, 8], "stepgable", "pairs", HOME, {
  walls: HOME_W,
  where: EUROPE,
});
b(
  "guildhall",
  "GUILD",
  "brick",
  "parquet",
  [3, 3],
  [9, 11],
  "stepgable",
  "arched",
  ["bookshelf", "desk", "painting", "bust"],
  { walls: ["painting"], board: "#3a2410", ink: "#ffd23f", where: EUROPE }
);
b("tudor", undefined, "timber", "plaster", [2, 2], [8, 10], "pitched", "pairs", HOME, {
  roofMat: "slate",
  walls: HOME_W,
  top: ["chimney"],
  where: EUROPE,
});
b("clocktower", undefined, "limestone", "parquet", [3, 4], [8, 10], "clock", "arched", ["desk", "bookshelf", "bust"], {
  walls: ["wallclock"],
  where: EUROPE,
});
b(
  "towerhouse",
  undefined,
  "stone",
  "plaster",
  [3, 4],
  [7, 9],
  "castle",
  "slits",
  ["weaponrack", "bookshelf", "lamp", "woodbarrels"],
  { where: EUROPE, guards: 2 }
);
b(
  "schloss",
  undefined,
  "plaster",
  "marble",
  [3, 4],
  [12, 15],
  "castle",
  "arched",
  ["armchair", "painting", "bookshelf", "bust"],
  { roofMat: "slate", walls: ["painting"], where: EUROPE }
);
b(
  "chalet",
  undefined,
  "wood",
  "wood",
  [2, 2],
  [9, 11],
  "chalet",
  "pairs",
  ["armchair", "stove", "lamp", "woodbarrels"],
  { top: ["chimney"], where: ["snow", "fort"] }
);
b("stavechurch", undefined, "wood", "wood", [2, 2], [8, 10], "pagoda", "slits", ["bench", "lamp"], {
  roofMat: "slate",
  where: ["snow", "fort"],
});
b("onionchurch", undefined, "stucco", "plaster", [2, 2], [10, 12], "onion", "arched", ["bench", "lamp", "painting"], {
  roofMat: "jade",
  where: ["snow", "fort", "city"],
});
b("kremlintower", undefined, "brick", "brick", [4, 5], [6, 7], "pagoda", "slits", ["weaponrack", "lamp"], {
  roofMat: "jade",
  where: ["snow", "fort"],
  guards: 2,
});
b("brutalist", undefined, "concrete", "plaster", [5, 7], [11, 14], "brutalist", "strip", HOME, {
  walls: HOME_W,
  where: [...EUROPE, "factory"],
});
b("brownstone", undefined, "darkbrick", "parquet", [3, 4], [7, 9], "parapet", "pairs", HOME, {
  walls: HOME_W,
  where: ["city"],
});
b("decotower", "DECO", "limestone", "marble", [5, 8], [10, 12], "artdeco", "strip", OFFICE, {
  walls: OFFICE_W,
  board: "#1e1e24",
  ink: "#ffd23f",
  where: ["city"],
});
b("colonial", undefined, "siding", "plaster", [2, 2], [9, 11], "pitched", "arched", [...HOME, "umbrellatable"], {
  roofMat: "tin",
  walls: HOME_W,
  where: [...AMERICAS, "monsoon"],
});
b("favela", undefined, "stucco", "plaster", [3, 4], [5, 6], "flat", "pairs", ["bed", "tv", "kitchencounter", "lamp"], {
  top: ["solarpanels", "acunit"],
  where: AMERICAS,
});
b("igloo", undefined, "snow", "snow", [1, 1], [5, 6], "igloo", "none", ["lamp", "fishingnets"], { where: ["snow"] });
b("ger", undefined, "plaster", "carpet", [1, 1], [6, 8], "dome", "none", ["rug", "stove", "lamp"], {
  roofMat: "plaster",
  where: ["snow", "canyon", "desert"],
});
b("whitehouse", undefined, "plaster", "plaster", [1, 2], [7, 9], "dome", "pairs", HOME, {
  roofMat: "azulejo",
  where: ["desert", "city", "monsoon"],
});
b(
  "homestead",
  undefined,
  "tin",
  "wood",
  [1, 1],
  [10, 12],
  "pitched",
  "pairs",
  ["armchair", "stove", "woodbarrels", "lamp"],
  { where: ["desert", "canyon", "jungle"] }
);
b("tubehouse", undefined, "stucco", "tiles", [4, 5], [5, 6], "flat", "pairs", HOME, {
  top: ["acunit", "solarpanels"],
  where: ASIA,
});
b(
  "shophouse",
  "TRADERS",
  "stucco",
  "tiles",
  [2, 3],
  [8, 10],
  "pitched",
  "shop",
  ["crates", "producecrates", "register", "lamp"],
  { roofMat: "terracotta", board: "#c8302a", ink: "#ffd23f", awning: "#3a8a3a", where: ASIA }
);
b("neontenement", "HOTEL", "concrete", "tiles", [5, 8], [9, 11], "neon", "pairs", HOME, {
  board: "#1a1424",
  ink: "#ff6fae",
  neon: true,
  where: ["city", "monsoon"],
});
b(
  "grandstation",
  "CENTRAL",
  "limestone",
  "tiles",
  [2, 2],
  [18, 22],
  "sawtooth",
  "arched",
  ["platformbench", "departureboard", "ticketmachine", "bench"],
  { walls: ["wallclock"], board: "#1e2a3a", ink: "#ffd23f", where: ["city", "factory", "fort"] }
);
b(
  "reactor",
  "REACTOR",
  "concrete",
  "steel",
  [2, 2],
  [12, 14],
  "dome",
  "slits",
  ["controlpanel", "serverrack", "boiler", "transformer", "toxicdrum"],
  { roofMat: "concrete", walls: ["xraybox"], board: "#ffcc33", ink: "#1e1e24", where: [], guards: 2 }
);

export const BUILDINGS: BuildingSpec[] = SPECS;
export const SPEC_BY_KEY = new Map(SPECS.map((s) => [s.key, s]));

/** The buildings that stand in this kind of land; if there are none, the low ones from the common outdoor backdrops. */
export function buildingsFor(kind: Backdrop) {
  const here = SPECS.filter((s) => s.where.includes(kind));
  return here.length ? here : SPECS.filter((s) => s.where.some((w) => ALL_OUT.includes(w)) && s.floors[1] <= 2);
}
