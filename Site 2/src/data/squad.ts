import type { DomainId } from "@/data/domains";
import type { CategoryId } from "@/data/profile";

/**
 * The eight players fighting through the levels behind the site. P1–P4 stand for the four domains
 * of the data lifecycle, in the same order and colours as the Domains cards, so hovering a card
 * points its player out in the scene. P5–P8 are specialist roles from the Work tabs.
 */
interface SquadMember {
  /** 1–8, as on a co-op game's HUD */
  player: number;
  /** The domain this player stands for, P1–P4 only */
  domain?: DomainId;
  /** The Work tab this player stands for, shown on that tab */
  spec?: CategoryId;
  role: string;
  /** Short label for the HUD card and the title-screen line-up */
  short: string;
  weapon: "Rifle" | "Shotgun" | "Plasma" | "Rockets" | "Sniper" | "Flamethrower" | "Grenades" | "Minigun";
  color: string;
}

export const SQUAD: SquadMember[] = [
  { player: 1, domain: "engineering", role: "Data Engineer", short: "Engineer", weapon: "Rifle", color: "#4aa3ff" },
  {
    player: 2,
    domain: "analytics",
    spec: "dataAnalytics",
    role: "Data Analyst",
    short: "Analyst",
    weapon: "Shotgun",
    color: "#ffcc33",
  },
  {
    player: 3,
    domain: "science",
    spec: "machineLearning",
    role: "Data Scientist",
    short: "Scientist",
    weapon: "Plasma",
    color: "#c77dff",
  },
  {
    player: 4,
    domain: "deployment",
    spec: "mlops",
    role: "ML Engineer",
    short: "ML Eng",
    weapon: "Rockets",
    color: "#ff6a3d",
  },
  { player: 5, spec: "computerVision", role: "CV Engineer", short: "Vision", weapon: "Sniper", color: "#5fd16a" },
  { player: 6, spec: "nlp", role: "NLP Engineer", short: "Language", weapon: "Flamethrower", color: "#3fd8e0" },
  { player: 7, spec: "research", role: "Researcher", short: "Research", weapon: "Grenades", color: "#ff6fae" },
  { player: 8, role: "LLM Engineer", short: "Agents", weapon: "Minigun", color: "#e8e2d0" },
];

/** Player colours, P1–P8. */
export const PLAYER_COLORS = SQUAD.map((s) => s.color);

/** The player who stands for a Work tab's specialization, if any. */
export const playerFor = (spec: CategoryId) => SQUAD.find((s) => s.spec === spec);
