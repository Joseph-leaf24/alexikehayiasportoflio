/**
 * The features a battle is made of, installed in this order. A feature is a folder whose `index.ts` exports
 * `install(game)`; adding one is one line here. Order matters only where one feature reads another's resources at
 * install time, and for event listeners of the same type (they run in subscription order). `flow` goes last: it
 * starts the first level once everything else is ready.
 */
import { install as art } from "../art";
import { install as enemies } from "../enemies";
import { install as flow } from "../flow";
import { install as level } from "../level";
import { install as physics } from "../physics";
import { install as render } from "../render";
import { install as squad } from "../squad";
import { install as world } from "../world";
import type { Feature } from "./options";

export const INSTALLS: readonly Feature[] = [
  { name: "art", install: art },
  { name: "render", install: render },
  { name: "physics", install: physics },
  { name: "level", install: level },
  { name: "squad", install: squad },
  { name: "enemies", install: enemies },
  { name: "world", install: world },
  { name: "flow", install: flow },
];
