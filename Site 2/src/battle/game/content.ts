/**
 * The content registries and the links between them. Defs name each other by string id (a unit's weapon, a
 * weapon's projectile); at boot `freeze()` checks every reference and resolves it into typed index tables, so hot
 * code never looks a string up and a typo fails at boot instead of mid-battle.
 */
import { Registry, type ReadonlyRegistry } from "@/engine";
import type { ArtDef, EffectDef, ProjectileDef, UnitDef, WeaponDef } from "./defs";

/** Def indices resolved at boot (0 = none), indexed by the owning def's index. */
export interface Links {
  readonly unitArt: Uint16Array;
  readonly unitWeapon: Uint16Array;
  readonly unitDeathFx: Uint16Array;
  readonly weaponProjectile: Uint16Array;
  readonly weaponMuzzleFx: Uint16Array;
  readonly projectileArt: Uint16Array;
  readonly projectileTrail: Uint16Array;
  readonly projectileImpact: Uint16Array;
  readonly projectileBlastFx: Uint16Array;
}

export class Content {
  readonly units = new Registry<UnitDef>("units");
  readonly weapons = new Registry<WeaponDef>("weapons");
  readonly projectiles = new Registry<ProjectileDef>("projectiles");
  readonly effects = new Registry<EffectDef>("effects");
  readonly art = new Registry<ArtDef>("art");
  private linked: Links | null = null;

  /** The resolved references; only after `freeze()`. */
  get links(): Links {
    if (!this.linked) throw new Error("Content: links are only available after boot");
    return this.linked;
  }

  get frozen(): boolean {
    return this.linked !== null;
  }

  /** Resolve every reference, then freeze every registry. Throws naming the first unknown id. */
  freeze(): Links {
    if (this.linked) return this.linked;
    const ref = <D>(reg: ReadonlyRegistry<D>, id: string | undefined, owner: string): number => {
      if (id === undefined) return 0;
      const k = reg.index(id);
      if (k === 0) throw new Error(`Content: ${owner} refers to unknown ${reg.name} "${id}"`);
      return k;
    };
    const table = <D>(reg: ReadonlyRegistry<D>, pick: (def: D, owner: string) => number): Uint16Array => {
      const out = new Uint16Array(reg.size + 1);
      for (let k = 1; k <= reg.size; k++) out[k] = pick(reg.get(k), `${reg.name} "${reg.id(k)}"`);
      return out;
    };
    const { units, weapons, projectiles, effects, art } = this;
    this.linked = {
      unitArt: table(units, (d, o) => ref(art, d.art, o)),
      unitWeapon: table(units, (d, o) => ref(weapons, d.weapon, o)),
      unitDeathFx: table(units, (d, o) => ref(effects, d.deathFx, o)),
      weaponProjectile: table(weapons, (d, o) => ref(projectiles, d.projectile, o)),
      weaponMuzzleFx: table(weapons, (d, o) => ref(effects, d.muzzleFx, o)),
      projectileArt: table(projectiles, (d, o) => ref(art, d.art, o)),
      projectileTrail: table(projectiles, (d, o) => ref(effects, d.trail, o)),
      projectileImpact: table(projectiles, (d, o) => ref(effects, d.impact, o)),
      projectileBlastFx: table(projectiles, (d, o) =>
        ref(effects, d.explode ? (d.explode.fx ?? "blast.fire") : undefined, o)
      ),
    };
    for (const reg of [units, weapons, projectiles, effects, art] as const) reg.freeze();
    return this.linked;
  }
}
