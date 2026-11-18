import { ui } from "./store";

/**
 * Generative ambient soundscape (arctic wind + a soft drone) and UI blips,
 * synthesised with WebAudio — no audio files to download. Off by default.
 */
class SoundEngine {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private fx: GainNode | null = null;
  private lastBlip = 0;
  private readonly storageKey = "ak:sound";

  get enabled() {
    return ui.get().soundOn;
  }

  /** If the visitor had sound on last time, re-enable it on their first interaction. */
  restorePreference() {
    let wanted = false;
    try {
      wanted = localStorage.getItem(this.storageKey) === "on";
    } catch {
      /* storage blocked */
    }
    if (!wanted) return;
    const resume = () => {
      window.removeEventListener("pointerdown", resume);
      window.removeEventListener("keydown", resume);
      if (!this.enabled) this.toggle(true);
    };
    window.addEventListener("pointerdown", resume, { once: true });
    window.addEventListener("keydown", resume, { once: true });
  }

  toggle(force?: boolean) {
    const next = force ?? !this.enabled;
    if (next && !this.ctx) this.build();
    if (!this.ctx || !this.master) return;
    if (this.ctx.state === "suspended") void this.ctx.resume();
    const now = this.ctx.currentTime;
    this.master.gain.cancelScheduledValues(now);
    this.master.gain.setValueAtTime(this.master.gain.value, now);
    this.master.gain.linearRampToValueAtTime(next ? 0.55 : 0, now + (next ? 1.6 : 0.5));
    ui.set({ soundOn: next });
    try {
      localStorage.setItem(this.storageKey, next ? "on" : "off");
    } catch {
      /* storage blocked */
    }
  }

  blip(kind: "hover" | "click" = "hover") {
    if (!this.enabled || !this.ctx || !this.fx) return;
    const t = this.ctx.currentTime;
    if (kind === "hover" && t - this.lastBlip < 0.045) return;
    this.lastBlip = t;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.type = kind === "hover" ? "sine" : "triangle";
    const f = kind === "hover" ? 1900 + Math.random() * 900 : 420;
    osc.frequency.setValueAtTime(f, t);
    osc.frequency.exponentialRampToValueAtTime(kind === "hover" ? f * 1.4 : 180, t + (kind === "hover" ? 0.05 : 0.22));
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(kind === "hover" ? 0.035 : 0.09, t + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, t + (kind === "hover" ? 0.08 : 0.28));
    osc.connect(g).connect(this.fx);
    osc.start(t);
    osc.stop(t + 0.32);
  }

  private build() {
    const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!AC) return;
    const ctx = new AC();
    this.ctx = ctx;

    const master = ctx.createGain();
    master.gain.value = 0;
    master.connect(ctx.destination);
    this.master = master;

    const fx = ctx.createGain();
    fx.gain.value = 1;
    fx.connect(master);
    this.fx = fx;

    // --- Wind: looping brown noise through a slowly sweeping low-pass
    const len = ctx.sampleRate * 6;
    const buffer = ctx.createBuffer(2, len, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const data = buffer.getChannelData(ch);
      let last = 0;
      for (let i = 0; i < len; i++) {
        const white = Math.random() * 2 - 1;
        last = (last + 0.02 * white) / 1.02;
        data[i] = last * 3.2;
      }
    }
    const noise = ctx.createBufferSource();
    noise.buffer = buffer;
    noise.loop = true;
    const lp = ctx.createBiquadFilter();
    lp.type = "lowpass";
    lp.frequency.value = 520;
    lp.Q.value = 0.8;
    const lfo = ctx.createOscillator();
    lfo.frequency.value = 0.06;
    const lfoGain = ctx.createGain();
    lfoGain.gain.value = 320;
    lfo.connect(lfoGain).connect(lp.frequency);
    const windGain = ctx.createGain();
    windGain.gain.value = 0.32;
    noise.connect(lp).connect(windGain).connect(master);
    noise.start();
    lfo.start();

    // --- Drone: detuned fifths, very quiet, slow tremolo
    const pad = ctx.createGain();
    pad.gain.value = 0.035;
    const padLp = ctx.createBiquadFilter();
    padLp.type = "lowpass";
    padLp.frequency.value = 900;
    pad.connect(padLp).connect(master);
    [110, 164.81, 246.94, 329.63].forEach((freq, i) => {
      const o = ctx.createOscillator();
      o.type = i % 2 ? "triangle" : "sine";
      o.frequency.value = freq;
      o.detune.value = (i - 1.5) * 6;
      const g = ctx.createGain();
      g.gain.value = 0.6 / (i + 1);
      const trem = ctx.createOscillator();
      trem.frequency.value = 0.08 + i * 0.03;
      const tremGain = ctx.createGain();
      tremGain.gain.value = 0.25 / (i + 1);
      trem.connect(tremGain).connect(g.gain);
      o.connect(g).connect(pad);
      o.start();
      trem.start();
    });
  }
}

export const sound = new SoundEngine();
