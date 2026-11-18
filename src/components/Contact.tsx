import { ArrowUp, ArrowUpRight } from "lucide-react";
import { profile } from "@/data/profile";
import { scrollToTarget } from "@/lib/smoothScroll";
import SectionHead from "@/components/fx/SectionHead";
import ScrambleText from "@/components/fx/ScrambleText";
import Reveal from "@/components/fx/Reveal";

const LINKS = [
  { key: "Contact Me", value: profile.email, href: `mailto:${profile.email}`, cursor: "EMAIL" },
  { key: "LinkedIn", value: "in/alexi-kehayias", href: profile.linkedin, external: true, cursor: "OPEN" },
  { key: "GitHub", value: "@Joseph-leaf24", href: profile.github, external: true, cursor: "OPEN" },
  { key: "Download CV", value: "PDF", href: profile.cv, download: true, cursor: "PDF" },
];

export default function Contact() {
  return (
    <section id="contact" className="section section--contact" data-scene="domeTop" data-section="04|Contact">
      <div className="container-x">
        <SectionHead index="04" label="Contact" />
        <h2 className="display-xl">
          <ScrambleText text={profile.contact.title} duration={1200} />
        </h2>
        <Reveal as="p" className="contact__text" delay={150}>
          {profile.contact.text}
        </Reveal>

        <ul className="contact-list">
          {LINKS.map((l, i) => (
            <Reveal as="li" key={l.key} delay={i * 90}>
              <a
                className="contact-row"
                href={l.href}
                data-cursor={l.cursor}
                {...(l.external ? { target: "_blank", rel: "noopener noreferrer" } : {})}
                {...(l.download ? { download: "" } : {})}
              >
                <span className="contact-row__fill" aria-hidden="true" />
                <span className="contact-row__idx">0{i + 1}</span>
                <ScrambleText text={l.key} hover trigger="none" className="contact-row__key" />
                <span className="contact-row__val">{l.value}</span>
                <ArrowUpRight className="contact-row__arrow" aria-hidden="true" />
              </a>
            </Reveal>
          ))}
        </ul>
      </div>

      <footer className="site-footer container-x">
        <span>
          © {new Date().getFullYear()} {profile.name}
        </span>
        <span className="hidden sm:inline">Data Science &amp; AI Portfolio</span>
        <span className="hidden md:inline">
          {profile.location} · {profile.coords}
        </span>
        <button type="button" onClick={() => scrollToTarget(0)} className="site-footer__top">
          Back to top <ArrowUp className="h-3.5 w-3.5" />
        </button>
      </footer>
    </section>
  );
}
