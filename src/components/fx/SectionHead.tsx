import ScrambleText from "./ScrambleText";
import Reveal from "./Reveal";

/** `[ 01 ] ———————— WHO I AM` */
export default function SectionHead({ index, label }: { index: string; label: string }) {
  return (
    <div className="section-head">
      <span className="section-head__idx">[ {index} ]</span>
      <Reveal variant="line" className="section-head__rule" />
      <ScrambleText text={label} className="section-head__lbl" hover={false} />
    </div>
  );
}
