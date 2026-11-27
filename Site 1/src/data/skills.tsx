import type { ComponentType } from "react";
import { DatabaseZap } from "lucide-react";
import {
  SiPython,
  SiPytorch,
  SiTensorflow,
  SiScikitlearn,
  SiApacheairflow,
  SiMlflow,
  SiDocker,
  SiFastapi,
  SiPandas,
  SiPostgresql,
  SiOpencv,
  SiHuggingface,
  SiSnowflake,
  SiTrello,
  SiLangchain,
} from "react-icons/si";

type IconType = ComponentType<{ className?: string; "aria-hidden"?: boolean | "true" | "false" }>;

/** Simple Icons dropped the Azure DevOps glyph, so it is inlined here. */
const AzureDevOpsIcon: IconType = ({ className }) => (
  <svg viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden="true">
    <path d="M0 8.877L2.247 5.91l8.405-3.416V.022l7.37 5.393L2.966 8.338v8.225L0 15.707zm24-4.45v14.651l-5.753 4.9-9.303-3.057v3.056l-5.978-7.416 15.057 1.798V5.415z" />
  </svg>
);

const RAGIcon: IconType = ({ className }) => <DatabaseZap className={className} aria-hidden="true" />;

interface Skill {
  name: string;
  Icon: IconType;
  /** Caption under the name on the Toolbox tile */
  hint?: string;
}

/** The Toolbox: drives the marquee, the tile grid and the tool count in About. */
export const SKILLS: Skill[] = [
  { name: "Python", Icon: SiPython, hint: "5+ yrs · data & backend" },
  { name: "PyTorch", Icon: SiPytorch, hint: "CV/NLP training & serving" },
  { name: "TensorFlow", Icon: SiTensorflow },
  { name: "scikit-learn", Icon: SiScikitlearn },
  { name: "Airflow", Icon: SiApacheairflow, hint: "40+ DAGs · SLAs" },
  { name: "MLflow", Icon: SiMlflow, hint: "tracking & model registry" },
  { name: "Docker", Icon: SiDocker, hint: "multi-stage builds" },
  { name: "FastAPI", Icon: SiFastapi, hint: "<100ms p95 inference" },
  { name: "Pandas", Icon: SiPandas },
  { name: "PostgreSQL", Icon: SiPostgresql },
  { name: "OpenCV", Icon: SiOpencv },
  { name: "Hugging Face", Icon: SiHuggingface },
  { name: "Snowflake", Icon: SiSnowflake, hint: "ELT & analytics" },
  { name: "Trello", Icon: SiTrello, hint: "Kanban · roadmaps" },
  { name: "Azure DevOps", Icon: AzureDevOpsIcon, hint: "Boards · Pipelines" },
  { name: "RAG", Icon: RAGIcon, hint: "Retrieval Augmented Generation" },
  { name: "LangGraph", Icon: SiLangchain, hint: "Agent workflows & graphs" },
  { name: "LangSmith", Icon: SiLangchain, hint: "LLM observability & tracing" },
];
