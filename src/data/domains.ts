import { BarChart3, Brain, Database, Rocket, type LucideIcon } from "lucide-react";
import { projectData } from "@/data/projects";
import type { CategoryId } from "@/data/profile";

export type DomainId = "engineering" | "analytics" | "science" | "deployment";

export interface ProjectLink {
  category: CategoryId;
  index: number;
}

export interface Domain {
  id: DomainId;
  /** Step in the data lifecycle */
  step: string;
  title: string;
  tagline: string;
  description: string;
  capabilities: string[];
  tools: string[];
  projects: ProjectLink[];
  /** Specialization tab to open when exploring this domain's work */
  explore: CategoryId;
  icon: LucideIcon;
}

/** The four disciplines of the data lifecycle, in the order data flows through them. */
export const domains: Domain[] = [
  {
    id: "engineering",
    step: "Collect & prepare",
    title: "Data Engineering",
    tagline: "Reliable pipelines from source to feature",
    description:
      "Building the plumbing that models and dashboards depend on: SQL-based data pipelines, orchestrated workflows and cloud data warehouses that keep data clean and flowing.",
    capabilities: [
      "SQL data modelling & pipelines",
      "Workflow orchestration with Airflow",
      "Cloud warehousing in Snowflake",
      "Cleaning, validation & feature engineering",
    ],
    tools: ["SQL", "PostgreSQL", "Apache Airflow", "Snowflake", "Pandas"],
    projects: [
      { category: "machineLearning", index: 0 },
      { category: "nlp", index: 1 },
      { category: "mlops", index: 0 },
      { category: "nlp", index: 0 },
    ],
    explore: "mlops",
    icon: Database,
  },
  {
    id: "analytics",
    step: "Understand",
    title: "Data Analytics",
    tagline: "From raw numbers to decisions",
    description:
      "Exploring data, finding the story in it and presenting it so people can act on it: interactive dashboards, KPI and SDG tracking, and statistical research.",
    capabilities: [
      "Exploratory data analysis",
      "Interactive Power BI dashboards",
      "KPI, risk & SDG monitoring",
      "Statistical testing & mixed-methods research",
    ],
    tools: ["Power BI", "Excel", "SQL", "Pandas", "SPSS", "Qualtrics"],
    projects: [
      { category: "dataAnalytics", index: 0 },
      { category: "dataAnalytics", index: 1 },
      { category: "research", index: 0 },
    ],
    explore: "dataAnalytics",
    icon: BarChart3,
  },
  {
    id: "science",
    step: "Model",
    title: "Data Science & AI",
    tagline: "Models that see, read and predict",
    description:
      "Designing, training and explaining machine-learning models, from CNN image classifiers with Grad-CAM to root segmentation, multilingual NLP and accident-risk prediction.",
    capabilities: [
      "Deep learning: CNNs & transformers",
      "Computer vision & image segmentation",
      "NLP, speech-to-text & translation",
      "Explainable & responsible AI",
    ],
    tools: ["Python", "PyTorch", "TensorFlow/Keras", "scikit-learn", "OpenCV", "Hugging Face"],
    projects: [
      { category: "computerVision", index: 0 },
      { category: "computerVision", index: 1 },
      { category: "machineLearning", index: 0 },
      { category: "nlp", index: 0 },
    ],
    explore: "computerVision",
    icon: Brain,
  },
  {
    id: "deployment",
    step: "Ship",
    title: "MLOps & Deployment",
    tagline: "Taking models out of the notebook",
    description:
      "Putting models in front of people: containerised services, APIs and apps, CI/CD pipelines and observability for LLM-powered agents.",
    capabilities: [
      "Docker & Azure ML deployment",
      "APIs & apps: FastAPI, Gradio, Streamlit",
      "CI/CD with GitHub Actions",
      "Agentic AI & LLM observability",
    ],
    tools: ["Docker", "Azure ML", "FastAPI", "GitHub Actions", "LangGraph", "LangSmith"],
    projects: [
      { category: "mlops", index: 0 },
      { category: "nlp", index: 1 },
      { category: "machineLearning", index: 0 },
    ],
    explore: "mlops",
    icon: Rocket,
  },
];

export const domainIntro = {
  title: "Domains",
  text: "Four disciplines across the data lifecycle — from raw data to models in production.",
};

/** Domains a project belongs to (used on cards and project pages). */
export function domainsFor(category: string, index: number): Domain[] {
  return domains.filter((d) => d.projects.some((p) => p.category === category && p.index === index));
}

/** Resolve a domain's project links to the actual projects. */
export const projectsOf = (d: Domain) => d.projects.map((p) => ({ ...p, project: projectData[p.category].projects[p.index] }));
