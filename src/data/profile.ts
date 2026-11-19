import { BarChart3, Eye, Brain, MessageSquare, Search, Settings, type LucideIcon } from "lucide-react";
import { projectData } from "@/data/projects";
import type { Project } from "@/types/project";

/** Personal details and section copy used across the site. */
export const profile = {
  name: "Alexi Kehayias",
  firstName: "Alexi",
  lastName: "Kehayias",
  role: "Data Science & AI Student",
  intro:
    "Hi there! I'm Alexi, a student in Breda studying Data Science and AI with a focus on a number of areas such as Computer Vision, Natural Language processing as well as MLOps. I enjoy building innovative solutions that make a difference to companies and society. Explore my projects below to see what I've been working on!",
  statement: "I enjoy building innovative solutions that make a difference to companies and society.",
  focus: ["Data Science", "Data Engineering", "Data Analytics"],
  location: "Breda, NL",
  coords: "51.5719° N · 4.7683° E",
  timezone: "Europe/Amsterdam",
  linkedin: "https://www.linkedin.com/in/alexi-kehayias/",
  github: "https://github.com/Joseph-leaf24",
  cv: `${import.meta.env.BASE_URL}Alexi_Kehayias_CV.pdf`,
  contact: {
    title: "Interested?",
    text: "Click the links below to get in touch or connect with me on LinkedIn!",
  },
  work: {
    title: "Explore My Work",
    text: "Navigate through my portfolio by selecting a specialization to view related projects",
  },
  toolbox: {
    title: "Toolbox",
    text: "The technologies I use in production.",
  },
};

/** Keys of projectData. */
export type CategoryId = "dataAnalytics" | "computerVision" | "machineLearning" | "nlp" | "research" | "mlops";

/** A Work-section tab. */
export interface Specialization {
  id: CategoryId;
  title: string;
  icon: LucideIcon;
  description: string;
}

/** The Work section's tabs, in display order. */
export const specializations: Specialization[] = [
  { id: "dataAnalytics", title: "Data Analytics", icon: BarChart3, description: "Interactive dashboards and business intelligence solutions" },
  { id: "computerVision", title: "Computer Vision", icon: Eye, description: "Image processing and visual recognition systems" },
  { id: "machineLearning", title: "Machine Learning", icon: Brain, description: "Predictive modeling and algorithmic solutions" },
  { id: "nlp", title: "Natural Language Processing", icon: MessageSquare, description: "Text analysis and language understanding" },
  { id: "research", title: "Research Methods", icon: Search, description: "Scientific research and statistical analysis" },
  { id: "mlops", title: "MLOps & Deployment", icon: Settings, description: "Production ML systems and infrastructure" },
];

/** A project together with where it lives in projectData. */
export interface ProjectRef {
  category: CategoryId;
  index: number;
  project: Project;
}

/** Every project in display order: the About stats and prev/next navigation on detail pages. */
export const allProjects: ProjectRef[] = specializations.flatMap(({ id }) =>
  projectData[id].projects.map((project, index) => ({ category: id, index, project }))
);

/** Router path of a project's detail page. */
export const projectPath = (category: string, index: number) => `/project/${category}/${index}`;
