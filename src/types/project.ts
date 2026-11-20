/** One portfolio project, as listed in src/data/projects.ts. */
export interface Project {
  title: string;
  description: string;
  /** Cover screenshot on the project card */
  image: string;
  /** Screenshots for the gallery on the project page */
  images?: string[];
  /** Key technologies ("Technologies Used"); the middle layer of the project network */
  tags: string[];
  /** AI / data-science concepts; the output layer of the project network */
  concepts: string[];
  /** Secondary facts and results, shown in a collapsed "Highlights" drop-down */
  highlights?: string[];
  /** Tools & languages; the input layer of the project network (tags stand in when absent) */
  tools?: string[];
  /** Live demo, linked from the project page */
  demoUrl?: string;
  /** Public Power BI report, embedded on the project page */
  powerBiUrl?: string;
}

/** A specialization (Work tab) and its projects. */
export interface ProjectCategory {
  title: string;
  description: string;
  projects: Project[];
}

/** Every specialization's projects, keyed by category id (see CategoryId). */
export interface ProjectData {
  [key: string]: ProjectCategory;
}