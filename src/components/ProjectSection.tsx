import ProjectCard from "./ProjectCard";
import { Project } from "@/types/project";

interface ProjectSectionProps {
  title: string;
  description: string;
  projects: Project[];
  id: string;
}

const ProjectSection = ({ title, description, projects, id }: ProjectSectionProps) => {
  return (
    <section id={id} className="py-16 bg-gradient-surface">
      <div className="container mx-auto px-4">
        <div className="max-w-3xl mx-auto text-center mb-12">
          <h2 className="text-4xl font-bold mb-4 bg-gradient-accent bg-clip-text text-transparent font-heading">
            {title}
          </h2>
          <p className="text-lg text-muted-foreground leading-relaxed">
            {description}
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {projects.map((project, index) => (
            <ProjectCard 
              key={index} 
              {...project} 
              category={id}
              projectIndex={index}
            />
          ))}
        </div>
      </div>
    </section>
  );
};

export default ProjectSection;