import { useState } from "react";
import Hero from "@/components/Hero";
import Skills from "@/components/Skills";
import SpecializationSlider from "@/components/SpecializationSlider";
import ProjectSection from "@/components/ProjectSection";
import { projectData } from "@/data/projects";

const Index = () => {
  const [activeSpecialization, setActiveSpecialization] = useState(0);
  return (
    <div className="min-h-screen bg-background">{/* Force refresh */}
      {/* Hero Section */}
      <Hero />
      
      {/* Skills Overview */}
      <Skills />
      
      <SpecializationSlider 
        activeSpecialization={activeSpecialization}
        setActiveSpecialization={setActiveSpecialization}
      />
      
      {/* Projects - Only show sections for the active specialization */}
      <div id="projects">
        {(() => {
          const specializationMapping: Record<number, string[]> = {
            0: ["dataAnalytics"], // Data Analytics
            1: ["computerVision"], // Computer Vision
            2: ["machineLearning"], // Machine Learning
            3: ["nlp"], // Natural Language Processing
            4: ["research"], // Research Methods
            5: ["mlops"] // MLOps & Deployment
          };

          const sectionsToShow = specializationMapping[activeSpecialization] || [];

          return sectionsToShow.map((sectionId) => {
            const sectionData = {
              dataAnalytics: {
                title: projectData.dataAnalytics.title,
                description: projectData.dataAnalytics.description,
                projects: projectData.dataAnalytics.projects
              },
              computerVision: {
                title: projectData.computerVision.title,
                description: projectData.computerVision.description,
                projects: projectData.computerVision.projects
              },
              machineLearning: {
                title: projectData.machineLearning.title,
                description: projectData.machineLearning.description,
                projects: projectData.machineLearning.projects
              },
              nlp: {
                title: projectData.nlp.title,
                description: projectData.nlp.description,
                projects: projectData.nlp.projects
              },
              research: {
                title: projectData.research.title,
                description: projectData.research.description,
                projects: projectData.research.projects
              },
              mlops: {
                title: projectData.mlops.title,
                description: projectData.mlops.description,
                projects: projectData.mlops.projects
              }
            };

            const section = sectionData[sectionId];
            if (!section) return null;

            return (
              <ProjectSection
                key={sectionId}
                id={sectionId}
                title={section.title}
                description={section.description}
                projects={section.projects}
              />
            );
          });
        })()}
      </div>
      
      {/* Footer */}
      <footer className="bg-background border-t border-primary/30 text-foreground py-12">
        <div className="container mx-auto px-4 text-center">
          <h3 className="text-2xl font-bold mb-4 text-primary font-heading">Ready to Collaborate?</h3>
          <p className="text-muted-foreground mb-6 max-w-2xl mx-auto">
            I'm always interested in discussing data science opportunities, 
            research collaborations, and innovative AI projects.
          </p>
          <div className="flex justify-center gap-4">
            <a href="mailto:your.email@example.com" className="text-primary hover:text-primary-light transition-colors hover:shadow-glow">
              Contact Me
            </a>
            <span className="text-muted-foreground">•</span>
            <a href="#" className="text-secondary hover:text-secondary-light transition-colors hover:shadow-cyan-glow">
              LinkedIn
            </a>
            <span className="text-muted-foreground">•</span>
            <a href="#" className="text-accent hover:text-accent transition-colors hover:shadow-orange-glow">
              GitHub
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
