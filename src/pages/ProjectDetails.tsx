import { useParams, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, Github, ExternalLink } from "lucide-react";
import { projectData } from "@/data/projects";
import { Project } from "@/types/project";
import React, { useState } from "react";
const ProjectDetails = () => {
  const { category, projectIndex } = useParams();
  const navigate = useNavigate();
  const [modalOpen, setModalOpen] = useState(false);
  const [modalImage, setModalImage] = useState<string | null>(null);
  
  // Find the project based on category and index
  const categoryData = category ? projectData[category as keyof typeof projectData] : null;
  const project: Project | null = categoryData && projectIndex ? categoryData.projects[parseInt(projectIndex)] : null;
  
  if (!project || !categoryData) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Project not found</h1>
          <Button onClick={() => navigate('/')}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Home
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-surface">
            {/* Fullscreen Image Modal */}
      {modalOpen && modalImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80"
          onClick={() => setModalOpen(false)}
        >
          <button
            className="absolute top-6 right-8 text-white text-3xl z-60 bg-black/50 rounded-full p-2 hover:bg-black/80 transition"
            onClick={e => {
              e.stopPropagation();
              setModalOpen(false);
            }}
            aria-label="Close"
          >
            
          </button>
          <img
            src={modalImage}
            alt="Fullscreen"
            className="max-h-[90vh] max-w-[90vw] rounded-lg shadow-2xl"
            onClick={e => e.stopPropagation()}
          />
        </div>
      )}
      <div className="container mx-auto px-4 py-8">
        <Button 
          variant="outline" 
          onClick={() => navigate('/')}
          className="mb-8"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Portfolio
        </Button>
        
        <div className="w-full mx-auto">
          {/* Project Header */}
          <div className="text-center mb-12">
            <div className="inline-block px-4 py-2 bg-primary/10 text-primary rounded-full text-sm font-medium mb-4">
              {categoryData.title}
            </div>
            <h1 className="text-4xl md:text-5xl font-bold mb-6 bg-gradient-accent bg-clip-text text-transparent font-heading">
              {project.title}
            </h1>
            <p className="text-xl text-muted-foreground leading-relaxed max-w-2xl mx-auto">
              {project.description}
            </p>
          </div>

          {/* Project Image or Power BI Dashboard */}
          {project.powerBiUrl ? (
            <div className="mb-12">
              <h3 className="text-2xl font-semibold mb-6 text-center text-foreground">Interactive Dashboard</h3>
              <div className="relative rounded-lg overflow-hidden shadow-glow bg-card w-full">
                <iframe 
                  title={project.title}
                  width="100%" 
                  height="600"
                  src={project.powerBiUrl}
                  frameBorder="0" 
                  allowFullScreen={true}
                  className="w-full block"
                  style={{ aspectRatio: '16/10' }}
                />
              </div>
            </div>
          ) : (
            <div className="mb-12">
            {/* Additional images gallery */}
            {project.images && project.images.length > 0 && (
              <div>
                <h3 className="text-2xl font-semibold mb-6 text-center text-foreground">Project Gallery</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {project.images.map((imageUrl, index) => (
                    <div
                      key={index}
                      className="relative rounded-lg overflow-hidden shadow-lg hover:shadow-xl transition-shadow flex justify-center items-center cursor-pointer"
                      onClick={() => {
                        setModalImage(imageUrl);
                        setModalOpen(true);
                      }}
                    >
                      <img
                        src={imageUrl}
                        alt={`${project.title} - Image ${index + 1}`}
                        className="block max-w-full h-auto transition-transform duration-300 hover:scale-105"
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          )}

          {/* Project Details Grid */}
          <div className="grid md:grid-cols-2 gap-12 mb-12">
            {/* Technologies & Tools */}
            <div className="space-y-8">
              <div>
                <h3 className="text-xl font-semibold mb-4 text-foreground">Technologies Used</h3>
                <div className="flex flex-wrap gap-2">
                  {project.tags.map((tag, index) => (
                    <Badge key={index} variant="secondary" className="text-sm px-3 py-1">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </div>
              
              {project.tools && project.tools.length > 0 && (
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-foreground">Tools & Languages</h3>
                  <div className="flex flex-wrap gap-2">
                    {project.tools.map((tool, index) => (
                      <Badge key={index} variant="outline" className="text-sm px-3 py-1">
                        {tool}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* AI/Data Science Concepts */}
            <div>
              <h3 className="text-xl font-semibold mb-4 text-foreground">AI/Data Science Concepts</h3>
              <div className="flex flex-wrap gap-2">
                {project.concepts.map((concept, index) => (
                  <Badge key={index} className="text-sm px-3 py-1 bg-gradient-primary text-white">
                    {concept}
                  </Badge>
                ))}
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          {project.githubUrl && (
            <div className="flex justify-center">
              <Button 
                variant="outline" 
                size="lg" 
                className="min-w-40"
                onClick={() => window.open(project.githubUrl, '_blank')}
              >
                <Github className="mr-2 h-5 w-5" />
                View Source Code
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ProjectDetails;