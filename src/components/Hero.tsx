import { Button } from "@/components/ui/button";
import { ArrowDown, Linkedin, Mail } from "lucide-react";
import profile_photo from "@/assets/profile_photo.jpg";
const Hero = () => {
  const scrollToProjects = () => {
    document.getElementById('projects')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section className="min-h-screen flex items-center justify-center bg-gradient-hero relative overflow-hidden">
      {/* Circuit board decorative elements */}
      <div className="absolute inset-0 bg-circuit-pattern opacity-30" />
      <div className="absolute top-10 left-10 w-32 h-32 border border-primary/30 rounded tech-glow" />
      <div className="absolute bottom-20 right-20 w-24 h-24 border border-secondary/30 rounded-full tech-glow" />
      <div className="absolute top-1/3 right-10 w-16 h-16 border border-accent/30 transform rotate-45 tech-glow" />
      
      {/* Glowing lines */}
      <div className="absolute top-0 left-1/4 w-px h-full bg-gradient-to-b from-transparent via-primary/30 to-transparent" />
      <div className="absolute top-0 right-1/3 w-px h-full bg-gradient-to-b from-transparent via-secondary/30 to-transparent" />
      
      <div className="container mx-auto px-4 relative z-10">
        <div className="text-center max-w-4xl mx-auto">

            {/* Profile photo */}
            <img
            src={profile_photo}
            alt="Alexi Kehayias"
            className="mx-auto mb-6 w-60 h-60 rounded-full shadow-lg border-4 border-primary/40 object-cover tech-glow"
          />
          <h1 className="text-5xl md:text-7xl font-bold mb-6 font-heading bg-gradient-accent bg-clip-text text-transparent tech-glow">
            Alexi Kehayias
          </h1>
          <h2 className="text-2xl md:text-3xl font-semibold mb-6 text-primary font-mono">
            Data Science & AI Student
          </h2>
          <p className="text-lg md:text-xl text-muted-foreground mb-8 max-w-2xl mx-auto leading-relaxed">
            Hi there! I'm Alexi, a student from Zimbabwe studying Data Science and AI with a focus on a number of areas such as Computer Vision, Natural Language processing as well as MLOps. I enjoy building innovative solutions that make a difference to companies and society. Explore my projects below to see what I've been working on!
          </p>
          
          <div className="flex flex-wrap gap-4 justify-center mb-12">
            <Button 
              variant="secondary" 
              size="lg"
              onClick={scrollToProjects}
              className="bg-primary/20 backdrop-blur border-primary hover:bg-primary/30 text-primary hover:shadow-glow"
            >
              View Projects
              <ArrowDown className="ml-2 h-4 w-4" />
            </Button>
          </div>
          
          <div className="flex justify-center gap-6">
            <a href="mailto:kehayiasjalexi@gmail.com" target="_blank" rel="noopener noreferrer">
              <Button variant="ghost" size="sm" className="text-primary hover:text-primary-light hover:shadow-glow">
                <Mail className="h-5 w-5" />
              </Button>
            </a>
            <a href="https://www.linkedin.com/in/alexi-kehayias/" target="_blank" rel="noopener noreferrer">
              <Button variant="ghost" size="sm" className="text-secondary hover:text-secondary-light hover:shadow-cyan-glow">
                <Linkedin className="h-5 w-5" />
              </Button>
            </a>
          </div>
        </div>
      </div>
      
      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
        <ArrowDown className="h-6 w-6 text-primary tech-glow" />
      </div>
    </section>
  );
};

export default Hero;