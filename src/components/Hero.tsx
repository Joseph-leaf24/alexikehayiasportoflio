import { Button } from "@/components/ui/button";
import { ArrowDown, Github, Linkedin, Mail } from "lucide-react";

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
          <h1 className="text-5xl md:text-7xl font-bold mb-6 font-heading bg-gradient-accent bg-clip-text text-transparent tech-glow">
            Alexi Kehayias
          </h1>
          <h2 className="text-2xl md:text-3xl font-semibold mb-6 text-primary font-mono">
            Data Science & AI Specialist
          </h2>
          <p className="text-lg md:text-xl text-muted-foreground mb-8 max-w-2xl mx-auto leading-relaxed">
            Transforming data into actionable insights through advanced analytics, machine learning, 
            and cutting-edge AI solutions for business growth and innovation.
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
            <Button 
              variant="outline" 
              size="lg"
              className="border-secondary text-secondary hover:bg-secondary/10 hover:shadow-cyan-glow"
            >
              <Github className="mr-2 h-4 w-4" />
              GitHub
            </Button>
          </div>
          
          <div className="flex justify-center gap-6">
            <Button variant="ghost" size="sm" className="text-primary hover:text-primary-light hover:shadow-glow">
              <Mail className="h-5 w-5" />
            </Button>
            <Button variant="ghost" size="sm" className="text-secondary hover:text-secondary-light hover:shadow-cyan-glow">
              <Linkedin className="h-5 w-5" />
            </Button>
            <Button variant="ghost" size="sm" className="text-accent hover:text-accent hover:shadow-orange-glow">
              <Github className="h-5 w-5" />
            </Button>
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