import { Button } from "@/components/ui/button";
import { ArrowDown, Linkedin, Mail } from "lucide-react";
import profile_photo from "@/assets/profile_photo.jpg";

const cvFile = "/alexikehayiasportoflio/Alexi_Kehayias_CV.pdf";

const Hero = () => {
  const scrollToProjects = () => {
    document.getElementById("projects")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="min-h-screen flex items-center justify-center bg-gradient-hero relative overflow-hidden">
      {/* Removed decorative shapes & glowing lines */}

      <div className="container mx-auto px-4 relative z-10">
        <div className="text-center max-w-4xl mx-auto">
          {/* Profile photo */}
          <img
            src={profile_photo}
            alt="Alexi Kehayias"
            className="mx-auto mb-6 w-60 h-60 rounded-full shadow-lg border-4 border-primary/40 object-cover"
          />

          {/* Name in solid black (no gradient/glow) */}
          <h1 className="text-5xl md:text-7xl font-bold mb-6 font-heading text-black">
            Alexi Kehayias
          </h1>

          {/* Subheading in black */}
          <h2 className="text-2xl md:text-3xl font-semibold mb-6 font-mono text-black">
            Data Science & AI Student
          </h2>

          {/* Intro paragraph in black */}
          <p className="text-lg md:text-xl mb-8 max-w-2xl mx-auto leading-relaxed text-black">
            Hi there! I'm Alexi, a student in Breda studying Data Science and AI with a focus on a number of areas such as Computer Vision, Natural Language processing as well as MLOps. I enjoy building innovative solutions that make a difference to companies and society. Explore my projects below to see what I've been working on!
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

            <a href={cvFile} download rel="noopener noreferrer">
              <Button
                variant="secondary"
                size="lg"
                className="bg-secondary/20 backdrop-blur border-secondary hover:bg-secondary/30 text-secondary hover:shadow-cyan-glow"
              >
                Download CV
              </Button>
            </a>
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

    </section>
  );
};

export default Hero;
