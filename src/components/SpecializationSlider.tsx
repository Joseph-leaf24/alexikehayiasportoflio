import { useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { 
  BarChart3, 
  Eye, 
  Brain, 
  MessageSquare, 
  Search, 
  Settings,
  ChevronLeft,
  ChevronRight
} from "lucide-react";
import { cn } from "@/lib/utils";

interface Specialization {
  id: string;
  title: string;
  icon: any;
  description: string;
}

interface SpecializationSliderProps {
  activeSpecialization: number;
  setActiveSpecialization: (index: number) => void;
}

const SpecializationSlider = ({ activeSpecialization, setActiveSpecialization }: SpecializationSliderProps) => {
  const sliderRef = useRef<HTMLDivElement>(null);

  const specializations: Specialization[] = [
    {
      id: "dataAnalytics",
      title: "Data Analytics",
      icon: BarChart3,
      description: "Interactive dashboards and business intelligence solutions"
    },
    {
      id: "computerVision", 
      title: "Computer Vision",
      icon: Eye,
      description: "Image processing and visual recognition systems"
    },
    {
      id: "machineLearning",
      title: "Machine Learning", 
      icon: Brain,
      description: "Predictive modeling and algorithmic solutions"
    },
    {
      id: "nlp",
      title: "Natural Language Processing",
      icon: MessageSquare,
      description: "Text analysis and language understanding"
    },
    {
      id: "research",
      title: "Research Methods",
      icon: Search,
      description: "Scientific research and statistical analysis"
    },
    {
      id: "mlops",
      title: "MLOps & Deployment",
      icon: Settings,
      description: "Production ML systems and infrastructure"
    }
  ];

  const scrollToSection = (sectionId: string) => {
    document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth' });
  };

  const scrollToActiveCard = (index: number) => {
    if (sliderRef.current) {
      const cardWidth = 300; // 280px + 20px gap
      const containerWidth = sliderRef.current.clientWidth;
      const scrollPosition = Math.max(0, (index * cardWidth) - (containerWidth / 2) + (cardWidth / 2));
      
      sliderRef.current.scrollTo({
        left: scrollPosition,
        behavior: 'smooth'
      });
    }
  };

  const handleSpecializationClick = (index: number) => {
    setActiveSpecialization(index);
    scrollToSection(specializations[index].id);
    scrollToActiveCard(index);
  };

  const canScrollLeft = () => {
    return sliderRef.current ? sliderRef.current.scrollLeft > 0 : false;
  };

  const canScrollRight = () => {
    if (!sliderRef.current) return false;
    return sliderRef.current.scrollLeft < 
           sliderRef.current.scrollWidth - sliderRef.current.clientWidth - 10;
  };

  const scrollLeft = () => {
    if (sliderRef.current) {
      sliderRef.current.scrollBy({ left: -300, behavior: 'smooth' });
    }
  };

  const scrollRight = () => {
    if (sliderRef.current) {
      sliderRef.current.scrollBy({ left: 300, behavior: 'smooth' });
    }
  };

  useEffect(() => {
    scrollToActiveCard(activeSpecialization);
  }, [activeSpecialization]);

  return (
    <section className="py-20 bg-gradient-surface">
      <div className="container mx-auto px-4">
        <div className="max-w-6xl mx-auto text-center mb-12">
          <h2 className="text-4xl font-bold mb-6 bg-gradient-accent bg-clip-text text-transparent font-heading">
            Explore My Work
          </h2>
          <p className="text-lg text-muted-foreground mb-8 max-w-2xl mx-auto">
            Navigate through my portfolio by selecting a specialization to view related projects
          </p>
        </div>

        {/* Dynamic Slider with Navigation */}
        <div className="max-w-6xl mx-auto relative">
          {/* Left Navigation Button */}
          <Button
            variant="outline"
            size="sm"
            className="absolute left-0 top-1/2 -translate-y-1/2 z-10 bg-background/80 backdrop-blur border-border hover:bg-card shadow-lg"
            onClick={scrollLeft}
            disabled={!canScrollLeft()}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>

          {/* Right Navigation Button */}
          <Button
            variant="outline"
            size="sm"
            className="absolute right-0 top-1/2 -translate-y-1/2 z-10 bg-background/80 backdrop-blur border-border hover:bg-card shadow-lg"
            onClick={scrollRight}
            disabled={!canScrollRight()}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>

          {/* Scrollable Specializations Container */}
          <div 
            ref={sliderRef}
            className="flex overflow-x-auto gap-5 pb-4 mb-8 mx-12 scroll-smooth [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]"
          >
            {specializations.map((spec, index) => {
              const IconComponent = spec.icon;
              return (
                <Button
                  key={spec.id}
                  variant={activeSpecialization === index ? "default" : "outline"}
                  className={cn(
                    "flex-shrink-0 h-auto p-6 min-w-[280px] flex flex-col items-center text-center transition-all duration-500 ease-out",
                    activeSpecialization === index
                      ? "bg-primary text-primary-foreground shadow-glow scale-105 animate-scale-in"
                      : "bg-card/50 backdrop-blur border-border hover:border-primary/30 hover:shadow-soft hover:scale-102"
                  )}
                  onClick={() => handleSpecializationClick(index)}
                >
                  <div className={cn(
                    "p-3 rounded-lg mb-3 transition-all duration-300",
                    activeSpecialization === index
                      ? "bg-primary-foreground/20"
                      : "bg-gradient-primary"
                  )}>
                    <IconComponent className={cn(
                      "h-6 w-6 transition-transform duration-300",
                      activeSpecialization === index
                        ? "text-primary-foreground animate-pulse"
                        : "text-background"
                    )} />
                  </div>
                  <h3 className="text-sm font-semibold mb-2 font-heading">
                    {spec.title}
                  </h3>
                  <p className={cn(
                    "text-xs leading-relaxed",
                    activeSpecialization === index
                      ? "text-primary-foreground/80"
                      : "text-muted-foreground"
                  )}>
                    {spec.description}
                  </p>
                </Button>
              );
            })}
          </div>

          {/* Enhanced Navigation Dots */}
          <div className="flex justify-center gap-3">
            {specializations.map((_, index) => (
              <button
                key={index}
                className={cn(
                  "w-2 h-2 rounded-full transition-all duration-300 hover:scale-125",
                  activeSpecialization === index
                    ? "bg-primary w-8 h-2 animate-pulse"
                    : "bg-muted-foreground/30 hover:bg-muted-foreground/50"
                )}
                onClick={() => handleSpecializationClick(index)}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default SpecializationSlider;