import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  BarChart3, 
  Eye, 
  Brain, 
  MessageSquare, 
  Search, 
  Settings 
} from "lucide-react";

const Skills = () => {
  const skillAreas = [
    {
      icon: BarChart3,
      title: "Data Analytics",
      description: "Interactive dashboards and business intelligence solutions",
      skills: ["Power BI", "Tableau", "SQL", "Excel", "Data Visualization"]
    },
    {
      icon: Eye,
      title: "Computer Vision",
      description: "Image processing and visual recognition systems",
      skills: ["OpenCV", "TensorFlow", "CNNs", "Image Segmentation", "Object Detection"]
    },
    {
      icon: Brain,
      title: "Machine Learning",
      description: "Predictive modeling and algorithmic solutions",
      skills: ["Scikit-learn", "Deep Learning", "CRISP-DM", "Model Deployment", "Feature Engineering"]
    },
    {
      icon: MessageSquare,
      title: "Natural Language Processing",
      description: "Text analysis and language understanding",
      skills: ["Transformers", "HuggingFace", "Sentiment Analysis", "Multilingual NLP", "Text Classification"]
    },
    {
      icon: Search,
      title: "Research Methods",
      description: "Scientific research and statistical analysis",
      skills: ["Statistical Testing", "Mixed Methods", "Survey Design", "Thematic Analysis", "Hypothesis Testing"]
    },
    {
      icon: Settings,
      title: "MLOps & Deployment",
      description: "Production ML systems and infrastructure",
      skills: ["Docker", "Azure", "Airflow", "CI/CD", "Model Monitoring", "FastAPI"]
    }
  ];

  return (
    <section className="py-16 bg-background">
      <div className="container mx-auto px-4">
        <div className="max-w-3xl mx-auto text-center mb-12">
          <h2 className="text-4xl font-bold mb-4 text-foreground">
            Core Specializations
          </h2>
          <p className="text-lg text-muted-foreground leading-relaxed">
            I specialize in end-to-end data science solutions, from initial analysis 
            through to production deployment and monitoring.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {skillAreas.map((area, index) => {
            const IconComponent = area.icon;
            return (
              <Card key={index} className="p-6 bg-card shadow-soft hover:shadow-glow transition-all duration-300 group border-border hover:border-primary/30">
                <div className="flex items-center mb-4">
                  <div className="p-3 rounded-lg bg-gradient-primary mr-4 group-hover:scale-110 transition-transform duration-300 group-hover:shadow-glow">
                    <IconComponent className="h-6 w-6 text-background" />
                  </div>
                  <h3 className="text-lg font-semibold text-foreground font-heading">
                    {area.title}
                  </h3>
                </div>
                
                <p className="text-muted-foreground mb-4 text-sm leading-relaxed">
                  {area.description}
                </p>
                
                <div className="flex flex-wrap gap-2">
                  {area.skills.map((skill, skillIndex) => (
                    <Badge key={skillIndex} variant="secondary" className="text-xs">
                      {skill}
                    </Badge>
                  ))}
                </div>
              </Card>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default Skills;