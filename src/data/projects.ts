// Import project images
import dataAnalyticsImg from "@/assets/data-analytics.jpg";
import computerVisionImg from "@/assets/computer-vision.jpg";
import cv1_accuracyCurves from "@/assets/Accuracy-Curves-Occupied-Unoccupied-Classrooms.png";
import cv1_lossCurves from "@/assets/Loss--Curves-Occupied-Unoccupied-Classrooms.png";
import cv1_modelAccuracy from "@/assets/Model-Accuracy-Occupied-Unoccupied-Classrooms.png";
import cv1_modelXAI from "@/assets/XAI-Empty-Classroom.png";

import computerVisionImg_root from "@/assets/CV-Roots-Output-2.png";
import computerVisionImg_root_1 from "@/assets/CV-Roots-Output-1.png";
import computerVisionImg_root_2 from "@/assets/CV-Roots-Output-3.png";
import computerVisionImg_root_3 from "@/assets/Robotics-Innoculation.gif";

import machineLearningImg from "@/assets/machine-learning.jpg";
import machineLearningImg_1 from  "@/assets/ML-ANWB-Application.png";
import machineLearningImg_2 from "@/assets/ML-ANWB-Model-Conf-Matrix.png";
import machineLearningImg_3 from "@/assets/ML-ANWB-Model-Learning-Curves.png";

    
import nlpImg from "@/assets/nlp.jpg";
import nlpImg_1 from "@/assets/NLP_Main_Page.png";
import nlpImg_2 from "@/assets/NPL_Results_Page.png";

import researchImg from "@/assets/research.jpg";
import researchImg_1 from "@/assets/Chatbots-Scientific-Research.png";

import mlopsImg from "@/assets/mlops.jpg";
import mlopsImg_1 from "@/assets/Azure Architecutre Diagram CV1.png";
import mlopsImg_2 from "@/assets/CLI Training Working.png";
import mlopsImg_3 from "@/assets/MLOps-Learning-Curves.png";


import { ProjectData } from "@/types/project";

export const projectData: ProjectData = {
  dataAnalytics: {
    title: "Data Analytics",
    description: "Interactive dashboards and business intelligence solutions using Microsoft Power BI for public health and security analytics.",
    projects: [
      {
        title: "Tuberculosis Analysis in Zimbabwe",
        description: "Designed an interactive Power BI dashboard to support public health decision-making by analyzing TB trends in Zimbabwe. Features regional benchmarking against Southern Africa and tracks progress toward SDG 3 goals.",
        image: dataAnalyticsImg,
        tags: ["Power BI", "Public Health", "SDG Tracking"],
        concepts: ["Exploratory Data Analysis", "Data Visualization", "Time Series Analysis", "Regional Benchmarking"],
        tools: ["Microsoft Power BI", "Excel", "WHO Data APIs"],
        
        githubUrl: "https://github.com/Joseph-leaf24/alexikehayiasportoflio/blob/90f7623b18decddaeca79c565abfa21c26d674bd/Projects/Microsoft%20Power%20BI/SDG%20Indicators-Zimbabwe%20Tuberculosis/SDGIndicatorsDashboard_AlexiKehayias.pbix", 
        powerBiUrl: "https://app.powerbi.com/view?r=eyJrIjoiZTQyMzU2ZTEtNzJjZi00MzRmLWE0M2UtYjBlYmIzM2MxYzExIiwidCI6IjBhMzM1ODliLTAwMzYtNGZlOC1hODI5LTNlZDA5MjZhZjg4NiIsImMiOjl9"
      },
      {
        title: "IKEA Card Skimming Detection Dashboard",
        description: "Comprehensive Power BI dashboard for monitoring payment terminal redeployment at IKEA Breda. Enables tracking of terminal movements, employee behavior analysis, and identification of security risk hotspots.",
        image: dataAnalyticsImg,
        tags: ["Power BI", "Security Analytics", "Retail"],
        concepts: ["Security Risk Analytics", "Operational Intelligence", "Employee Behavior Analysis", "Compliance Monitoring"],
        tools: ["Microsoft Power BI", "SQL Server", "Excel"],
        githubUrl: "https://github.com/Joseph-leaf24/alexikehayiasportoflio/blob/90f7623b18decddaeca79c565abfa21c26d674bd/Projects/Microsoft%20Power%20BI/IKEA%20Card%20Skimming%20Detection/IKEA%20Terminal%20Dashboard-Alexi%20Kehayias.pbix",
        powerBiUrl: "https://app.powerbi.com/view?r=eyJrIjoiMTMxMDM2MGItYzI3ZC00MzI3LTk2YjktMzg0ODE2YzhkM2E5IiwidCI6IjBhMzM1ODliLTAwMzYtNGZlOC1hODI5LTNlZDA5MjZhZjg4NiIsImMiOjl9"
      }
    ]
  },
  
  computerVision: {
    title: "Computer Vision",
    description: "Advanced image processing and AI solutions for applications such occupancy detection and agricultural phenotyping.",
    projects: [
      {
        title: "Classroom Occupancy Detection with Explainable AI",
        description: "CNN-based image classification system to detect classroom occupancy with Grad-CAM explainability. Designed for university space optimization with privacy-first approach and real-time analytics.",
        image: computerVisionImg,
        images: [ // These show in the gallery inside the project
          cv1_accuracyCurves,
          cv1_lossCurves,
          cv1_modelAccuracy,
          cv1_modelXAI
        ],
        tags: ["CNN", "Explainable AI", "Real-time Processing"],
        concepts: ["Convolutional Neural Networks", "Grad-CAM Explainability", "Image Classification", "Responsible AI"],
        tools: ["Python", "TensorFlow/Keras", "OpenCV", "Matplotlib"],
        githubUrl: "https://github.com/Joseph-leaf24/alexikehayiasportoflio/tree/90f7623b18decddaeca79c565abfa21c26d674bd/Projects/Computer%20Vision-%20Deep%20Learning/Occupied%20And%20Unoccupied%20Classroom%20Detection" 
      },
      {
        title: "Primary Root Detection Pipeline (NPEC)",
        description: "Full AI pipeline for automated detection, segmentation, and measurement of plant root structures. Integrates with robotics for automated lab tasks and supports agricultural phenotyping research.",
        image: computerVisionImg_root,
        images: [ // Additional gallery images 
          computerVisionImg_root_1,
          computerVisionImg_root_2,
          computerVisionImg_root_3
        ],
        tags: ["Image Segmentation", "Agricultural AI", "Robotics Integration"],
        concepts: ["Image Segmentation", "Skeletonization", "Feature Extraction", "Transfer Learning"],
        tools: ["Python", "TensorFlow/Keras", "OpenCV", "Stable-Baselines3"],
        githubUrl: "https://github.com/Joseph-leaf24/alexikehayiasportoflio/blob/90f7623b18decddaeca79c565abfa21c26d674bd/Projects/Computer%20Vision-%20Deep%20Learning/Root%20Segmentation%20Analysis%20%26%20Robotics-NPEC/Computer%20Vision%20%26%20Robotics/Codebase%20RL%20Pipeline%20Documentation%20232230.md" 
      }
    ]
  },
  
  machineLearning: {
    title: "Machine Learning",
    description: "End-to-end ML solutions following CRISP-DM methodology for real-world prediction and classification tasks.",
    projects: [
      {
        title: "Accident Detection & Risk Analysis",
        description: "Full-stack driving risk prediction system using CRISP-DM methodology. Deep learning classifier for accident risk levels with SQL-based data pipeline and interactive Streamlit deployment.",
        image: machineLearningImg,
        images: [ // Additional gallery images 
          machineLearningImg_1,
          machineLearningImg_2,
          machineLearningImg_3

        ],
        tags: ["Deep Learning", "CRISP-DM", "Risk Prediction"],
        concepts: ["CRISP-DM Lifecycle", "Deep Learning Classification", "Feature Engineering", "Real-time Prediction"],
        tools: ["Python", "TensorFlow/Keras", "SQL", "Streamlit"],
        githubUrl: "https://github.com/Joseph-leaf24/alexikehayiasportoflio/tree/90f7623b18decddaeca79c565abfa21c26d674bd/Projects/Exploratory%20Data%20Analysis%20And%20Machine%20Learning/ANWB%20Accident%20Risk%20Detection" 
      }
    ]
  },
  
  nlp: {
    title: "Natural Language Processing",
    description: "Advanced text processing and multilingual AI solutions for emotion detection and content analysis.",
    projects: [
      {
        title: "Greek Emotion Detection with Translation",
        description: "End-to-end NLP pipeline for emotion detection in Greek YouTube videos. Combines Whisper speech recognition, multilingual translation, and transformer-based emotion classification with intensity scoring.",
        image: nlpImg,
        images: [ // Additional gallery images 
          nlpImg_1,
          nlpImg_2,

        ],
        tags: ["Multilingual NLP", "Speech Recognition", "Emotion AI"],
        concepts: ["Speech-to-Text", "Machine Translation", "Emotion Classification", "Pipeline Engineering"],
        tools: ["Python", "HuggingFace Transformers", "Whisper", "OpenAI API"],
        githubUrl: "https://github.com/Joseph-leaf24/alexikehayiasportoflio/blob/90f7623b18decddaeca79c565abfa21c26d674bd/Projects/Natural%20Language%20Processing/Emotion%20Detection%20%26%20Translation/README_Task11.md" 
      }
    ]
  },
  
  research: {
    title: "Scientific Research",
    description: "Mixed-methods research combining quantitative analysis and qualitative insights for business and technology studies.",
    projects: [
      {
        title: "Impact of Chatbots on SMEs",
        description: "Comprehensive mixed-methods study exploring chatbot effectiveness in Small and Medium Enterprises. Combined statistical analysis with thematic research to provide actionable business recommendations.",
        image: researchImg,
        images: [ // Additional gallery images - replace with your actual filenames
          researchImg_1

        ],
        tags: ["Mixed Methods", "Business Research", "Statistical Analysis"],
        concepts: ["Mixed-Methods Research", "Statistical Testing", "Thematic Analysis", "Stakeholder Mapping"],
        tools: ["Python", "SPSS", "Qualtrics", "Excel"],
        githubUrl: "https://github.com/Joseph-leaf24/alexikehayiasportoflio/blob/90f7623b18decddaeca79c565abfa21c26d674bd/Projects/Scientific%20Research/Chatbots%20Impact%20On%20SME's/Chatbots_3__Final_Policy_Paper.pdf" 
      }
    ]
  },
  
  mlops: {
    title: "MLOps & Deployment",
    description: "Production-ready ML systems with containerization, cloud deployment, and automated workflow orchestration.",
    projects: [
      {
        title: "ROALT: Root Analysis Toolkit Deployment",
        description: "End-to-end MLOps deployment of deep learning pipelines using Docker, Azure ML, and Airflow. Features FastAPI backend, Gradio UI, and full CI/CD integration for plant research applications.",
        image: mlopsImg,
        images: [ // Additional gallery images - replace with your actual filenames
            mlopsImg_1,
            mlopsImg_2,
            mlopsImg_3
        ],
        tags: ["Docker", "Azure ML", "CI/CD"],
        concepts: ["MLOps Lifecycle", "Container Orchestration", "Cloud Deployment", "Workflow Automation"],
        tools: ["Docker", "Azure ML Studio", "Apache Airflow", "FastAPI", "GitHub Actions"],
        githubUrl: "https://github.com/Joseph-leaf24/alexikehayiasportoflio/blob/90f7623b18decddaeca79c565abfa21c26d674bd/Projects/MLOps%20Deployment/README.md" 
      }
    ]
  }
};