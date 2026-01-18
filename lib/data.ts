import fs from "fs";
import path from "path";
import yaml from "js-yaml";

const dataDir = path.join(process.cwd(), "data");

export function loadYamlFile<T>(filename: string): T {
  const filePath = path.join(dataDir, filename);
  const fileContents = fs.readFileSync(filePath, "utf8");
  return yaml.load(fileContents) as T;
}

export interface PersonalInfo {
  name: string;
  fullName: string;
  title: string;
  tagline: string;
  location: string;
  phone: string;
  email: string;
  social: {
    github: string;
    linkedin: string;
    website: string;
  };
  about: {
    description: string;
    skills: Array<{
      icon: string;
      title: string;
      description: string;
    }>;
  };
  techStack: string[];
}

export interface Experience {
  experience: Array<{
    company: string;
    role: string;
    period: string;
    location: string;
    description: string;
    achievements: string[];
  }>;
}

export interface Education {
  education: Array<{
    school: string;
    degree: string;
    period: string;
    location: string;
    concentration?: string;
    minor?: string;
    coursework: string[];
  }>;
}

export interface Projects {
  projects: Array<{
    title: string;
    description: string;
    tags: string[];
    github: string;
    demo?: string;
  }>;
}

export function getPersonalInfo(): PersonalInfo {
  return loadYamlFile<PersonalInfo>("personal.yaml");
}

export function getExperience(): Experience {
  return loadYamlFile<Experience>("experience.yaml");
}

export function getEducation(): Education {
  return loadYamlFile<Education>("education.yaml");
}

export function getProjects(): Projects {
  return loadYamlFile<Projects>("projects.yaml");
}
