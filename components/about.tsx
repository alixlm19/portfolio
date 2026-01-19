"use client";

import { Code2, Database, Brain, Workflow } from "lucide-react";
import { motion } from "framer-motion";

const iconMap = {
  Brain,
  Code2,
  Database,
  Workflow,
};

interface Skill {
  icon: string;
  title: string;
  description: string;
}

interface AboutProps {
  data: {
    about: {
      description: string;
      skills: Skill[];
    };
    techStack: string[];
  };
}

export function About({ data }: AboutProps) {

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
      },
    },
  };

  const cardVariants = {
    hidden: { opacity: 0, scale: 0.8 },
    visible: {
      opacity: 1,
      scale: 1,
      transition: {
        duration: 0.5,
      },
    },
  };

  return (
    <section id="about" className="py-20 px-4">
      <div className="max-w-7xl mx-auto">
        <motion.div
          className="text-center space-y-4 mb-16"
          initial={{ opacity: 0, y: -20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-3xl sm:text-4xl font-bold">About Me</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            {data.about.description}
          </p>
        </motion.div>

        <motion.div
          className="grid grid-cols-1 md:grid-cols-2 gap-6"
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
        >
          {data.about.skills.map((skill) => {
            const Icon = iconMap[skill.icon as keyof typeof iconMap];
            return (
              <motion.div
                key={skill.title}
                variants={cardVariants}
                whileHover={{
                  scale: 1.02,
                  boxShadow: "0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)",
                }}
                className="p-6 rounded-lg border bg-card transition-shadow cursor-pointer"
              >
                <div className="flex items-start gap-4">
                  <motion.div
                    className="p-3 rounded-lg bg-primary/10"
                    whileHover={{ scale: 1.05 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Icon className="h-6 w-6 text-primary" />
                  </motion.div>
                  <div className="space-y-2">
                    <h3 className="text-xl font-semibold">{skill.title}</h3>
                    <p className="text-muted-foreground">{skill.description}</p>
                  </div>
                </div>
              </motion.div>
            );
          })}
        </motion.div>

        <motion.div
          className="mt-12 p-8 rounded-lg border bg-card"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <h3 className="text-2xl font-semibold mb-4">Tech Stack</h3>
          <div className="flex flex-wrap gap-2">
            {data.techStack.map((tech, index) => (
              <motion.span
                key={tech}
                className="px-4 py-2 rounded-full bg-secondary text-secondary-foreground text-sm"
                initial={{ opacity: 0, scale: 0 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                whileHover={{ scale: 1.1, rotate: 5 }}
              >
                {tech}
              </motion.span>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
