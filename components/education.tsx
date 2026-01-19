"use client";

import { GraduationCap } from "lucide-react";
import { motion } from "framer-motion";

interface EducationItem {
  school: string;
  degree: string;
  concentration?: string;
  minor?: string;
  period: string;
  location: string;
  coursework: string[];
}

interface EducationProps {
  education: EducationItem[];
}

export function Education({ education }: EducationProps) {

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  const cardVariants = {
    hidden: { opacity: 0, scale: 0.9 },
    visible: {
      opacity: 1,
      scale: 1,
      transition: {
        duration: 0.5,
      },
    },
  };

  return (
    <section id="education" className="py-20 px-4 bg-muted/30">
      <div className="max-w-7xl mx-auto">
        <motion.div
          className="text-center space-y-4 mb-16"
          initial={{ opacity: 0, y: -20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <h2 className="text-3xl sm:text-4xl font-bold">Education</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Academic foundation in Computer Science, Statistics, and Machine Learning.
          </p>
        </motion.div>

        <motion.div
          className="grid grid-cols-1 md:grid-cols-2 gap-6"
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
        >
          {education.map((edu, index) => (
            <motion.div
              key={index}
              variants={cardVariants}
              whileHover={{
                scale: 1.02,
                boxShadow: "0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)",
              }}
              className="p-6 rounded-lg border bg-card transition-shadow"
            >
              <div className="flex items-start gap-4">
                <motion.div
                  className="p-3 rounded-lg bg-primary/10"
                  whileHover={{ scale: 1.05 }}
                  transition={{ duration: 0.6 }}
                >
                  <GraduationCap className="h-6 w-6 text-primary" />
                </motion.div>
                <div className="flex-1">
                  <h3 className="text-xl font-semibold mb-1">{edu.school}</h3>
                  <p className="font-medium text-primary mb-1">{edu.degree}</p>
                  {(edu.concentration || edu.minor) && (
                    <p className="text-sm text-muted-foreground mb-2">
                      {edu.concentration || edu.minor}
                    </p>
                  )}
                  <div className="flex items-center justify-between text-sm text-muted-foreground mb-4">
                    <span>{edu.period}</span>
                    <span>{edu.location}</span>
                  </div>
                  <div>
                    <p className="text-sm font-medium mb-2">Relevant Coursework:</p>
                    <div className="flex flex-wrap gap-2">
                      {edu.coursework.map((course, idx) => (
                        <motion.span
                          key={course}
                          className="px-2 py-1 rounded-md bg-secondary text-secondary-foreground text-xs"
                          initial={{ opacity: 0, scale: 0 }}
                          whileInView={{ opacity: 1, scale: 1 }}
                          viewport={{ once: true }}
                          transition={{ duration: 0.3, delay: idx * 0.05 }}
                          whileHover={{ scale: 1.1 }}
                        >
                          {course}
                        </motion.span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
