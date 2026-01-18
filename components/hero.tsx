"use client";

import { ArrowRight, Github, Linkedin, Mail } from "lucide-react";
import Link from "next/link";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { MagneticButton } from "./magnetic-button";

interface HeroProps {
  data: {
    name: string;
    title: string;
    tagline: string;
    email: string;
    social: {
      github: string;
      linkedin: string;
      website: string;
    };
  };
}

export function Hero({ data }: HeroProps) {
  const [typedText, setTypedText] = useState("");
  const fullText = data.title;

  useEffect(() => {
    let index = 0;
    const timer = setInterval(() => {
      if (index <= fullText.length) {
        setTypedText(fullText.slice(0, index));
        index++;
      } else {
        clearInterval(timer);
      }
    }, 100);

    return () => clearInterval(timer);
  }, [fullText]);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5,
      },
    },
  };

  return (
    <section className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4">
      <motion.div
        className="max-w-4xl mx-auto text-center space-y-8"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <div className="space-y-4">
          <motion.h1
            className="text-4xl sm:text-6xl font-bold tracking-tight"
            variants={itemVariants}
          >
            Hi, I'm{" "}
            <span className="bg-gradient-to-r from-primary via-chart-2 to-chart-3 bg-clip-text text-transparent animate-gradient">
              {data.name}
            </span>
          </motion.h1>
          <motion.h2
            className="text-2xl sm:text-3xl text-muted-foreground min-h-[2.5rem]"
            variants={itemVariants}
          >
            {typedText}
            <span className="animate-pulse">|</span>
          </motion.h2>
          <motion.p
            className="text-lg text-muted-foreground max-w-2xl mx-auto"
            variants={itemVariants}
          >
            {data.tagline}
          </motion.p>
        </div>

        <motion.div
          className="flex flex-wrap items-center justify-center gap-4"
          variants={itemVariants}
        >
          <MagneticButton>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Link
                href="#projects"
                className="inline-flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:opacity-90 transition-opacity"
              >
                View My Work
                <ArrowRight className="h-4 w-4" />
              </Link>
            </motion.div>
          </MagneticButton>
          <MagneticButton>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Link
                href="#contact"
                className="inline-flex items-center gap-2 px-6 py-3 border border-border rounded-lg hover:bg-accent transition-colors"
              >
                Get In Touch
              </Link>
            </motion.div>
          </MagneticButton>
        </motion.div>

        <motion.div
          className="flex items-center justify-center gap-6 pt-4"
          variants={itemVariants}
        >
          <motion.a
            href={data.social.github}
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 rounded-lg hover:bg-accent transition-colors"
            whileHover={{ scale: 1.2, rotate: 5 }}
            whileTap={{ scale: 0.9 }}
          >
            <Github className="h-6 w-6" />
          </motion.a>
          <motion.a
            href={data.social.linkedin}
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 rounded-lg hover:bg-accent transition-colors"
            whileHover={{ scale: 1.2, rotate: 5 }}
            whileTap={{ scale: 0.9 }}
          >
            <Linkedin className="h-6 w-6" />
          </motion.a>
          <motion.a
            href={`mailto:${data.email}`}
            className="p-2 rounded-lg hover:bg-accent transition-colors"
            whileHover={{ scale: 1.2, rotate: 5 }}
            whileTap={{ scale: 0.9 }}
          >
            <Mail className="h-6 w-6" />
          </motion.a>
        </motion.div>
      </motion.div>
    </section>
  );
}
