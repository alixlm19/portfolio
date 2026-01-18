import { ArrowRight, Github, Linkedin, Mail } from "lucide-react";
import Link from "next/link";
import { getPersonalInfo } from "@/lib/data";

export function Hero() {
  const data = getPersonalInfo();

  return (
    <section className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4">
      <div className="max-w-4xl mx-auto text-center space-y-8">
        <div className="space-y-4">
          <h1 className="text-4xl sm:text-6xl font-bold tracking-tight">
            Hi, I'm{" "}
            <span className="bg-gradient-to-r from-primary to-chart-2 bg-clip-text text-transparent">
              {data.name}
            </span>
          </h1>
          <h2 className="text-2xl sm:text-3xl text-muted-foreground">
            {data.title}
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            {data.tagline}
          </p>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-4">
          <Link
            href="#projects"
            className="inline-flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:opacity-90 transition-opacity"
          >
            View My Work
            <ArrowRight className="h-4 w-4" />
          </Link>
          <Link
            href="#contact"
            className="inline-flex items-center gap-2 px-6 py-3 border border-border rounded-lg hover:bg-accent transition-colors"
          >
            Get In Touch
          </Link>
        </div>

        <div className="flex items-center justify-center gap-6 pt-4">
          <a
            href={data.social.github}
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 rounded-lg hover:bg-accent transition-colors"
          >
            <Github className="h-6 w-6" />
          </a>
          <a
            href={data.social.linkedin}
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 rounded-lg hover:bg-accent transition-colors"
          >
            <Linkedin className="h-6 w-6" />
          </a>
          <a
            href={`mailto:${data.email}`}
            className="p-2 rounded-lg hover:bg-accent transition-colors"
          >
            <Mail className="h-6 w-6" />
          </a>
        </div>
      </div>
    </section>
  );
}
