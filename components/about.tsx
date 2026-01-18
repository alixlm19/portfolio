import { Code2, Database, Brain, Workflow } from "lucide-react";
import { getPersonalInfo } from "@/lib/data";

const iconMap = {
  Brain,
  Code2,
  Database,
  Workflow,
};

export function About() {
  const data = getPersonalInfo();

  return (
    <section id="about" className="py-20 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="text-center space-y-4 mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold">About Me</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            {data.about.description}
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {data.about.skills.map((skill) => {
            const Icon = iconMap[skill.icon as keyof typeof iconMap];
            return (
              <div
                key={skill.title}
                className="p-6 rounded-lg border bg-card hover:shadow-lg transition-shadow"
              >
                <div className="flex items-start gap-4">
                  <div className="p-3 rounded-lg bg-primary/10">
                    <Icon className="h-6 w-6 text-primary" />
                  </div>
                  <div className="space-y-2">
                    <h3 className="text-xl font-semibold">{skill.title}</h3>
                    <p className="text-muted-foreground">{skill.description}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-12 p-8 rounded-lg border bg-card">
          <h3 className="text-2xl font-semibold mb-4">Tech Stack</h3>
          <div className="flex flex-wrap gap-2">
            {data.techStack.map((tech) => (
              <span
                key={tech}
                className="px-4 py-2 rounded-full bg-secondary text-secondary-foreground text-sm"
              >
                {tech}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
