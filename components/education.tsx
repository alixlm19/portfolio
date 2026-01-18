import { GraduationCap } from "lucide-react";
import { getEducation } from "@/lib/data";

export function Education() {
  const { education } = getEducation();

  return (
    <section id="education" className="py-20 px-4 bg-muted/30">
      <div className="max-w-7xl mx-auto">
        <div className="text-center space-y-4 mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold">Education</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Academic foundation in Computer Science, Statistics, and Machine Learning.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {education.map((edu, index) => (
            <div
              key={index}
              className="p-6 rounded-lg border bg-card hover:shadow-lg transition-shadow"
            >
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-lg bg-primary/10">
                  <GraduationCap className="h-6 w-6 text-primary" />
                </div>
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
                      {edu.coursework.map((course) => (
                        <span
                          key={course}
                          className="px-2 py-1 rounded-md bg-secondary text-secondary-foreground text-xs"
                        >
                          {course}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
