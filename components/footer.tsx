import { Github, Linkedin, Mail } from "lucide-react";
import { getPersonalInfo } from "@/lib/data";

export function Footer() {
  const data = getPersonalInfo();

  return (
    <footer className="border-t py-12 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-muted-foreground text-sm">
            © {new Date().getFullYear()} {data.name}. All rights reserved.
          </p>

          <div className="flex items-center gap-4">
            <a
              href={data.social.github}
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg hover:bg-accent transition-colors"
            >
              <Github className="h-5 w-5" />
            </a>
            <a
              href={data.social.linkedin}
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg hover:bg-accent transition-colors"
            >
              <Linkedin className="h-5 w-5" />
            </a>
            <a
              href={`mailto:${data.email}`}
              className="p-2 rounded-lg hover:bg-accent transition-colors"
            >
              <Mail className="h-5 w-5" />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
