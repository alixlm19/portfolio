import { Navbar } from "@/components/navbar";
import { Hero } from "@/components/hero";
import { About } from "@/components/about";
import { Experience } from "@/components/experience";
import { Education } from "@/components/education";
import { Projects } from "@/components/projects";
import { Contact } from "@/components/contact";
import { Footer } from "@/components/footer";
import { FunCursor } from "@/components/fun-cursor";
import { EasterEgg } from "@/components/easter-egg";
import { ClickEffects } from "@/components/click-effects";
import { ScrollProgress } from "@/components/scroll-progress";
import { AnimatedGradientOrbs } from "@/components/animated-gradient-orbs";
import { getPersonalInfo, getProjects, getExperience, getEducation } from "@/lib/data";

export default function Page() {
  const personalInfo = getPersonalInfo();
  const { projects } = getProjects();
  const { experience } = getExperience();
  const { education } = getEducation();

  return (
    <>
      <FunCursor />
      <EasterEgg />
      <ClickEffects />
      <ScrollProgress />
      <AnimatedGradientOrbs />
      <div className="min-h-screen relative">
        <Navbar />
        <main>
          <Hero data={personalInfo} />
          <About data={personalInfo} />
          <Experience experience={experience} />
          <Projects projects={projects} />
          <Education education={education} />
          <Contact data={personalInfo} />
        </main>
        <Footer />
      </div>
    </>
  );
}