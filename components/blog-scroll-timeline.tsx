"use client";

import { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

export function BlogScrollTimeline() {
  const timelineRef = useRef<HTMLDivElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!timelineRef.current || !progressRef.current) return;

    // Animate the vertical timeline progress line
    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: "article",
        start: "top 20%",
        end: "bottom bottom",
        scrub: 1,
      },
    });

    tl.fromTo(
      progressRef.current,
      { height: "0%" },
      { height: "100%", ease: "none" }
    );

    // Animate all prose headings (h2, h3) as scroll markers
    const headings = document.querySelectorAll(".prose h2, .prose h3");
    headings.forEach((heading, index) => {
      gsap.fromTo(
        heading,
        {
          opacity: 0,
          x: -30,
        },
        {
          opacity: 1,
          x: 0,
          duration: 0.8,
          ease: "power2.out",
          scrollTrigger: {
            trigger: heading,
            start: "top 80%",
            end: "top 50%",
            toggleActions: "play none none none",
          },
        }
      );
    });

    // Animate paragraphs with fade in
    const paragraphs = document.querySelectorAll(".prose p");
    paragraphs.forEach((p) => {
      gsap.fromTo(
        p,
        {
          opacity: 0,
          y: 20,
        },
        {
          opacity: 1,
          y: 0,
          duration: 0.6,
          ease: "power2.out",
          scrollTrigger: {
            trigger: p,
            start: "top 85%",
            toggleActions: "play none none none",
          },
        }
      );
    });

    // Animate code blocks
    const codeBlocks = document.querySelectorAll(".prose pre");
    codeBlocks.forEach((block) => {
      gsap.fromTo(
        block,
        {
          opacity: 0,
          scale: 0.95,
        },
        {
          opacity: 1,
          scale: 1,
          duration: 0.8,
          ease: "back.out(1.2)",
          scrollTrigger: {
            trigger: block,
            start: "top 85%",
            toggleActions: "play none none none",
          },
        }
      );
    });

    // Animate lists
    const listItems = document.querySelectorAll(".prose li");
    listItems.forEach((item, index) => {
      gsap.fromTo(
        item,
        {
          opacity: 0,
          x: -20,
        },
        {
          opacity: 1,
          x: 0,
          duration: 0.5,
          delay: index * 0.05,
          ease: "power2.out",
          scrollTrigger: {
            trigger: item,
            start: "top 90%",
            toggleActions: "play none none none",
          },
        }
      );
    });

    return () => {
      ScrollTrigger.getAll().forEach((trigger) => trigger.kill());
    };
  }, []);

  return (
    <div
      ref={timelineRef}
      className="fixed left-8 top-32 bottom-32 w-1 hidden lg:block z-10"
    >
      <div className="absolute inset-0 bg-border rounded-full" />
      <div
        ref={progressRef}
        className="absolute top-0 left-0 w-full bg-gradient-to-b from-secondary via-primary to-secondary rounded-full"
        style={{ height: "0%" }}
      />
      {/* Timeline dots */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-3 h-3 bg-secondary rounded-full shadow-lg shadow-secondary/50" />
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-3 h-3 bg-primary rounded-full shadow-lg shadow-primary/50" />
    </div>
  );
}
