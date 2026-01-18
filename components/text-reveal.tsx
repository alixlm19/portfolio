"use client";

import { useEffect, useRef } from "react";
import gsap from "gsap";

export function TextReveal({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  const textRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = textRef.current;
    if (!element) return;

    // Split text into characters for animation
    const text = element.textContent || "";
    element.innerHTML = text
      .split("")
      .map((char, i) => `<span class="inline-block" style="display: inline-block">${char === " " ? "&nbsp;" : char}</span>`)
      .join("");

    const chars = element.querySelectorAll("span");

    // Animate characters on scroll into view
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            gsap.fromTo(
              chars,
              {
                y: 50,
                opacity: 0,
                rotationX: -90,
              },
              {
                y: 0,
                opacity: 1,
                rotationX: 0,
                duration: 0.8,
                stagger: 0.02,
                ease: "back.out(1.7)",
              }
            );
            observer.disconnect();
          }
        });
      },
      { threshold: 0.5 }
    );

    observer.observe(element);

    return () => observer.disconnect();
  }, [children]);

  return (
    <div ref={textRef} className={className}>
      {children}
    </div>
  );
}
