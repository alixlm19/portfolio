"use client";

import { useEffect, useRef } from "react";
import gsap from "gsap";

export function AnimatedGradientOrbs() {
  const orb1Ref = useRef<HTMLDivElement>(null);
  const orb2Ref = useRef<HTMLDivElement>(null);
  const orb3Ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const orbs = [orb1Ref.current, orb2Ref.current, orb3Ref.current];

    orbs.forEach((orb, index) => {
      if (!orb) return;

      // Different animation parameters for each orb
      const duration = 20 + index * 5;
      const delay = index * 2;

      gsap.to(orb, {
        x: `+=${100 + index * 50}`,
        y: `+=${80 + index * 40}`,
        duration: duration,
        ease: "sine.inOut",
        repeat: -1,
        yoyo: true,
        delay: delay,
      });

      // Add rotation for extra movement
      gsap.to(orb, {
        rotation: 360,
        duration: duration * 1.5,
        ease: "none",
        repeat: -1,
      });
    });
  }, []);

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
      {/* Orb 1 - Primary color (orange) */}
      <div
        ref={orb1Ref}
        className="absolute top-20 right-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl"
      />
      
      {/* Orb 2 - Secondary color (blue) */}
      <div
        ref={orb2Ref}
        className="absolute bottom-1/4 left-1/4 w-[500px] h-[500px] bg-secondary/10 rounded-full blur-3xl"
      />
      
      {/* Orb 3 - Mix of both */}
      <div
        ref={orb3Ref}
        className="absolute top-1/2 right-1/3 w-80 h-80 bg-gradient-to-br from-primary/8 to-secondary/8 rounded-full blur-3xl"
      />
    </div>
  );
}
