"use client";

import { useEffect } from "react";
import gsap from "gsap";

export function ClickEffects() {
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      // Create particle burst on click
      const colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F7DC6F", "#BB8FCE"];
      const particleCount = 12;

      for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement("div");
        particle.className = "fixed w-2 h-2 rounded-full pointer-events-none z-50";
        particle.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
        particle.style.left = `${e.clientX}px`;
        particle.style.top = `${e.clientY}px`;
        document.body.appendChild(particle);

        const angle = (Math.PI * 2 * i) / particleCount;
        const velocity = 50 + Math.random() * 50;

        gsap.to(particle, {
          x: Math.cos(angle) * velocity,
          y: Math.sin(angle) * velocity,
          opacity: 0,
          scale: 0,
          duration: 0.6 + Math.random() * 0.4,
          ease: "power2.out",
          onComplete: () => particle.remove(),
        });
      }

      // Ripple effect
      const ripple = document.createElement("div");
      ripple.className = "fixed rounded-full border-2 border-primary pointer-events-none z-50";
      ripple.style.left = `${e.clientX}px`;
      ripple.style.top = `${e.clientY}px`;
      ripple.style.width = "10px";
      ripple.style.height = "10px";
      ripple.style.transform = "translate(-50%, -50%)";
      document.body.appendChild(ripple);

      gsap.to(ripple, {
        width: 100,
        height: 100,
        opacity: 0,
        duration: 0.6,
        ease: "power2.out",
        onComplete: () => ripple.remove(),
      });
    };

    // Only trigger on specific elements to not be too overwhelming
    const interactiveElements = document.querySelectorAll("a, button");
    interactiveElements.forEach((el) => {
      el.addEventListener("click", handleClick as EventListener);
    });

    return () => {
      interactiveElements.forEach((el) => {
        el.removeEventListener("click", handleClick as EventListener);
      });
    };
  }, []);

  return null;
}
