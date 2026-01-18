"use client";

import { useEffect, useRef } from "react";
import gsap from "gsap";

export function FloatingEmojis() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const emojis = ["💻", "🚀", "⚡", "🎯", "🔥", "💡", "🎨", "🌟", "✨", "🎉"];
    const emojiElements: HTMLDivElement[] = [];

    // Create floating emojis
    for (let i = 0; i < 15; i++) {
      const emoji = document.createElement("div");
      emoji.textContent = emojis[Math.floor(Math.random() * emojis.length)];
      emoji.className = "absolute text-4xl pointer-events-none opacity-20";
      emoji.style.left = `${Math.random() * 100}%`;
      emoji.style.top = `${Math.random() * 100}%`;
      container.appendChild(emoji);
      emojiElements.push(emoji);

      // Animate each emoji
      gsap.to(emoji, {
        y: `${-100 - Math.random() * 100}`,
        x: `${Math.random() * 100 - 50}`,
        rotation: Math.random() * 360,
        duration: 10 + Math.random() * 10,
        repeat: -1,
        ease: "none",
        delay: Math.random() * 5,
      });

      // Pulse animation
      gsap.to(emoji, {
        scale: 1.2,
        duration: 2 + Math.random() * 2,
        repeat: -1,
        yoyo: true,
        ease: "power1.inOut",
      });
    }

    return () => {
      emojiElements.forEach((emoji) => emoji.remove());
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="fixed inset-0 pointer-events-none overflow-hidden z-0"
    />
  );
}
