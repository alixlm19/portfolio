"use client";

import { useEffect, useRef, useState } from "react";
import gsap from "gsap";
import { X } from "lucide-react";

export function EasterEgg() {
  const [showSecret, setShowSecret] = useState(false);
  const [clickCount, setClickCount] = useState(0);
  const secretRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Konami code implementation
    const konamiCode = ["ArrowUp", "ArrowUp", "ArrowDown", "ArrowDown", "ArrowLeft", "ArrowRight", "ArrowLeft", "ArrowRight", "b", "a"];
    const sequence: string[] = [];

    const handleKeyPress = (event: KeyboardEvent) => {
      sequence.push(event.key);
      if (sequence.length > konamiCode.length) {
        sequence.shift();
      }
      if (sequence.join(",") === konamiCode.join(",")) {
        triggerSecretAnimation();
        sequence.length = 0; // Reset sequence
      }
    };

    // Triple click on logo triggers easter egg
    const handleLogoClick = () => {
      setClickCount((prev) => {
        const newCount = prev + 1;
        if (newCount === 3) {
          triggerSecretAnimation();
          return 0;
        }
        setTimeout(() => setClickCount(0), 2000);
        return newCount;
      });
    };

    const logoElement = document.querySelector('a[href="/"]');
    
    window.addEventListener("keydown", handleKeyPress);
    if (logoElement) {
      logoElement.addEventListener("click", handleLogoClick);
    }

    return () => {
      window.removeEventListener("keydown", handleKeyPress);
      if (logoElement) {
        logoElement.removeEventListener("click", handleLogoClick);
      }
    };
  }, []);

  useEffect(() => {
    if (showSecret && secretRef.current) {
      gsap.fromTo(
        secretRef.current,
        {
          scale: 0,
          rotation: -180,
          opacity: 0,
        },
        {
          scale: 1,
          rotation: 0,
          opacity: 1,
          duration: 0.6,
          ease: "back.out(1.7)",
        }
      );

      // Confetti effect
      const confettiContainer = secretRef.current.querySelector(".confetti-container");
      if (confettiContainer) {
        for (let i = 0; i < 50; i++) {
          const confetti = document.createElement("div");
          confetti.className = "absolute w-2 h-2 rounded-full";
          confetti.style.backgroundColor = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"][
            Math.floor(Math.random() * 5)
          ];
          confetti.style.left = "50%";
          confetti.style.top = "50%";
          confettiContainer.appendChild(confetti);

          gsap.to(confetti, {
            x: (Math.random() - 0.5) * 400,
            y: Math.random() * -400 - 100,
            rotation: Math.random() * 720,
            opacity: 0,
            duration: 1.5 + Math.random(),
            ease: "power2.out",
            onComplete: () => confetti.remove(),
          });
        }
      }
    }
  }, [showSecret]);

  const triggerSecretAnimation = () => {
    setShowSecret(true);
    
    // Make the whole page do a little wiggle
    const mainContent = document.querySelector("main");
    if (mainContent) {
      gsap.to(mainContent, {
        rotation: 2,
        duration: 0.1,
        yoyo: true,
        repeat: 5,
        ease: "power1.inOut",
        onComplete: () => {
          gsap.set(mainContent, { rotation: 0 });
        },
      });
    }
  };

  if (!showSecret) return null;

  return (
    <div 
      className="fixed top-0 left-0 w-full h-screen bg-black/50 backdrop-blur-sm z-[100] flex items-center justify-center p-4"
      style={{ margin: 0, padding: '1rem' }}
    >
      <div
        ref={secretRef}
        className="relative bg-gradient-to-br from-primary to-chart-3 p-8 rounded-2xl max-w-md w-full text-center shadow-2xl"
      >
        <div className="confetti-container absolute inset-0" />
        <button
          onClick={() => setShowSecret(false)}
          className="absolute top-4 right-4 p-2 hover:bg-white/20 rounded-full transition-colors"
        >
          <X className="h-5 w-5 text-white" />
        </button>
        <div className="relative z-10">
          <div className="text-6xl mb-4 animate-bounce">🎉</div>
          <h3 className="text-2xl font-bold text-white mb-3">
            You Found the Secret!
          </h3>
          <p className="text-white/90 mb-4">
            Congratulations! You've unlocked the hidden easter egg. 
            <br />
            <span className="text-sm">
              (Triple-click the logo or use the Konami code to see this again!)
            </span>
          </p>
          <div className="text-4xl">
            ✨🚀💻🎨✨
          </div>
        </div>
      </div>
    </div>
  );
}
