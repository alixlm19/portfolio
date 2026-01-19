"use client";

import { useEffect, useState } from "react";

interface TOCItem {
  id: string;
  title: string;
  level: number;
}

export function BlogTableOfContents() {
  const [headings, setHeadings] = useState<TOCItem[]>([]);
  const [activeId, setActiveId] = useState<string>("");

  useEffect(() => {
    // Extract headings from the page - only from article content
    const articleContent = document.querySelector("article .prose");
    if (!articleContent) return;
    
    const elements = Array.from(articleContent.querySelectorAll("h2, h3"));
    const items: TOCItem[] = elements
      .map((element, index) => ({
        id: element.id || `heading-${index}`,
        title: element.textContent || "",
        level: parseInt(element.tagName.substring(1)),
      }))
      .filter((item) => item.id && item.title); // Filter out empty items
    
    setHeadings(items);

    // Track which sections are currently visible
    const visibleSections = new Set<string>();

    // Intersection observer for tracking visible headings
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            visibleSections.add(entry.target.id);
          } else {
            visibleSections.delete(entry.target.id);
          }
        });

        // Find the topmost visible section
        if (visibleSections.size > 0) {
          const firstVisible = items.find((item) => visibleSections.has(item.id));
          if (firstVisible) {
            setActiveId(firstVisible.id);
          }
        }
      },
      { 
        rootMargin: "-80px 0px -80% 0px",
        threshold: [0, 0.25, 0.5, 0.75, 1]
      }
    );

    elements.forEach((element) => observer.observe(element));

    // Fallback: update on scroll for better accuracy
    const handleScroll = () => {
      const scrollPosition = window.scrollY + 100;
      
      for (let i = items.length - 1; i >= 0; i--) {
        const element = document.getElementById(items[i].id);
        if (element && element.offsetTop <= scrollPosition) {
          setActiveId(items[i].id);
          break;
        }
      }
    };

    window.addEventListener("scroll", handleScroll, { passive: true });

    return () => {
      observer.disconnect();
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  if (headings.length === 0) return null;

  return (
    <nav className="sticky top-24 space-y-1">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-4">
        Index
      </h3>
      <ul className="space-y-2 text-sm">
        {headings.map((heading, index) => (
          <li key={`${heading.id}-${index}`}>
            <a
              href={`#${heading.id}`}
              onClick={(e) => {
                e.preventDefault();
                document.getElementById(heading.id)?.scrollIntoView({
                  behavior: "smooth",
                  block: "start",
                });
                setActiveId(heading.id);
              }}
              className={`block transition-colors hover:text-foreground ${
                activeId === heading.id
                  ? "text-primary font-medium border-l-2 border-primary"
                  : "text-muted-foreground border-l-2 border-transparent"
              } ${
                heading.level === 3 ? "pl-7 text-xs" : "pl-3"
              }`}
            >
              {heading.title}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
