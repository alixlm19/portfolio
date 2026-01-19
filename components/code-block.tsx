"use client";

import { useState, ReactNode, useRef, useEffect } from "react";
import { Check, Copy } from "lucide-react";

interface CodeBlockProps {
  children: ReactNode;
}

export function CodeBlock({ children }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const [code, setCode] = useState("");
  const preRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Extract text content from the code element
    if (preRef.current) {
      const codeElement = preRef.current.querySelector("code");
      if (codeElement) {
        setCode(codeElement.textContent || "");
      }
    }
  }, [children]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  return (
    <div ref={preRef} className="relative">
      <button
        onClick={handleCopy}
        className="absolute top-3 right-3 z-10 p-2 rounded-md bg-zinc-700/90 hover:bg-zinc-600 border border-zinc-500/50 transition-colors backdrop-blur-sm"
        aria-label="Copy code"
      >
        {copied ? (
          <Check className="h-4 w-4 text-green-400" />
        ) : (
          <Copy className="h-4 w-4 text-zinc-200" />
        )}
      </button>
      {children}
    </div>
  );
}
