"use client";

import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import rehypeRaw from "rehype-raw";
import "highlight.js/styles/github-dark.css";
import { CodeBlock } from "./code-block";

// Function to generate slug from heading text
function generateSlug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

// Pre-process content to extract heading texts and generate stable IDs
function generateHeadingIds(content: string): Map<string, string> {
  const headingMap = new Map<string, string>();
  const usedIds = new Map<string, number>();
  
  // Simple regex to extract headings (this is a rough approximation)
  const headingRegex = /^#{2,3}\s+(.+)$/gm;
  let match;
  
  while ((match = headingRegex.exec(content)) !== null) {
    const text = match[1].trim();
    const baseSlug = generateSlug(text);
    const count = usedIds.get(baseSlug) || 0;
    usedIds.set(baseSlug, count + 1);
    
    const id = count === 0 ? baseSlug : `${baseSlug}-${count}`;
    headingMap.set(text, id);
  }
  
  return headingMap;
}

export function MarkdownContent({ content }: { content: string }) {
  // Generate stable heading IDs based on content
  const headingIds = useMemo(() => generateHeadingIds(content), [content]);

  const components = useMemo(() => ({
    h2: ({ node, children, ...props }: any) => {
      const text = String(children);
      const id = headingIds.get(text) || generateSlug(text);
      return (
        <h2 id={id} {...props}>
          {children}
        </h2>
      );
    },
    h3: ({ node, children, ...props }: any) => {
      const text = String(children);
      const id = headingIds.get(text) || generateSlug(text);
      return (
        <h3 id={id} {...props}>
          {children}
        </h3>
      );
    },
    pre: ({ node, children, ...props }: any) => {
      return (
        <CodeBlock>
          <pre {...props}>{children}</pre>
        </CodeBlock>
      );
    },
  }), [headingIds]);

  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight, rehypeRaw]}
        components={components}
      >
        {content.trim()}
      </ReactMarkdown>
    </div>
  );
}

