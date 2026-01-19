import fs from "fs";
import path from "path";
import matter from "gray-matter";
import type { BlogPost } from "@/types/blog";

// Calculate reading time based on word count (200 words per minute)
function calculateReadTime(content: string): string {
  const wordsPerMinute = 200;
  const words = content.trim().split(/\s+/).length;
  const minutes = Math.ceil(words / wordsPerMinute);
  return `${minutes} min read`;
}

function loadBlogPostsFromFiles(): BlogPost[] {
  const postsDirectory = path.join(process.cwd(), "content/posts");
  
  // Check if directory exists
  if (!fs.existsSync(postsDirectory)) {
    console.warn("Blog posts directory not found:", postsDirectory);
    return [];
  }

  const fileNames = fs.readdirSync(postsDirectory);
  const posts: BlogPost[] = [];

  for (const fileName of fileNames) {
    if (!fileName.endsWith(".md")) continue;

    const fullPath = path.join(postsDirectory, fileName);
    const fileContents = fs.readFileSync(fullPath, "utf8");
    const { data, content } = matter(fileContents);

    const slug = fileName.replace(/\.md$/, "");

    posts.push({
      slug,
      title: data.title || slug,
      description: data.description || "",
      date: data.date || new Date().toISOString().split("T")[0],
      tags: data.tags || [],
      author: data.author || "Alix Leon",
      content: content,
      featured: data.featured,
    });
  }

  return posts;
}

export const blogPosts: BlogPost[] = loadBlogPostsFromFiles();

export function getBlogPosts(): BlogPost[] {
  return blogPosts
    .map((post) => ({
      ...post,
      readTime: calculateReadTime(post.content),
    }))
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

export function getBlogPost(slug: string): BlogPost | undefined {
  const post = blogPosts.find((post) => post.slug === slug);
  if (!post) return undefined;
  return {
    ...post,
    readTime: calculateReadTime(post.content),
  };
}
