export interface BlogPost {
  slug: string;
  title: string;
  description: string;
  date: string;
  tags: string[];
  content: string;
  author: string;
  readTime?: string; // Computed from content
  featured?: boolean; // Optional featured flag
}
