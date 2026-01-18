import Link from "next/link";
import { Calendar, Clock, ArrowLeft } from "lucide-react";
import { getBlogPost, getBlogPosts } from "@/lib/blog";
import { notFound } from "next/navigation";

export async function generateStaticParams() {
  const posts = getBlogPosts();
  return posts.map((post) => ({
    slug: post.slug,
  }));
}

export default async function BlogPostPage(props: { params: Promise<{ slug: string }> }) {
  const params = await props.params;
  const post = getBlogPost(params.slug);

  if (!post) {
    notFound();
  }

  return (
    <div className="min-h-screen">
      <article className="max-w-4xl mx-auto px-4 py-20">
        <Link
          href="/blog"
          className="inline-flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors mb-8"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Blog
        </Link>

        <header className="mb-8">
          <h1 className="text-4xl sm:text-5xl font-bold mb-4">{post.title}</h1>

          <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground mb-6">
            <div className="flex items-center gap-2">
              <Calendar className="h-4 w-4" />
              {new Date(post.date).toLocaleDateString("en-US", {
                month: "long",
                day: "numeric",
                year: "numeric",
              })}
            </div>
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4" />
              {post.readTime}
            </div>
            <span>By {post.author}</span>
          </div>

          <div className="flex flex-wrap gap-2">
            {post.tags.map((tag) => (
              <span
                key={tag}
                className="px-3 py-1 rounded-full bg-secondary text-secondary-foreground text-xs"
              >
                {tag}
              </span>
            ))}
          </div>
        </header>

        <div className="prose prose-lg dark:prose-invert max-w-none">
          <div
            dangerouslySetInnerHTML={{
              __html: post.content
                .split("\n")
                .map((line) => {
                  if (line.startsWith("# ")) {
                    return `<h1>${line.slice(2)}</h1>`;
                  }
                  if (line.startsWith("## ")) {
                    return `<h2>${line.slice(3)}</h2>`;
                  }
                  if (line.startsWith("### ")) {
                    return `<h3>${line.slice(4)}</h3>`;
                  }
                  if (line.startsWith("- ")) {
                    return `<li>${line.slice(2)}</li>`;
                  }
                  if (line.includes("```")) {
                    return line.replace("```python", "<pre><code>").replace("```", "</code></pre>");
                  }
                  if (line.trim() === "") {
                    return "<br/>";
                  }
                  return `<p>${line}</p>`;
                })
                .join(""),
            }}
          />
        </div>
      </article>
    </div>
  );
}
