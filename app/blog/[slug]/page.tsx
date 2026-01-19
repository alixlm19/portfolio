import Link from "next/link";
import { Calendar, Clock, ArrowLeft, User } from "lucide-react";
import { getBlogPost, getBlogPosts } from "@/lib/blog";
import { notFound } from "next/navigation";
import { MarkdownContent } from "@/components/markdown-content";

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
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <article className="max-w-4xl mx-auto px-4 py-12 sm:py-20">
        <Link
          href="/blog"
          className="group inline-flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors mb-8"
        >
          <ArrowLeft className="h-4 w-4 group-hover:-translate-x-1 transition-transform" />
          Back to Blog
        </Link>

        <header className="mb-12">
          <div className="flex flex-wrap gap-2 mb-4">
            {post.tags.map((tag) => (
              <span
                key={tag}
                className="px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-medium"
              >
                {tag}
              </span>
            ))}
          </div>

          <h1 className="text-4xl sm:text-6xl font-bold mb-6 leading-tight bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
            {post.title}
          </h1>

          <div className="flex flex-wrap items-center gap-4 sm:gap-6 text-sm text-muted-foreground p-4 rounded-lg bg-muted/50 border">
            <div className="flex items-center gap-2">
              <User className="h-4 w-4" />
              <span className="font-medium">{post.author}</span>
            </div>
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
          </div>
        </header>

        <MarkdownContent content={post.content} />

        <div className="mt-16 pt-8 border-t">
          <Link
            href="/blog"
            className="inline-flex items-center gap-2 text-primary hover:gap-3 transition-all font-medium"
          >
            <ArrowLeft className="h-4 w-4" />
            Read more articles
          </Link>
        </div>
      </article>
    </div>
  );
}
