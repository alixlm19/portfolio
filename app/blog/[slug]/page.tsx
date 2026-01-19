import Link from "next/link";
import { Calendar, Clock, ArrowLeft, User, ChevronLeft, ChevronRight } from "lucide-react";
import { getBlogPost, getBlogPosts } from "@/lib/blog";
import { notFound } from "next/navigation";
import { Navbar } from "@/components/navbar";
import { MarkdownContent } from "@/components/markdown-content";
import { BlogScrollTimeline } from "@/components/blog-scroll-timeline";
import { LikeButton } from "@/components/like-button";
import { getLikes } from "@/app/actions/likes";
import { BlogTableOfContents } from "@/components/blog-table-of-contents";
import { ShareArticle } from "@/components/share-article";
import { FunCursor } from "@/components/fun-cursor";
import { ClickEffects } from "@/components/click-effects";

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

  // Get all posts and find current post index for navigation
  const allPosts = getBlogPosts();
  const currentIndex = allPosts.findIndex((p) => p.slug === params.slug);
  const previousPost = currentIndex < allPosts.length - 1 ? allPosts[currentIndex + 1] : null;
  const nextPost = currentIndex > 0 ? allPosts[currentIndex - 1] : null;

  // Get initial like count
  const initialLikes = await getLikes(params.slug);

  return (
    <>
      <FunCursor />
      <ClickEffects />
      <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
        <Navbar />
        <BlogScrollTimeline />
      
      <div className="max-w-7xl mx-auto px-4 py-12 sm:py-20">
        <div className="grid grid-cols-1 lg:grid-cols-[240px_1fr_240px] gap-8 xl:gap-12">
          {/* Left Sidebar - Table of Contents */}
          <aside className="hidden lg:block">
            <BlogTableOfContents />
          </aside>

          {/* Main Content */}
          <article className="min-w-0">
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
                    className="px-3 py-1 rounded-full bg-secondary/10 text-secondary text-sm font-medium"
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

            {/* Like Section */}
            <div className="mt-12 flex justify-center">
              <LikeButton slug={params.slug} initialLikes={initialLikes} />
            </div>

            {/* Post Navigation */}
            <nav className="mt-16 pt-8 border-t">
              <div className="flex flex-col sm:flex-row justify-between gap-4">
                {previousPost ? (
                  <Link
                    href={`/blog/${previousPost.slug}`}
                    className="group flex-1 p-4 rounded-lg border hover:border-primary/50 hover:bg-accent/50 transition-all"
                  >
                    <div className="flex items-center gap-2 text-sm text-muted-foreground mb-2">
                      <ChevronLeft className="h-4 w-4" />
                      <span>Previous Post</span>
                    </div>
                    <h3 className="font-semibold group-hover:text-primary transition-colors line-clamp-2">
                      {previousPost.title}
                    </h3>
                  </Link>
                ) : (
                  <div className="flex-1" />
                )}

                {nextPost ? (
                  <Link
                    href={`/blog/${nextPost.slug}`}
                    className="group flex-1 p-4 rounded-lg border hover:border-primary/50 hover:bg-accent/50 transition-all text-right"
                  >
                    <div className="flex items-center justify-end gap-2 text-sm text-muted-foreground mb-2">
                      <span>Next Post</span>
                      <ChevronRight className="h-4 w-4" />
                    </div>
                    <h3 className="font-semibold group-hover:text-primary transition-colors line-clamp-2">
                      {nextPost.title}
                    </h3>
                  </Link>
                ) : (
                  <div className="flex-1" />
                )}
              </div>

              <Link
                href="/blog"
                className="inline-flex items-center gap-2 text-primary hover:gap-3 transition-all font-medium mt-6"
              >
                <ArrowLeft className="h-4 w-4" />
                Read more articles
              </Link>
            </nav>
          </article>

          {/* Right Sidebar - Share Panel */}
          <aside className="hidden lg:block">
            <ShareArticle title={post.title} slug={params.slug} />
          </aside>
        </div>
      </div>
    </div>
    </>
  );
}
