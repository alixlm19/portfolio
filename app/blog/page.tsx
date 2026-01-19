import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { getBlogPosts } from "@/lib/blog";

export default function BlogPage() {
  const posts = getBlogPosts();
  const featuredPost = posts[0];
  const otherPosts = posts.slice(1);

  return (
    <div className="min-h-screen">
      <div className="max-w-4xl mx-auto px-6 py-16 sm:py-24">
        {/* Header */}
        <div className="mb-16">
          <Link
            href="/"
            className="group inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-12"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Home
          </Link>
          <h1 className="text-6xl sm:text-7xl font-bold tracking-tight mb-6">
            Writing
          </h1>
          <p className="text-lg text-muted-foreground max-w-xl">
            Essays on machine learning, software, and building things.
          </p>
        </div>

        {posts.length > 0 ? (
          <>
            {/* Featured Post */}
            {featuredPost && (
              <Link href={`/blog/${featuredPost.slug}`} className="group block mb-20">
                <article className="border-b border-border pb-12">
                  <div className="mb-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                    Featured
                  </div>
                  <h2 className="text-3xl sm:text-4xl font-bold mb-4 group-hover:text-primary transition-colors">
                    {featuredPost.title}
                  </h2>
                  <p className="text-lg text-muted-foreground mb-6 leading-relaxed">
                    {featuredPost.description}
                  </p>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground">
                    <time dateTime={featuredPost.date}>
                      {new Date(featuredPost.date).toLocaleDateString("en-US", {
                        month: "long",
                        day: "numeric",
                        year: "numeric",
                      })}
                    </time>
                    <span>·</span>
                    <span>{featuredPost.readTime}</span>
                  </div>
                </article>
              </Link>
            )}

            {/* Timeline List */}
            <div className="space-y-6">
              {otherPosts.map((post) => (
                <Link
                  key={post.slug}
                  href={`/blog/${post.slug}`}
                  className="group block"
                >
                  <article className="p-6 rounded-lg border border-border hover:border-primary/50 hover:bg-accent/50 transition-all">
                    <div className="flex items-baseline gap-8 mb-3">
                      <time
                        dateTime={post.date}
                        className="text-sm text-muted-foreground font-mono flex-shrink-0"
                      >
                        {new Date(post.date).toLocaleDateString("en-US", {
                          month: "short",
                          year: "numeric",
                        })}
                      </time>
                      <div className="flex-1 min-w-0">
                        <h2 className="text-xl sm:text-2xl font-semibold mb-2 group-hover:text-primary transition-colors">
                          {post.title}
                        </h2>
                        <p className="text-muted-foreground leading-relaxed line-clamp-2">
                          {post.description}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 ml-[7.5rem] text-xs">
                      {post.tags.slice(0, 3).map((tag) => (
                        <span
                          key={tag}
                          className="text-muted-foreground"
                        >
                          #{tag}
                        </span>
                      ))}
                    </div>
                  </article>
                </Link>
              ))}
            </div>
          </>
        ) : (
          <div className="py-20 text-center">
            <p className="text-muted-foreground">No posts yet. Check back soon!</p>
          </div>
        )}
      </div>
    </div>
  );
}
