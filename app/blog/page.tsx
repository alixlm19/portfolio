import Link from "next/link";
import { ArrowLeft, Heart, TrendingUp, BookOpen, Clock, Sparkles } from "lucide-react";
import { getBlogPosts } from "@/lib/blog";
import { getLikes } from "@/app/actions/likes";

export default async function BlogPage() {
  const posts = getBlogPosts();
  const featuredPost = posts[0];
  const otherPosts = posts.slice(1);

  // Get like counts for all posts
  const likeCounts = await Promise.all(
    posts.map((post) => getLikes(post.slug))
  );
  const likesMap = Object.fromEntries(
    posts.map((post, i) => [post.slug, likeCounts[i]])
  );

  // Get popular posts (top 3 by likes, excluding featured)
  const popularPosts = posts
    .map((post) => ({
      ...post,
      likes: likesMap[post.slug] || 0,
    }))
    .filter((post) => post.slug !== featuredPost?.slug)
    .sort((a, b) => b.likes - a.likes)
    .slice(0, 3)
    .filter((post) => post.likes > 0);

  // Calculate stats
  const totalLikes = Object.values(likesMap).reduce((sum, likes) => sum + likes, 0);
  const totalReadTime = posts.reduce((sum, post) => {
    const minutes = parseInt(post.readTime?.match(/\d+/)?.[0] || "0");
    return sum + minutes;
  }, 0);

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary/5">
      {/* Decorative elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 right-20 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-40 left-20 w-96 h-96 bg-secondary/5 rounded-full blur-3xl" />
      </div>

      <div className="relative max-w-4xl mx-auto px-6 py-16 sm:py-24">
        {/* Header */}
        <div className="mb-16">
          <Link
            href="/"
            className="group inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-12"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Home
          </Link>
          <div className="flex items-center gap-3 mb-6">
            <Sparkles className="h-8 w-8 text-primary" />
            <h1 className="text-6xl sm:text-7xl font-bold tracking-tight">
              Writing
            </h1>
          </div>
          <p className="text-lg text-muted-foreground max-w-xl mb-8">
            Essays on machine learning, software, and building things.
          </p>

          {/* Stats Bar */}
          <div className="flex flex-wrap gap-6 p-4 rounded-lg bg-muted/50 border">
            <div className="flex items-center gap-2">
              <BookOpen className="h-4 w-4 text-primary" />
              <span className="text-sm">
                <span className="font-bold text-foreground">{posts.length}</span>
                <span className="text-muted-foreground"> {posts.length === 1 ? "article" : "articles"}</span>
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Heart className="h-4 w-4 text-secondary" />
              <span className="text-sm">
                <span className="font-bold text-foreground">{totalLikes}</span>
                <span className="text-muted-foreground"> {totalLikes === 1 ? "like" : "likes"}</span>
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm">
                <span className="font-bold text-foreground">{totalReadTime}</span>
                <span className="text-muted-foreground"> min total</span>
              </span>
            </div>
          </div>
        </div>

        {posts.length > 0 ? (
          <>
            {/* Featured Post */}
            {featuredPost && (
              <Link href={`/blog/${featuredPost.slug}`} className="group block mb-20">
                <article className="relative overflow-hidden rounded-xl border border-border p-8 bg-gradient-to-br from-primary/5 via-background to-secondary/5 hover:border-primary/50 transition-all">
                  <div className="mb-3 flex items-center gap-2">
                    <span className="text-xs font-medium text-primary uppercase tracking-wider">
                      Featured
                    </span>
                    <span className="h-1 w-1 rounded-full bg-primary" />
                    <span className="text-xs text-muted-foreground">Latest</span>
                  </div>
                  <h2 className="text-3xl sm:text-4xl font-bold mb-4 group-hover:text-primary transition-colors">
                    {featuredPost.title}
                  </h2>
                  <p className="text-lg text-muted-foreground mb-6 leading-relaxed">
                    {featuredPost.description}
                  </p>
                  <div className="flex flex-wrap items-center gap-3 mb-4">
                    {featuredPost.tags.slice(0, 4).map((tag) => (
                      <span
                        key={tag}
                        className="px-2.5 py-1 rounded-full bg-secondary/10 text-secondary text-xs font-medium"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
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
                    <span>·</span>
                    <span className="flex items-center gap-1.5">
                      <Heart className="h-3.5 w-3.5" />
                      {likesMap[featuredPost.slug] || 0}
                    </span>
                  </div>
                </article>
              </Link>
            )}

            {/* Popular Posts Section */}
            {popularPosts.length > 0 && (
              <div className="mb-20">
                <div className="flex items-center gap-3 mb-6">
                  <TrendingUp className="h-5 w-5 text-secondary" />
                  <h2 className="text-2xl font-bold">Popular Posts</h2>
                </div>
                <div className="grid gap-4 sm:grid-cols-3">
                  {popularPosts.map((post, index) => (
                    <Link
                      key={post.slug}
                      href={`/blog/${post.slug}`}
                      className="group block"
                    >
                      <article className="h-full p-5 rounded-lg border border-secondary/30 bg-secondary/5 hover:border-secondary hover:bg-secondary/10 transition-all">
                        <div className="flex items-center gap-2 mb-3">
                          <span className="flex items-center justify-center w-7 h-7 rounded-full bg-secondary/20 text-secondary text-sm font-bold">
                            {index + 1}
                          </span>
                          <span className="flex items-center gap-1.5 text-xs text-secondary font-medium">
                            <Heart className="h-3 w-3 fill-secondary" />
                            {post.likes}
                          </span>
                        </div>
                        <h3 className="text-base font-semibold mb-2 group-hover:text-secondary transition-colors line-clamp-2">
                          {post.title}
                        </h3>
                        <p className="text-sm text-muted-foreground line-clamp-2">
                          {post.description}
                        </p>
                      </article>
                    </Link>
                  ))}
                </div>
              </div>
            )}

            {/* Timeline List */}
            <div className="space-y-6">
              <div className="flex items-center gap-3 mb-8">
                <div className="h-px flex-1 bg-gradient-to-r from-transparent via-border to-transparent" />
                <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                  All Articles
                </span>
                <div className="h-px flex-1 bg-gradient-to-r from-transparent via-border to-transparent" />
              </div>
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
                          className="px-2 py-0.5 rounded-full bg-secondary/10 text-secondary font-medium"
                        >
                          {tag}
                        </span>
                      ))}
                      <span className="ml-auto flex items-center gap-1.5 text-muted-foreground">
                        <Heart className="h-3 w-3" />
                        {likesMap[post.slug] || 0}
                      </span>
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
