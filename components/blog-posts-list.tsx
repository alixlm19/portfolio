"use client";

import { useMemo } from "react";
import Link from "next/link";
import { Heart, TrendingUp } from "lucide-react";
import type { BlogPost } from "@/types/blog";

interface Filter {
  type: "tag" | "featured" | "popular" | "year" | "recent";
  value: string;
  label: string;
}

interface BlogPostsListProps {
  posts: BlogPost[];
  likesMap: Record<string, number>;
  filters: Filter[];
  searchQuery: string;
}

export function BlogPostsList({ posts, likesMap, filters, searchQuery }: BlogPostsListProps) {

  const featuredPost = posts[0];
  const otherPosts = posts.slice(1);

  // Apply all filters
  const filteredPosts = useMemo(() => {
    let result = otherPosts;

    // Apply search query first
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter((post) => {
        return (
          post.title.toLowerCase().includes(query) ||
          post.description.toLowerCase().includes(query) ||
          post.tags.some((tag) => tag.toLowerCase().includes(query))
        );
      });
    }

    // Apply each filter
    filters.forEach((filter) => {
      if (filter.type === "tag" && filter.value) {
        result = result.filter((post) =>
          post.tags.some((tag) => tag.toLowerCase().includes(filter.value.toLowerCase()))
        );
      }
      
      if (filter.type === "year" && filter.value) {
        result = result.filter((post) => {
          const postYear = new Date(post.date).getFullYear().toString();
          return postYear === filter.value;
        });
      }
      
      if (filter.type === "recent") {
        const threeMonthsAgo = new Date();
        threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3);
        result = result.filter((post) => {
          const postDate = new Date(post.date);
          return postDate >= threeMonthsAgo;
        });
      }
    });

    return result;
  }, [otherPosts, filters, searchQuery]);

  // Get popular posts (top 3 by likes, excluding featured)
  const popularPosts = posts
    .map((post) => ({
      ...post,
      likes: likesMap[post.slug] || 0,
    }))
    .filter((post) => post.slug !== featuredPost?.slug)
    .sort((a, b) => b.likes - a.likes)
    .slice(0, 3);

  // Determine what to show based on filters
  const hasFeaturedFilter = filters.some(f => f.type === "featured");
  const hasPopularFilter = filters.some(f => f.type === "popular");
  
  const showFeatured = hasFeaturedFilter || (filters.length === 0 && !searchQuery);
  const showPopular = hasPopularFilter || (filters.length === 0 && !searchQuery);
  const showAllPosts = !hasFeaturedFilter && !hasPopularFilter;

  return (
    <>
      {/* Featured Post */}
      {featuredPost && showFeatured && (
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
      {popularPosts.length > 0 && showPopular && (
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
      {showAllPosts && (
        <div className="space-y-6">
          {filters.length === 0 && !searchQuery && (
            <div className="flex items-center gap-3 mb-8">
              <div className="h-px flex-1 bg-gradient-to-r from-transparent via-border to-transparent" />
              <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                All Articles
              </span>
              <div className="h-px flex-1 bg-gradient-to-r from-transparent via-border to-transparent" />
            </div>
          )}
          {filteredPosts.length > 0 ? (
          filteredPosts.map((post) => (
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
          ))
        ) : (
          <div className="py-12 text-center">
            <p className="text-muted-foreground">
              No articles found{(searchQuery || filters.length > 0) && " matching your criteria"}
            </p>
          </div>
        )}
        </div>
      )}
    </>
  );
}
