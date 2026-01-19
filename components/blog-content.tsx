"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowLeft, Heart, BookOpen, Clock, Sparkles, Search } from "lucide-react";
import { BlogPostsList } from "@/components/blog-posts-list";
import type { BlogPost } from "@/types/blog";

interface BlogContentProps {
  posts: BlogPost[];
  likesMap: Record<string, number>;
  totalLikes: number;
  totalReadTime: number;
}

export function BlogContent({ posts, likesMap, totalLikes, totalReadTime }: BlogContentProps) {
  const [searchQuery, setSearchQuery] = useState("");

  return (
    <div className="relative max-w-4xl mx-auto px-6 py-16 sm:py-24">
      {/* Header */}
      <div className="mb-16">
        <div className="flex items-start justify-between gap-4 mb-12">
          <Link
            href="/"
            className="group inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Home
          </Link>
          
          {/* Search Bar */}
          <div className="flex-1 max-w-md">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search articles..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 rounded-lg border-2 bg-card shadow-sm text-sm placeholder:text-muted-foreground/60 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary hover:shadow-md transition-all"
              />
            </div>
          </div>
        </div>

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

      {searchQuery && (
        <p className="mb-6 text-sm text-muted-foreground">
          Searching for "{searchQuery}"
        </p>
      )}

      {posts.length > 0 ? (
        <BlogPostsList posts={posts} likesMap={likesMap} searchQuery={searchQuery} />
      ) : (
        <div className="py-20 text-center">
          <p className="text-muted-foreground">No posts yet. Check back soon!</p>
        </div>
      )}
    </div>
  );
}
