"use client";

import { useState, useMemo, useRef, useEffect } from "react";
import Link from "next/link";
import { ArrowLeft, Heart, BookOpen, Clock, Sparkles, Search, Tag, Star, TrendingUp, X, Calendar } from "lucide-react";
import { BlogPostsList } from "@/components/blog-posts-list";
import type { BlogPost } from "@/types/blog";

interface BlogContentProps {
  posts: BlogPost[];
  likesMap: Record<string, number>;
  totalLikes: number;
  totalReadTime: number;
}

interface SlashCommand {
  command: string;
  description: string;
  icon: React.ReactNode;
  example: string;
}

interface Filter {
  type: "tag" | "featured" | "popular" | "year" | "recent";
  value: string;
  label: string;
}

export function BlogContent({ posts, likesMap, totalLikes, totalReadTime }: BlogContentProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [filters, setFilters] = useState<Filter[]>([]);
  const [showCommands, setShowCommands] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const slashCommands: SlashCommand[] = [
    {
      command: "/tag:",
      description: "Filter by tag",
      icon: <Tag className="h-4 w-4" />,
      example: "/tag:mlops",
    },
    {
      command: "/featured",
      description: "Show only featured post",
      icon: <Star className="h-4 w-4" />,
      example: "/featured",
    },
    {
      command: "/popular",
      description: "Show popular posts",
      icon: <TrendingUp className="h-4 w-4" />,
      example: "/popular",
    },
    {
      command: "/year:",
      description: "Filter by year",
      icon: <Calendar className="h-4 w-4" />,
      example: "/year:2024",
    },
    {
      command: "/recent",
      description: "Show posts from last 3 months",
      icon: <Clock className="h-4 w-4" />,
      example: "/recent",
    },
  ];

  // Check if filter type already exists
  const hasFilterType = (type: string) => filters.some(f => f.type === type);

  // Add filter from search query
  const addFilter = () => {
    const query = searchQuery.trim();
    
    if (query.startsWith("/tag:")) {
      const value = query.substring(5).trim();
      if (value && !hasFilterType("tag")) {
        setFilters([...filters, { type: "tag", value, label: `Tag: ${value}` }]);
        setSearchQuery("");
      }
    } else if (query === "/featured" && !hasFilterType("featured")) {
      setFilters([...filters, { type: "featured", value: "", label: "Featured" }]);
      setSearchQuery("");
    } else if (query === "/popular" && !hasFilterType("popular")) {
      setFilters([...filters, { type: "popular", value: "", label: "Popular" }]);
      setSearchQuery("");
    } else if (query.startsWith("/year:")) {
      const value = query.substring(6).trim();
      if (value && !hasFilterType("year")) {
        setFilters([...filters, { type: "year", value, label: `Year: ${value}` }]);
        setSearchQuery("");
      }
    } else if (query === "/recent" && !hasFilterType("recent")) {
      setFilters([...filters, { type: "recent", value: "", label: "Recent" }]);
      setSearchQuery("");
    }
  };

  // Remove filter
  const removeFilter = (index: number) => {
    setFilters(filters.filter((_, i) => i !== index));
  };

  // Handle enter key
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && searchQuery.startsWith("/")) {
      e.preventDefault();
      addFilter();
    }
  };

  // Show command suggestions when user types /
  useEffect(() => {
    if (searchQuery.startsWith("/") && searchQuery.length > 0) {
      setShowCommands(true);
    } else {
      setShowCommands(false);
    }
  }, [searchQuery]);

  const handleCommandClick = (command: string) => {
    setSearchQuery(command);
    setShowCommands(false);
    inputRef.current?.focus();
  };

  const clearSearch = () => {
    setSearchQuery("");
    inputRef.current?.focus();
  };

  const filteredCommands = slashCommands.filter((cmd) => {
    const matchesQuery = cmd.command.toLowerCase().startsWith(searchQuery.toLowerCase());
    const typeUsed = hasFilterType(cmd.command.replace("/", "").replace(":", ""));
    return matchesQuery && !typeUsed;
  });

  // Determine search type for non-slash queries
  const searchType = useMemo(() => {
    if (searchQuery.startsWith("/")) return "command";
    if (searchQuery.trim()) return "search";
    return "none";
  }, [searchQuery]);

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
          
          {/* Search Bar with Slash Commands */}
          <div className="flex-1 max-w-md relative">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <input
                ref={inputRef}
                type="text"
                placeholder="Search or type / for commands..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                className="w-full pl-10 pr-10 py-2 rounded-lg border-2 bg-card shadow-sm text-sm placeholder:text-muted-foreground/60 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary hover:shadow-md transition-all"
              />
              {searchQuery && (
                <button
                  onClick={clearSearch}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>

            {/* Command Suggestions Dropdown */}
            {showCommands && filteredCommands.length > 0 && (
              <div className="absolute top-full mt-2 w-full bg-card border-2 rounded-lg shadow-lg overflow-hidden z-50">
                {filteredCommands.map((cmd) => (
                  <button
                    key={cmd.command}
                    onClick={() => handleCommandClick(cmd.command)}
                    className="w-full px-4 py-3 flex items-start gap-3 hover:bg-muted/50 transition-colors text-left"
                  >
                    <div className="text-primary mt-0.5">{cmd.icon}</div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <code className="text-sm font-mono text-foreground">{cmd.command}</code>
                      </div>
                      <p className="text-xs text-muted-foreground">{cmd.description}</p>
                      <p className="text-xs text-muted-foreground/60 mt-1">
                        Example: <code className="font-mono">{cmd.example}</code>
                      </p>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Active Filters Pills */}
        {filters.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-8">
            {filters.map((filter, index) => (
              <span
                key={index}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-sm font-medium"
              >
                {filter.type === "tag" && <Tag className="h-3.5 w-3.5" />}
                {filter.type === "featured" && <Star className="h-3.5 w-3.5" />}
                {filter.type === "popular" && <TrendingUp className="h-3.5 w-3.5" />}
                {filter.type === "year" && <Calendar className="h-3.5 w-3.5" />}
                {filter.type === "recent" && <Clock className="h-3.5 w-3.5" />}
                {filter.label}
                <button
                  onClick={() => removeFilter(index)}
                  className="hover:text-primary/70 transition-colors"
                  aria-label="Remove filter"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </span>
            ))}
          </div>
        )}

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

      {searchType === "search" && (
        <p className="mb-6 text-sm text-muted-foreground">
          Searching for "{searchQuery}"
        </p>
      )}

      {posts.length > 0 ? (
        <BlogPostsList posts={posts} likesMap={likesMap} filters={filters} searchQuery={searchType === "search" ? searchQuery : ""} />
      ) : (
        <div className="py-20 text-center">
          <p className="text-muted-foreground">No posts yet. Check back soon!</p>
        </div>
      )}
    </div>
  );
}
