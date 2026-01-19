"use client";

import { useState, useEffect, useTransition } from "react";
import { Heart } from "lucide-react";
import { incrementLikes } from "@/app/actions/likes";

interface LikeButtonProps {
  slug: string;
  initialLikes: number;
}

export function LikeButton({ slug, initialLikes }: LikeButtonProps) {
  const [likes, setLikes] = useState(initialLikes);
  const [isLiked, setIsLiked] = useState(false);
  const [isPending, startTransition] = useTransition();
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    // Check if user has already liked this post
    const likedPosts = JSON.parse(localStorage.getItem("likedPosts") || "[]");
    setIsLiked(likedPosts.includes(slug));
  }, [slug]);

  const handleLike = () => {
    if (isLiked || isPending) return;

    // Optimistic update
    setLikes(likes + 1);
    setIsLiked(true);
    setIsAnimating(true);

    // Save to localStorage
    const likedPosts = JSON.parse(localStorage.getItem("likedPosts") || "[]");
    likedPosts.push(slug);
    localStorage.setItem("likedPosts", JSON.stringify(likedPosts));

    // Server action
    startTransition(async () => {
      try {
        const newLikes = await incrementLikes(slug);
        setLikes(newLikes);
      } catch (error) {
        // Rollback on error
        setLikes(likes);
        setIsLiked(false);
        const likedPosts = JSON.parse(localStorage.getItem("likedPosts") || "[]");
        const updated = likedPosts.filter((s: string) => s !== slug);
        localStorage.setItem("likedPosts", JSON.stringify(updated));
      }
    });

    setTimeout(() => setIsAnimating(false), 600);
  };

  return (
    <button
      onClick={handleLike}
      disabled={isLiked || isPending}
      className={`group relative inline-flex items-center gap-3 px-6 py-3 rounded-full border transition-all duration-300 ${
        isLiked
          ? "bg-primary/10 border-primary text-primary cursor-default"
          : "bg-background border-border hover:border-primary hover:bg-primary/5"
      }`}
      aria-label={isLiked ? "Already liked" : "Like this post"}
    >
      <Heart
        className={`h-5 w-5 transition-all duration-300 ${
          isLiked ? "fill-primary" : "group-hover:fill-primary/20"
        } ${isAnimating ? "animate-bounce" : ""}`}
      />
      <span className="font-medium tabular-nums">
        {likes} {likes === 1 ? "like" : "likes"}
      </span>
      {isLiked && (
        <span className="absolute -top-8 left-1/2 -translate-x-1/2 text-xs text-primary font-medium whitespace-nowrap opacity-0 animate-fade-out">
          Thanks! ❤️
        </span>
      )}
    </button>
  );
}
