import { getBlogPosts } from "@/lib/blog";
import { getLikes } from "@/app/actions/likes";
import { FunCursor } from "@/components/fun-cursor";
import { BlogContent } from "@/components/blog-content";
import { Navbar } from "@/components/navbar";
import { ClickEffects } from "@/components/click-effects";

export default async function BlogPage() {
  const posts = getBlogPosts();

  // Get like counts for all posts
  const likeCounts = await Promise.all(
    posts.map((post) => getLikes(post.slug))
  );
  const likesMap = Object.fromEntries(
    posts.map((post, i) => [post.slug, likeCounts[i]])
  );

  // Calculate stats
  const totalLikes = Object.values(likesMap).reduce((sum, likes) => sum + likes, 0);
  const totalReadTime = posts.reduce((sum, post) => {
    const minutes = parseInt(post.readTime?.match(/\d+/)?.[0] || "0");
    return sum + minutes;
  }, 0);

  return (
    <>
      <FunCursor />
      <ClickEffects />
      <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary/5">
        <Navbar />
        {/* Decorative elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 right-20 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-40 left-20 w-96 h-96 bg-secondary/5 rounded-full blur-3xl" />
      </div>

      <BlogContent 
        posts={posts} 
        likesMap={likesMap}
        totalLikes={totalLikes}
        totalReadTime={totalReadTime}
      />
      </div>
    </>
  );
}
