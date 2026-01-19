"use server";

import { kv } from "@vercel/kv";
import { revalidatePath } from "next/cache";

export async function getLikes(slug: string): Promise<number> {
  try {
    const likes = await kv.get<number>(`blog:likes:${slug}`);
    return likes || 0;
  } catch (error) {
    console.error("Error getting likes:", error);
    return 0;
  }
}

export async function incrementLikes(slug: string): Promise<number> {
  try {
    const newLikes = await kv.incr(`blog:likes:${slug}`);
    revalidatePath(`/blog/${slug}`);
    return newLikes;
  } catch (error) {
    console.error("Error incrementing likes:", error);
    throw new Error("Failed to increment likes");
  }
}
