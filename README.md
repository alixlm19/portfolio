# Portfolio Website

A modern, interactive portfolio website built with Next.js 16, featuring a blog with advanced filtering, interactive animations, and a clean, professional design.

## Features

### Interactive Portfolio
- **Responsive Design** - Fully responsive across all devices
- **Dark/Light Mode** - Theme switcher with persistent preferences
- **Smooth Animations** - GSAP and Framer Motion animations throughout
- **Custom Cursor** - Interactive cursor effects on desktop
- **Animated Gradient Orbs** - Subtle, professional background animations
- **Click Effects** - Particle bursts and ripple effects on interactions
- **Scroll Progress** - Visual indicator of page scroll progress

### Dynamic Blog
- **Markdown Support** - Write posts in markdown with GitHub-flavored markdown
- **Syntax Highlighting** - Code blocks with syntax highlighting
- **Table of Contents** - Auto-generated TOC with active section tracking
- **Share Functionality** - Copy link, LinkedIn, and X (Twitter) sharing
- **Like System** - Persistent like tracking with Vercel KV/Upstash Redis
- **Reading Time** - Automatic reading time calculation
- **Popular Posts** - Displays top posts by likes

### Advanced Search & Filtering
- **Slash Commands** - Discord/Teams-style command system
  - `/tag:name` - Filter by tag
  - `/year:YYYY` - Filter by year
  - `/recent` - Posts from last 3 months
  - `/featured` - Show featured post only
  - `/popular` - Show popular posts
- **Multi-Filter Support** - Apply multiple filters simultaneously
- **Real-time Search** - Instant search across titles, descriptions, and tags
- **Filter Pills** - Visual pills for active filters with remove option

### Additional Features
- **Code Copy Buttons** - One-click copy for code blocks
- **Scroll Timeline** - Visual progress indicator on blog posts
- **Navigation** - Sticky navbar with smooth scrolling
- **SEO Optimized** - Proper meta tags and structured data
- **Performance** - Server-side rendering and static generation

## Getting Started

### Prerequisites
- Node.js 20+ or Bun
- Vercel account (for KV database)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/portfolio.git
cd portfolio
```

2. Install dependencies:
```bash
bun install
# or
npm install
```

3. Set up environment variables:
```bash
cp .env.example .env.local
```

Add your Vercel KV credentials:
```env
KV_URL="your_kv_url"
KV_REST_API_URL="your_kv_rest_api_url"
KV_REST_API_TOKEN="your_kv_rest_api_token"
KV_REST_API_READ_ONLY_TOKEN="your_kv_rest_api_read_only_token"
```

4. Run the development server:
```bash
bun dev
# or
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Content Management

### Adding Blog Posts
1. Create a new `.md` file in `content/posts/`
2. Add frontmatter:
```markdown
---
title: "Your Post Title"
date: "2024-01-01"
description: "Brief description"
featured: true
tags: ["tag1", "tag2"]
readTime: "5 min read"
---

Your content here...
```

### Updating Personal Info
Edit `data/personal-info.yaml` to update your:
- Bio and skills
- Experience
- Education
- Projects
- Contact information

## Tech Stack

- **Framework**: Next.js 16.1.3 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS 4
- **Animations**: GSAP 3.14.2, Framer Motion 12.27.0
- **Markdown**: ReactMarkdown with rehype-highlight
- **Database**: Vercel KV (Upstash Redis)
- **Icons**: Lucide React
- **Deployment**: Vercel

## Project Structure

```
portfolio/
├── app/                    # Next.js app directory
│   ├── blog/              # Blog pages
│   ├── actions/           # Server actions
│   └── page.tsx           # Home page
├── components/            # React components
├── content/              # Blog posts (markdown)
├── data/                 # Personal info (YAML)
├── lib/                  # Utilities and helpers
└── public/              # Static assets
```

## Customization

### Colors
Edit `app/globals.css` to customize the color scheme:
- Primary: Orange (hue 41.116)
- Secondary: Blue (hue 252)

### Fonts
The project uses Geist font family. Modify in `app/layout.tsx`.

### Animations
Adjust animation parameters in component files:
- `components/fun-cursor.tsx` - Cursor effects
- `components/animated-gradient-orbs.tsx` - Background orbs
- `components/click-effects.tsx` - Click animations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this code for personal or commercial purposes.

## Acknowledgments

- Built with [Next.js](https://nextjs.org/)
- Animations powered by [GSAP](https://greensock.com/gsap/) and [Framer Motion](https://www.framer.com/motion/)
- Icons from [Lucide](https://lucide.dev/)
- Syntax highlighting with [rehype-highlight](https://github.com/rehypejs/rehype-highlight)

---

Made with Next.js and TypeScript
