# Portfolio Configuration Guide

Your portfolio is now configured using YAML files for easy updates. All your personal information, experience, projects, and education can be modified by editing the YAML files in the `data/` directory.

## 📁 Configuration Files

### `data/personal.yaml`
Contains your personal information, social links, about section, and tech stack.

**Fields:**
- `name`: Your display name
- `fullName`: Your full name
- `title`: Your professional title
- `tagline`: Brief description shown on hero section
- `location`: Your current location
- `phone`: Your phone number
- `email`: Your email address
- `social`: GitHub, LinkedIn, and website URLs
- `about.description`: About section description
- `about.skills`: Array of skill cards with icon, title, and description
  - Available icons: `Brain`, `Code2`, `Database`, `Workflow`
- `techStack`: Array of technologies to display

### `data/experience.yaml`
Your work experience history.

**Structure:**
```yaml
experience:
  - company: "Company Name"
    role: "Your Role"
    period: "Start Date - End Date"
    location: "Location"
    description: "Brief description of the role"
    achievements:
      - "Achievement 1"
      - "Achievement 2"
```

### `data/education.yaml`
Your educational background.

**Structure:**
```yaml
education:
  - school: "University Name"
    degree: "Degree Name"
    period: "Start - End Year"
    location: "Location"
    concentration: "Optional concentration/specialization"
    minor: "Optional minor"
    coursework:
      - "Course 1"
      - "Course 2"
```

### `data/projects.yaml`
Your portfolio projects.

**Structure:**
```yaml
projects:
  - title: "Project Name"
    description: "Project description"
    tags:
      - "Technology 1"
      - "Technology 2"
    github: "GitHub URL"
    demo: "Optional demo URL"
```

## 🚀 How to Update Your Portfolio

1. **Edit the YAML files** in the `data/` directory
2. **Save the changes**
3. **Restart the dev server** (if running) or **rebuild** for production

The portfolio will automatically read the updated YAML files and reflect your changes.

## 📝 Blog Posts

Blog posts are still managed in `lib/blog.ts`. You can convert these to YAML as well if needed.

## 💡 Tips

- Keep descriptions concise and impactful
- Use consistent date formats (e.g., "Jan 2025 - Jan 2026")
- Ensure all URLs start with `https://`
- Leave `demo: ""` empty if you don't have a demo link
- Test your changes locally before deploying

## 🔧 Technical Details

- YAML files are parsed using the `js-yaml` library
- Data loading functions are in `lib/data.ts`
- Components automatically fetch data on render
- Type definitions ensure data integrity

## 🎨 Customization

To add more sections or fields:
1. Add the data to the appropriate YAML file
2. Update the TypeScript interfaces in `lib/data.ts`
3. Modify the component to display the new data

---

Need help? Check the existing YAML files for examples or refer to the component source code.
