import type { BlogPost } from "@/types/blog";

// Calculate reading time based on word count (200 words per minute)
function calculateReadTime(content: string): string {
  const wordsPerMinute = 200;
  const words = content.trim().split(/\s+/).length;
  const minutes = Math.ceil(words / wordsPerMinute);
  return `${minutes} min read`;
}

export const blogPosts: BlogPost[] = [
  {
    slug: "introduction-to-mlops",
    title: "Introduction to MLOps: Bridging ML and Production",
    description:
      "Learn the fundamentals of MLOps and how to deploy machine learning models effectively in production environments.",
    date: "2024-01-15",
    tags: ["MLOps", "Machine Learning", "DevOps", "Python"],
    author: "Alix Leon",
    content: `
# Introduction to MLOps: Bridging ML and Production

MLOps (Machine Learning Operations) is a set of practices that combines Machine Learning, DevOps, and Data Engineering to deploy and maintain ML systems in production reliably and efficiently.

## Why MLOps Matters

Building a machine learning model is just the beginning. The real challenge lies in deploying, monitoring, and maintaining these models in production. MLOps addresses this gap by providing:

- **Reproducibility**: Ensure experiments and models can be recreated
- **Scalability**: Handle increasing data and prediction volumes
- **Monitoring**: Track model performance and data drift
- **Automation**: Streamline the ML workflow

## Key Components

1. **Version Control**: Track code, data, and models
2. **CI/CD Pipelines**: Automate testing and deployment
3. **Model Registry**: Centralized model storage and versioning
4. **Monitoring**: Real-time performance tracking

## Getting Started

Start with small, incremental changes to your ML workflow. Focus on version control and basic automation before tackling complex orchestration.

Stay tuned for more deep dives into specific MLOps tools and practices!
    `,
  },
  {
    slug: "building-scalable-apis-fastapi",
    title: "Building Scalable ML APIs with FastAPI",
    description:
      "A comprehensive guide to creating high-performance machine learning APIs using FastAPI and Python.",
    date: "2024-01-10",
    tags: ["FastAPI", "Python", "API", "Machine Learning"],
    author: "Alix Leon",
    content: `
# Building Scalable ML APIs with FastAPI

FastAPI is a modern, fast web framework for building APIs with Python. It's perfect for serving machine learning models in production.

## Why FastAPI?

- **Fast**: High performance, on par with NodeJS and Go
- **Type Safety**: Automatic validation with Python type hints
- **Auto Documentation**: Interactive API docs with Swagger UI
- **Async Support**: Native async/await for concurrent requests

## Example: ML Model API

\`\`\`python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict(request: PredictionRequest):
    prediction = model.predict([request.features])
    return {"prediction": prediction.tolist()}
\`\`\`

## Best Practices

1. Use dependency injection for models
2. Implement proper error handling
3. Add request validation
4. Enable CORS for web clients
5. Use async for I/O-bound operations

FastAPI makes it easy to create production-ready ML APIs with minimal code!
    `,
  },
  {
    slug: "deep-learning-optimization-techniques",
    title: "Deep Learning Optimization Techniques",
    description:
      "Explore advanced optimization strategies to improve deep learning model training and performance.",
    date: "2024-01-05",
    tags: ["Deep Learning", "PyTorch", "Neural Networks", "Optimization"],
    author: "Alix Leon",
    content: `
# Deep Learning Optimization Techniques

Training deep neural networks efficiently requires understanding various optimization techniques. Let's explore some key strategies.

## Learning Rate Scheduling

Dynamic learning rate adjustment can significantly improve training:

- **Step Decay**: Reduce LR at specific intervals
- **Cosine Annealing**: Smooth LR decrease following cosine curve
- **OneCycleLR**: Single cycle with warm-up and cool-down

## Gradient Clipping

Prevent exploding gradients by clipping:

\`\`\`python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
\`\`\`

## Mixed Precision Training

Speed up training with lower precision:

- Faster computation on modern GPUs
- Reduced memory usage
- Maintained model accuracy

## Batch Size Strategies

- Larger batches: More stable gradients, better hardware utilization
- Smaller batches: Better generalization, less memory
- Gradient accumulation: Simulate larger batches

## Regularization

- Dropout: Prevent overfitting
- Weight decay: L2 regularization
- Batch normalization: Stabilize training

Combining these techniques can dramatically improve your model's training efficiency and final performance!
    `,
  },
];

export function getBlogPosts(): BlogPost[] {
  return blogPosts
    .map((post) => ({
      ...post,
      readTime: calculateReadTime(post.content),
    }))
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
}

export function getBlogPost(slug: string): BlogPost | undefined {
  const post = blogPosts.find((post) => post.slug === slug);
  if (!post) return undefined;
  return {
    ...post,
    readTime: calculateReadTime(post.content),
  };
}
