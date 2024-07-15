import { Container } from "@/components/Container";
import { Heading } from "@/components/Heading";
import { Highlight } from "@/components/Highlight";
import { Paragraph } from "@/components/Paragraph";
import { Products } from "@/components/Products";
import { TechStack } from "@/components/TechStack";

export default function Home() {
  return (
    <Container>
      <span className="text-4xl">👋</span>
      <Heading className="font-black">Hello there! I&apos;m Alix</Heading>
      <Paragraph className="max-w-xl mt-4">
        I&apos;m a data scientist / software engineer that loves{" "}
        <Highlight>building products with data</Highlight> and using science-based approaches
        to find the hidden patterns, generating actionable insights
        that can impact millions of lives
      </Paragraph>
      <Paragraph className="max-w-xl mt-4">
        I&apos;m have more than{" "}
        <Highlight>3 years of experience</Highlight> building scalable and robust products,
        creating groundbreaking analises, and delivering powerfull predictions and insights.  
      </Paragraph>
      <Heading
        as="h2"
        className="font-black text-lg md:text-lg lg:text-lg mt-20 mb-4"
      >
        What I&apos;ve been working on
      </Heading>
      <Products />
      <TechStack />
    </Container>
  );
}
