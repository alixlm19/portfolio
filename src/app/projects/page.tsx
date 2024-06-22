import { Container } from "@/components/Container";
import { Heading } from "@/components/Heading";
import { Highlight } from "@/components/Highlight";
import { Paragraph } from "@/components/Paragraph";
import { Products } from "@/components/Products";
import { Metadata } from "next";
import Image from "next/image";

export const metadata: Metadata = {
    title: "Projects | Alix Leon",
    description:
        "Alix Leon is a data scientist / software engineer that loves building products with data and using science-based approaches to find the hidden patterns, generating actionable insights that can impact millions of lives",
};

export default function Projects() {
    return (
        <Container>
            <span className="text-4xl">⚡</span>
            <Heading className="font-black mb-10">
                {" "}
                What I&apos;ve been working on
            </Heading>

            <Products />
        </Container>
    );
}
