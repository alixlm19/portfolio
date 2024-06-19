import Image from "next/image";
import React from "react";
import { Heading } from "./Heading";
import { twMerge } from "tailwind-merge";

export const TechStack = () => {
    const stack = [
        {
            title: "Python",
            src: "/images/logos/python.png",

            className: "h-10 w-8",
        },
        {
            title: "C++",
            src: "/images/logos/cplusplus.png",

            className: "h-10 w-12",
        },
        {
            title: "Rust",
            src: "/images/logos/rust.png",

            className: "h-10 w-12",
        },
        {
            title: "Tensorflow",
            src: "/images/logos/tensorflow.png",

            className: "h-10 w-12",
        },
        {
            title: "PyTorch",
            src: "/images/logos/pytorch.png",

            className: "h-10 w-24",
        },
        {
            title: "PostgreSQL",
            src: "/images/logos/postgresql.png",

            className: "h-10 w-14",
        },
        {
            title: "AWS",
            src: "/images/logos/aws.webp",

            className: "h-10 w-10",
        },
        {
            title: "Tableau",
            src: "/images/logos/tableau.png",

            className: "h-10 w-28",
        },
        // {
        //     title: "Tailwind",
        //     src: "/images/logos/tailwind.png",
        //
        //     className: "h-10 w-24",
        // },
        // {
        //     title: "Vercel",
        //     src: "/images/logos/vercel.png",
        //
        //     className: "h-10 w-24",
        // },
    ];
    return (
        <div>
            <Heading
                as="h2"
                className="font-black text-lg md:text-lg lg:text-lg mt-20 mb-4"
            >
                Tech Stack
            </Heading>
            <div className="flex flex-wrap">
                {stack.map((item) => (
                    <Image
                        src={item.src}
                        key={item.src}
                        width={`200`}
                        height={`200`}
                        alt={item.title}
                        className={twMerge("object-contain mr-4 mb-4", item.className)}
                    />
                ))}
            </div>
        </div>
    );
};
