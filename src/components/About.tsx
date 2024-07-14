"use client";
import { Paragraph } from "@/components/Paragraph";
import Image from "next/image";

import { motion } from "framer-motion";

export default function About() {
    const images = [];
    // const images = [
    //     "https://images.unsplash.com/photo-1692544350322-ac70cfd63614?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxlZGl0b3JpYWwtZmVlZHw1fHx8ZW58MHx8fHx8&auto=format&fit=crop&w=800&q=60",
    //     "https://images.unsplash.com/photo-1692374227159-2d3592f274c9?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxlZGl0b3JpYWwtZmVlZHw4fHx8ZW58MHx8fHx8&auto=format&fit=crop&w=800&q=60",
    //     "https://images.unsplash.com/photo-1692005561659-cdba32d1e4a1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxlZGl0b3JpYWwtZmVlZHwxOHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&q=60",
    //     "https://images.unsplash.com/photo-1692445381633-7999ebc03730?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxlZGl0b3JpYWwtZmVlZHwzM3x8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&q=60",
    // ];
    return (
        <div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-10 my-10">
                {images.map((image, index) => (
                    <motion.div
                        key={image}
                        initial={{
                            opacity: 0,
                            y: -50,
                            rotate: 0,
                        }}
                        animate={{
                            opacity: 1,
                            y: 0,
                            rotate: index % 2 === 0 ? 3 : -3,
                        }}
                        transition={{ duration: 0.2, delay: index * 0.1 }}
                    >
                        <Image
                            src={image}
                            width={200}
                            height={400}
                            alt="about"
                            className="rounded-md object-cover transform rotate-3 shadow-xl block w-full h-40 md:h-60 hover:rotate-0 transition duration-200"
                        />
                    </motion.div>
                ))}
                {/* 
        // <Image
        //   src="https://images.unsplash.com/photo-1692544350322-ac70cfd63614?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxlZGl0b3JpYWwtZmVlZHw1fHx8ZW58MHx8fHx8&auto=format&fit=crop&w=800&q=60"
        //   width={200}
        //   height={400}
        //   alt="about"
        //   className="rounded-md object-cover transform rotate-3 shadow-xl block w-full h-40 md:h-60 hover:rotate-0 transition duration-200"
        // />
        // <Image
        //   src="https://images.unsplash.com/photo-1692374227159-2d3592f274c9?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxlZGl0b3JpYWwtZmVlZHw4fHx8ZW58MHx8fHx8&auto=format&fit=crop&w=800&q=60"
        //   width={200}
        //   height={400}
        //   alt="about"
        //   className="rounded-md object-cover transform -rotate-3 shadow-xl block w-full h-40 md:h-60  hover:rotate-0 transition duration-200"
        // />
        // <Image
        //   src="https://images.unsplash.com/photo-1692005561659-cdba32d1e4a1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxlZGl0b3JpYWwtZmVlZHwxOHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&q=60"
        //   width={200}
        //   height={400}
        //   alt="about"
        //   className="rounded-md object-cover transform rotate-3 shadow-xl block w-full h-40 md:h-60  hover:rotate-0 transition duration-200"
        // />
        // <Image
        //   src="https://images.unsplash.com/photo-1692445381633-7999ebc03730?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxlZGl0b3JpYWwtZmVlZHwzM3x8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&q=60"
        //   width={200}
        //   height={400}
        //   alt="about"
        //   className="rounded-md object-cover transform -rotate-3 shadow-xl block w-full h-40 md:h-60  hover:rotate-0 transition duration-200"
        // /> */}
            </div>

            <div className="max-w-4xl">
                <Paragraph className="mt-4">
                    Hello and welcome to my online portfolio! Let me take you on a journey through my professional life, where my passion for data and technology has led me to exciting places and projects.
                </Paragraph>
                <Paragraph className="mt-4">
                    My story begins at CUNY Lehman College, where I immersed myself in the world of computer science, complemented by a minor in mathematics. This foundation was more than just a degree; it was the beginning of a love affair with data and algorithms that would shape my future. I learned to see the world through the lens of numbers and code, discovering the elegance in patterns and the thrill of problem-solving.
                </Paragraph>
                <Paragraph className="mt-4">
                    After earning my BS, my quest for deeper understanding led me to Columbia University, where I pursued an MA in Statistics with a concentration in Machine Learning. This was where theory met practice, and I delved into the complexities of data analysis, predictive modeling, and machine learning. The rigorous curriculum and cutting-edge research environment honed my skills, preparing me to tackle real-world challenges.
                </Paragraph>
                <Paragraph className="mt-4">
                    As I stepped into the professional world, my technical repertoire expanded to include a diverse set of tools and languages. From the precise data manipulation with Pandas and NumPy to the sophisticated visualizations with Matplotlib and Seaborn, I became adept at turning raw data into insightful narratives. Model design and development became my playground, where TensorFlow, Keras, Scikit-Learn, and other frameworks allowed me to craft intelligent systems capable of learning and adaptation.
                </Paragraph>
                <Paragraph className="mt-4">
                    One of my most cherished projects is Label Hub. This project was a testament to my ability to blend creativity with technical prowess. I designed a system to minimize user search time and reduce system loads, periodically clustering images by tags and replicating them across multiple locations. It was a complex challenge that required innovative thinking and meticulous execution, resulting in a significantly enhanced user experience.
                </Paragraph>
                <Paragraph className="mt-4">
                    Another highlight of my journey is the Real-time Hand Gesture Control Interface for IoT Devices. This project was particularly exciting as it combined machine learning with the burgeoning field of IoT. Using Google’s Mediapipe framework, I developed a cloud-based pipeline and API that enabled IoT devices to understand and respond to hand gestures in real time. It was a fascinating intersection of technology and human interaction, showcasing the potential of machine learning in everyday applications.
                </Paragraph>
                <Paragraph className="mt-4">
                    Throughout my career, I've mastered a variety of programming languages like Python, Java, C++, and R, and have become proficient in cloud platforms such as AWS and GCP. My work with databases like PostgreSQL and BigQuery has further solidified my ability to manage and analyze large datasets effectively.
                </Paragraph>
                <Paragraph className="mt-4">
                    Fluent in both English and Spanish, I thrive in diverse environments, always ready to collaborate and communicate complex ideas clearly. My journey is far from over, and I am constantly seeking new challenges and opportunities to apply my skills in innovative ways.
                </Paragraph>
                <Paragraph className="mt-4">
                    Thank you for taking the time to explore my portfolio. I invite you to dive into my projects and get in touch if you'd like to collaborate or learn more about my work. Let's create something amazing together!
                </Paragraph>
            </div>
        </div>
    );
}
