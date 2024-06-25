import CodeBlock from "@/components/CodeBlock";

export const products = [
    {
        href: "https://aceternity.com",
        title: "Statistical Insights into Diversified Portfolio Analysis: Unveiling Financial Trends and Strategies",
        description:
            "A design and development studio that focuses on building quality apps.",
        thumbnail: "/images/sidefolio-aceternity.png",
        images: [
            "/images/sidefolio-aceternity.png",
            "/images/sidefolio-aceternity-2.png",
        ],
        stack: ["Nextjs", "Tailwindcss"],
        slug: "aceternity",
        content: (
            <div>
                <p>
                    Sit eiusmod ex mollit sit quis ad deserunt. Sint aliqua aliqua ullamco
                    dolore nulla amet tempor sunt est ipsum. Dolor laborum eiusmod
                    cupidatat consectetur velit ipsum. Deserunt nisi in culpa laboris
                    cupidatat elit velit aute mollit nisi. Officia ad exercitation laboris
                    non cupidatat duis esse velit ut culpa et.{" "}
                </p>
                <p>
                    Exercitation pariatur enim occaecat adipisicing nostrud adipisicing
                    Lorem tempor ullamco exercitation quis et dolor sint. Adipisicing sunt
                    sit aute fugiat incididunt nostrud consequat proident fugiat id.
                    Officia aliquip laborum labore eu culpa dolor reprehenderit eu ex enim
                    reprehenderit. Cillum Lorem veniam eu magna exercitation.
                    Reprehenderit adipisicing minim et officia enim et veniam Lorem
                    excepteur velit adipisicing et Lorem magna.
                </p>{" "}
            </div>
        ),
    },
    {
        href: "https://algochurn.com",
        title: "English-Spanish Translation with Transformers",
        description:
            "This project develops an English-Spanish translation model using transformer architectures. By training on a large corpus of parallel texts, the model achieves high accuracy and fluency. It involves data preprocessing, model training, and evaluation, showcasing the power of transformers in enhancing multilingual communication.",
        thumbnail: "/images/transformer-decoding.gif",
        images: [
            "/images/transformer.png",
            "/images/transformer-decoding.gif",
        ],
        stack: ["Python", "Tensorflow", "Machine Learning", "Transformers", "Autoencoders", "Embeddings"],
        slug: "transformers",
        content: (
            <div>
                <p>
                    Sit eiusmod ex mollit sit quis ad deserunt. Sint aliqua aliqua ullamco
                    dolore nulla amet tempor sunt est ipsum. Dolor laborum eiusmod
                    cupidatat consectetur velit ipsum. Deserunt nisi in culpa laboris
                    cupidatat elit velit aute mollit nisi. Officia ad exercitation laboris
                    non cupidatat duis esse velit ut culpa et.{" "}
                </p>
                <code>
                    Exercitation pariatur enim occaecat adipisicing nostrud adipisicing
                    Lorem tempor ullamco exercitation quis et dolor sint. Adipisicing sunt
                    sit aute fugiat incididunt nostrud consequat proident fugiat id.
                    Officia aliquip laborum labore eu culpa dolor reprehenderit eu ex enim
                    reprehenderit. Cillum Lorem veniam eu magna exercitation.
                    Reprehenderit adipisicing minim et officia enim et veniam Lorem
                    excepteur velit adipisicing et Lorem magna.
                </code>{" "}
                <CodeBlock code="console.log('a')" lang="javascript"/> 
            </div>
        ),
    },
    {
        href: "https://www.alixleon.me/",
        title: "Statistical Insights into Diversified Portfolio Analysis: Unveiling Financial Trends and Strategies",
        description:
            "Financial analysis on big-tech companies stocks until 2021 using statistical financial methods.",
        thumbnail: "/images/stock_price_over_time.png",
        images: [
            "/images/stock_price_over_time.png",
            "/images/stock_price_over_time_2.png",
            "/images/pca_1.png",
            "/images/pca_2.png",
            "/images/pca_3.png",
        ],
        stack: ["R", "Portoflio Theory", "Hypothesis Testing", "Risk Management"],
        slug: "statistical-insights-diversified-portfolio-analysis",
        content: (
            <div>
                <p><strong>Introduction:</strong> The project centered on analyzing a diverse portfolio comprising stocks from the technology sector and various other industries. This approach aimed to evaluate the risk-return profile of each asset using advanced statistical techniques, providing valuable insights into their behavior over time.</p>

                <p><strong>Key Findings:</strong></p>
                <p>1. <strong>Non-Normal Distributions:</strong> One of the prominent discoveries was that the majority of assets in the portfolio did not adhere to normal distributions. Instead, they followed Skewed Standardized-t distributions, with some fitting a Generalized Error Distribution pattern.</p>
                <p>2. <strong>Performance Insights:</strong> Among the analyzed assets, Microsoft emerged as a standout performer with consistently high returns and a robust reward-to-risk ratio. This highlighted Microsoft's resilience and profitability within the portfolio context.</p>
                <p>3. <strong>Asset Independence:</strong> A significant observation was the near-zero covariance among many assets. This finding indicated a level of independence between assets, underscoring the importance of diversification in minimizing portfolio risk effectively.</p>
                <p>4. <strong>Copula Analysis:</strong> Employing t-Copula modeling, the study revealed an increased likelihood of joint extreme events among asset values. This aspect emphasized the necessity of robust risk management strategies to mitigate potential losses during market downturns.</p>
                <p>5. <strong>Portfolio Construction:</strong> Various portfolio strategies were explored, including the Minimum Variance Portfolio (MVP) and Efficient Portfolio, aiming to optimize returns while managing risk levels prudently. These strategies underscored the application of statistical methods in constructing balanced and efficient investment portfolios.</p>

                <p><strong>Methodology:</strong></p>
                <p>- <strong>Data Collection:</strong> Historical monthly prices and returns data spanning a five-year period were collected for stocks such as AMD, Microsoft, Apple Inc., Meta Platforms Inc., and others.</p>
                <p>- <strong>Statistical Techniques:</strong> The project employed a range of statistical tools including Q-Q plots, boxplots, and hypothesis testing to validate assumptions and derive meaningful insights from the data.</p>
                <p>- <strong>Risk Management:</strong> Techniques like Value at Risk (VaR) and Expected Shortfall (ES) were utilized to quantify and manage potential financial risks associated with the portfolio holdings.</p>

                <p><strong>Conclusion:</strong> The project exemplifies the critical role of statistical methods in financial analysis, providing actionable insights into asset performance, risk assessment, and portfolio optimization. These insights are invaluable for investors seeking to navigate the complexities of modern financial markets with confidence and precision.</p>

                <p><strong>Future Directions:</strong> Future research endeavors could expand on this analysis by incorporating a broader range of assets or integrating machine learning algorithms for predictive modeling. Such advancements could further enhance portfolio management strategies, offering deeper insights and improved decision-making capabilities.</p>

                <p><strong>Conclusion:</strong> In summary, the Statistical Methods In Finance final project offers a comprehensive exploration of portfolio dynamics and performance metrics, demonstrating the power of statistical rigor in illuminating financial trends and informing investment strategies. By leveraging these insights, investors can effectively manage risks and capitalize on opportunities in today's dynamic financial landscape.</p>
            </div>
        ),
    },
    {
        href: "https://alixleon.shinyapps.io/reu-project/",
        title: "Poor State, Rich State",
        description:
            "The Poor State, Rich State web app is an open-source tool built with R and Shiny that provides interactive visualizations and analyses of poverty data across different states.",
        thumbnail: "/images/poor_state-rich_state.jpg",
        images: [
            "/images/poor_state-rich_state.jpg",
        ],
        stack: ["R", "Shiny", "HTML", "CSS"],
        slug: "poorstaterichstate",
        content: (
            <div className="pt-4">
                <h2>What Does the Poor State, Rich State Web App Do?</h2>
                <p>
                    At its core, the <strong>Poor State, Rich State</strong> web app is a comprehensive tool for displaying and interpreting poverty statistics. Leveraging a robust dataset on poverty levels across various states, the app transforms this complex information into a format that is both easy to understand and visually appealing. Here are some of the key features and functionalities:
                </p>

                <ul>
                    <li><strong>Interactive Visualizations:</strong> The app features a variety of interactive charts and graphs, allowing users to explore poverty data dynamically. Users can filter data by state, time period, and other relevant criteria to gain deeper insights.</li>
                    <li><strong>User-Friendly Interface:</strong> The design prioritizes accessibility and ease of use, making it suitable for a wide range of users, including students, researchers, policymakers, and professionals in the social sciences.</li>
                    <li><strong>Data Comparison:</strong> Users can compare poverty statistics between states, track changes over time, and identify trends and patterns that may not be immediately apparent from raw data alone.</li>
                    <li><strong>Custom Reports:</strong> The app allows users to generate custom reports based on their selected criteria. These reports can be downloaded and used for academic research, policy making, or personal analysis.</li>
                    <li><strong>Educational Resource:</strong> As an educational tool, the app is a valuable resource for students learning about socio-economic issues. It provides a hands-on way to engage with real-world data and understand the impact of poverty across different regions.</li>
                    <li><strong>Open Source:</strong> Being an open-source project, the <strong>Poor State, Rich State</strong> web app encourages community contributions and collaboration. Developers and data scientists can contribute to the project, suggest improvements, and use the app as a foundation for their own projects.</li>
                </ul>

                <p>
                    The <strong>Poor State, Rich State</strong> web app exemplifies how data visualization and modern web technologies can be harnessed to make important socio-economic data accessible and actionable. Whether you are a student looking to enhance your understanding of poverty issues or a professional seeking reliable data for analysis, this app provides a powerful, user-friendly platform to meet your needs.
                </p>

                <p>
                    Explore the <strong>Poor State, Rich State</strong> web app today and discover how data can illuminate the challenges and disparities faced by states across the nation.
                </p>
            </div>
        ),
    },
];
