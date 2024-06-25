import { bundledLanguages, createHighlighter } from 'shiki/bundle/web';

const codeToHTML = async (
    { code, lang }: { code: string, lang: string }
) => {
    const highlighter = await createHighlighter({
        themes: ['github-light', 'github-dark'],
        langs: [
            ...Object.keys(bundledLanguages),
        ],
    });

    const html = highlighter.codeToHtml(code, {
        lang: lang,
        themes: {
            dark: 'github-dark',
            light: 'github-light',
        },
    });

    return <div dangerouslySetInnerHTML={{ __html: html }} />;
};

export default codeToHTML;
