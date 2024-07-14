'use client';

import { bundledLanguages, createHighlighter, Highlighter } from 'shiki/bundle/web';

const langs = [...Object.keys(bundledLanguages)]
let highlighter: Highlighter | undefined = undefined;
(async () => {
    highlighter = await createHighlighter({
        themes: ['github-light', 'github-dark'],
        langs: langs
    })
})();

const codeToHTML = (
    { code, lang = "python" }: { code: string, lang: string }
) => {

    const html = highlighter ?
        highlighter.codeToHtml(code, {
            lang: lang,
            themes: {
                dark: 'github-dark',
                light: 'github-light',
            },
        }) :
        `<p>${code}</p>`;

    return <div dangerouslySetInnerHTML={{ __html: html }} />;
};

export default codeToHTML;
