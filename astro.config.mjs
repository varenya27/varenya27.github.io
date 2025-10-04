// @ts-check

import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { defineConfig } from 'astro/config';
import remarkMath from "remark-math";
import rehypeMathjax from "rehype-mathjax";
import rehypeCollapsibleCode from './src/plugins/rehype-collapsible-code.js';

// https://astro.build/config
export default defineConfig({
	site: 'https://varenya27.github.io',
	integrations: [mdx(), sitemap()],
	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeCollapsibleCode],
		rehypePlugins: [rehypeMathjax],
	},
});
