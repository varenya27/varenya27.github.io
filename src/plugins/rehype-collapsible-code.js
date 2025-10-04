export default function rehypeCollapsibleCode() {
  return (tree) => {
    visit(tree, "element", (node, index, parent) => {
      if (node.tagName === "pre" && parent) {
        parent.children[index] = {
          type: "element",
          tagName: "details",
          properties: { className: ["collapsible-code"] },
          children: [
            {
              type: "element",
              tagName: "summary",
              properties: {},
              children: [{ type: "text", value: "Show Code" }],
            },
            node, // original <pre> stays inside
          ],
        };
      }
    });
  };
}

import { visit } from "unist-util-visit";
