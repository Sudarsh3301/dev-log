import type { CollectionEntry } from "astro:content";

const NOTEBOOK_TAGS = new Set([
  "ai",
  "ml",
  "systems",
  "inference",
  "optimization",
  "reasoning",
  "architecture",
  "architectures",
  "multimodal",
  "research",
  "latent",
  "benchmark",
  "benchmarks",
  "note",
  "notes",
  "lab",
  "experiment",
  "experiments",
  "project",
  "projects",
  "paper",
  "papers",
  "reproduction",
  "random",
  "books",
]);

export const hasNotebookTag = (tags: string[]) =>
  tags.some(tag => NOTEBOOK_TAGS.has(tag.toLowerCase()));

const notebookPostFilter = ({ data }: CollectionEntry<"blog">) =>
  hasNotebookTag(data.tags);

export default notebookPostFilter;
