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
  "cuda",
  "gpu",
  "kernels",
]);

export const hasNotebookTag = (tags: string[]) =>
  tags.some(tag => NOTEBOOK_TAGS.has(tag.toLowerCase()));

// Tags that route a post into one of the specific sections below instead of
// the default /blog feed. Keep this in sync with random.astro / papers.astro /
// lab.astro / notes.astro / projects.astro's own tag checks.
const SECTION_TAGS = [
  ["random", "books", "misc"],
  ["paper", "papers", "reproduction"],
  ["lab", "experiment", "experiments"],
  ["note", "notes"],
  ["project", "projects"],
];

export const hasSectionTag = (tags: string[]) =>
  SECTION_TAGS.some(section =>
    tags.some(tag => section.includes(tag.toLowerCase()))
  );

const notebookPostFilter = ({ data }: CollectionEntry<"blog">) =>
  hasNotebookTag(data.tags);

export default notebookPostFilter;
