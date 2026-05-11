export const SITE = {
  website: "https://sudarsh.dev/",
  author: "Sudarsh",
  profile: "",
  desc: "A quiet research notebook on AI systems, inference optimization, reasoning architectures, and multimodal AI.",
  title: "Sudarsh",
  ogImage: "",
  lightAndDarkMode: true,
  postPerIndex: 5,
  postPerPage: 8,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: false,
  showBackButton: false, // show back button in post detail
  editPost: {
    enabled: false,
    text: "Edit page",
    url: "",
  },
  dynamicOgImage: false,
  dir: "ltr", // "rtl" | "auto"
  lang: "en", // html lang code. Set this empty and default will be "en"
  timezone: "Asia/Kolkata", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
} as const;
